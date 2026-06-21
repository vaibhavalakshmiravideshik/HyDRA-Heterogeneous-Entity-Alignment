"""
HyDRA Stage 1 — Unsupervised Relation Family Induction
=======================================================
Trains RotatE on each KG's relation triples, clusters relations
into families via KMeans, detects cross-KG anchor relations via
LLM, and aligns embedding spaces via Procrustes + OT.
"""

import numpy as np
import torch
import ot
from collections import defaultdict
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline as pykeen_pipeline

from utils.helpers import (save, load, exists, resolve,
                            llm_call, parse_json_response)


# ── RotatE training ───────────────────────────────────────────────────────────

def train_rotate(rel_triples: list, name: str, cache_dir: str,
                 dim: int = 64, epochs: int = 50,
                 k: int = 4, device: str = "cuda") -> dict:
    """
    Train RotatE on relation triples, cluster relations into k families.

    Returns dict with keys:
        fam_map   : {relation_uri: family_id}
        fam_names : {family_id: representative_label}
        embs      : relation embeddings (n_rels, 2*dim)
        rels      : list of relation URIs
        factory   : PyKEEN TriplesFactory
    """
    cache_key = Path(cache_dir) / f"rotate_{name}.pkl"
    if exists(cache_key):
        print(f"  {name}: loaded from cache")
        return load(cache_key)

    print(f"  Training RotatE on {name} ({len(rel_triples):,} triples)...")

    triples_arr = np.array(
        [(h, r, t) for h, r, t in rel_triples], dtype=str)
    factory = TriplesFactory.from_labeled_triples(triples_arr)
    train, test, valid = factory.split([0.8, 0.1, 0.1], random_state=42)

    result = pykeen_pipeline(
        training=train, testing=test, validation=valid,
        model="RotatE",
        model_kwargs=dict(embedding_dim=dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=4096),
        optimizer_kwargs=dict(lr=1e-3),
        random_seed=42, device=device,
    )
    print(f"  {name} final loss: {result.losses[-1]:.4f}")

    # Extract relation embeddings
    with torch.no_grad():
        embs = result.model.relation_representations[0](
            indices=torch.arange(factory.num_relations)
        ).cpu().numpy()
    if np.iscomplexobj(embs):
        embs = np.concatenate([embs.real, embs.imag], axis=1)
    embs = normalize(embs)

    rels = list(factory.relation_to_id.keys())
    k_actual = min(k, len(rels))
    if len(rels) <= k_actual:
        labels = list(range(len(rels)))
    else:
        labels = KMeans(
            n_clusters=k_actual, random_state=42, n_init=10
        ).fit_predict(embs)

    fam_map = {r: int(l) for r, l in zip(rels, labels)}

    fam2rels = defaultdict(list)
    for rel, fam in fam_map.items():
        short = rel.split("#")[-1].split("/")[-1]
        fam2rels[fam].append(short)
    fam_names = {f: min(rs, key=len) for f, rs in fam2rels.items()}

    out = {
        "fam_map":   fam_map,
        "fam_names": fam_names,
        "embs":      embs,
        "rels":      rels,
        "factory":   factory,
    }
    save(out, cache_key)
    return out


# ── Anchor detection ──────────────────────────────────────────────────────────

def detect_anchors(rels_a: list, rels_b: list,
                   ds_name: str, cache_dir: str,
                   client, model: str) -> list:
    """
    Use LLM to identify cross-KG semantically equivalent relations.
    Returns list of short-name strings present in both KGs.
    """
    cache_key = Path(cache_dir) / f"anchors_{ds_name}.pkl"
    if exists(cache_key):
        return load(cache_key)

    def short(uri):
        return uri.split("#")[-1].split("/")[-1]

    shorts_a = [short(r) for r in rels_a]
    shorts_b = [short(r) for r in rels_b]

    prompt = f"""You are analyzing relation types from two biomedical KGs.
Identify relations that are semantically equivalent across both KGs.
These anchor the embedding space alignment.

KG-A relations: {shorts_a}

KG-B relations: {shorts_b}

Return a JSON list of short name strings that appear (exactly or
approximately) in both KGs with the same meaning. High-confidence only.
Only output a valid JSON list of strings, nothing else.

Example: ["subClassOf", "part_of"]"""

    try:
        raw_resp = llm_call(client, model, prompt, max_tokens=300,
                            temperature=0.1)
        anchors = parse_json_response(raw_resp)
        if not isinstance(anchors, list):
            anchors = []
        print(f"  LLM anchors ({ds_name}): {anchors}")
    except Exception as e:
        print(f"  LLM fallback for {ds_name}: {e}")
        anchors = list(set(shorts_a) & set(shorts_b))

    save(anchors, cache_key)
    return anchors


# ── Procrustes + OT family alignment ──────────────────────────────────────────

def align_families(rot_left: dict, rot_right: dict,
                   anchors: list, ds_name: str,
                   cache_dir: str) -> np.ndarray:
    """
    Align relation family embeddings across two KGs via
    Procrustes rotation (anchor-guided) + Optimal Transport.

    Returns mapping array: family_left[i] -> family_right[mapping[i]]
    """
    cache_key = Path(cache_dir) / f"family_mapping_{ds_name}.pkl"
    if exists(cache_key):
        return load(cache_key)

    embs_a = rot_left["embs"]
    rels_a = rot_left["rels"]
    embs_b = rot_right["embs"]
    rels_b = rot_right["rels"]
    fam_a  = rot_left["fam_map"]
    fam_b  = rot_right["fam_map"]
    k      = len(rot_left["fam_names"])

    def short(uri):
        return uri.split("#")[-1].split("/")[-1]

    # Procrustes on anchor pairs
    sv, tv = [], []
    for anchor in anchors:
        si = next((i for i, r in enumerate(rels_a)
                   if anchor in short(r)), None)
        ti = next((i for i, r in enumerate(rels_b)
                   if anchor in short(r)), None)
        if si is not None and ti is not None:
            sv.append(embs_a[si])
            tv.append(embs_b[ti])

    if sv:
        R, _ = orthogonal_procrustes(np.array(sv), np.array(tv))
        embs_a_aligned = normalize(embs_a @ R)
        print(f"  Procrustes ({ds_name}): {len(sv)} anchor pairs")
    else:
        embs_a_aligned = embs_a
        print(f"  Procrustes ({ds_name}): no anchors, skipping")

    # Family centroids
    def centroids(embs, rels, fmap, k):
        c = np.zeros((k, embs.shape[1]))
        n = np.zeros(k)
        for i, r in enumerate(rels):
            f = fmap.get(r, 0)
            if f < k:
                c[f] += embs[i]
                n[f] += 1
        return normalize(c / np.maximum(n[:, None], 1))

    cent_a = centroids(embs_a_aligned, rels_a, fam_a, k)
    cent_b = centroids(embs_b,         rels_b, fam_b, k)

    # OT family assignment
    cost = 1 - cent_a @ cent_b.T
    a = np.ones(k) / k
    b = np.ones(k) / k
    plan = ot.emd(a, b, cost)
    mapping = plan.argmax(axis=1)
    print(f"  OT mapping ({ds_name}): {mapping.tolist()}")

    save(mapping, cache_key)
    return mapping


# ── Stage 1 entrypoint ────────────────────────────────────────────────────────

def run_stage1(raw: dict, datasets_cfg: dict,
               cache_dir: str, client, model: str,
               device: str = "cuda") -> dict:
    """
    Run Stage 1 for all datasets.

    Returns:
        rotate_results  : {ds_name: {left: ..., right: ...}}
        family_mappings : {ds_name: mapping_array}
    """
    print("\n" + "=" * 60)
    print("Stage 1: RotatE Relation Family Induction")
    print("=" * 60)

    rotate_results  = {}
    family_mappings = {}

    for ds_name, cfg in datasets_cfg.items():
        print(f"\n{ds_name}:")
        data = raw[ds_name]

        rot_left  = train_rotate(data["rel1"], f"{ds_name}_left",
                                 cache_dir,
                                 dim=cfg["rotate_dim"],
                                 epochs=cfg["rotate_epochs"],
                                 k=cfg["k_rotate"],
                                 device=device)
        rot_right = train_rotate(data["rel2"], f"{ds_name}_right",
                                 cache_dir,
                                 dim=cfg["rotate_dim"],
                                 epochs=cfg["rotate_epochs"],
                                 k=cfg["k_rotate"],
                                 device=device)
        rotate_results[ds_name] = {"left": rot_left, "right": rot_right}

        anchors = detect_anchors(
            rot_left["rels"], rot_right["rels"],
            ds_name, cache_dir, client, model)

        mapping = align_families(
            rot_left, rot_right, anchors, ds_name, cache_dir)
        family_mappings[ds_name] = mapping

        print(f"  Left families:  {rot_left['fam_names']}")
        print(f"  Right families: {rot_right['fam_names']}")

    save(rotate_results,  Path(cache_dir) / "rotate_results.pkl")
    save(family_mappings, Path(cache_dir) / "family_mappings.pkl")
    print("\n✓ Stage 1 complete")
    return rotate_results, family_mappings
