"""
HyDRA Baselines
===============
Lexical matching, SapBERT (zero-shot + fine-tuned),
EasyEA (wrapper), GraphEmbeddingAligner (OntoAligner).
"""

import gc
import re
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from utils.helpers import save, load, exists, extract_literal


# ── Lexical matching ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return (s or "").lower().strip().rstrip(".")


def lexical_baseline(gold_pairs: list, profs_left: dict,
                      profs_right: dict, ds_name: str,
                      cache_dir: str) -> dict:
    """
    Exact + synonym string matching over normalised entity names.
    Returns {h1, p, r, f1, n}.
    """
    cache_key = Path(cache_dir) / f"lexical_{ds_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_name}: lexical loaded from cache")
        return load(cache_key)

    gt = dict(gold_pairs)

    right_by_name: dict = defaultdict(list)
    for uri, p in profs_right.items():
        for name in ([p.get("label")] + (p.get("synonyms") or [])):
            n = _normalize(name)
            if n:
                right_by_name[n].append(uri)

    hits = n = 0
    for src, tgt in gold_pairs:
        p = profs_left.get(src, {})
        src_names = [_normalize(p.get("label"))] + \
                    [_normalize(s) for s in (p.get("synonyms") or [])]
        n += 1
        for nm in src_names:
            if nm and tgt in right_by_name.get(nm, []):
                hits += 1
                break

    h1 = round(hits / n, 4)
    p  = h1    # precision == recall == F1 for top-1 exact match
    out = {"h1": h1, "p": p, "r": p, "f1": p, "n": n}
    print(f"  {ds_name} lexical: H@1={h1:.4f} ({hits}/{n})")
    save(out, cache_key)
    return out


# ── SapBERT ───────────────────────────────────────────────────────────────────

def _sapbert_text(p: dict) -> str:
    name = p.get("label") or "[MISSING]"
    syns = (p.get("synonyms") or [])[:2]
    return name + (" | " + " | ".join(syns) if syns else "")


def sapbert_eval(ds_name: str, profs_left: dict,
                  profs_right: dict, gold_pairs: list,
                  cache_dir: str, device: str = "cuda",
                  test_size: float = 0.20,
                  epochs: int = 5, batch_size: int = 16,
                  lr: float = 2e-5) -> dict:
    """
    Evaluate SapBERT zero-shot and fine-tuned on gold-pool setting.
    Returns {zeroshot: h1, finetuned: h1}.
    """
    cache_key = Path(cache_dir) / f"sapbert_{ds_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_name}: SapBERT loaded from cache")
        return load(cache_key)

    sapbert_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    train_pairs, test_pairs = train_test_split(
        gold_pairs, test_size=test_size, random_state=42)

    all_right_uris  = [u for u in profs_right if profs_right[u].get("label")]
    all_right_texts = [_sapbert_text(profs_right[u]) for u in all_right_uris]
    test_left_uris  = [a for a, b in test_pairs if a in profs_left]
    test_left_texts = [_sapbert_text(profs_left[u]) for u in test_left_uris]
    gt_test   = {a: b for a, b in test_pairs}
    right_idx = {u: i for i, u in enumerate(all_right_uris)}

    def eval_embs(embs_l, embs_r):
        scores = embs_l @ embs_r.T
        top1   = np.argmax(scores, axis=1)
        hits = n = 0
        for i, src in enumerate(test_left_uris):
            tgt = gt_test.get(src)
            if not tgt or tgt not in right_idx: continue
            n += 1
            if all_right_uris[top1[i]] == tgt: hits += 1
        return round(hits / n, 4)

    # Zero-shot
    model   = SentenceTransformer(sapbert_name, device=device)
    embs_r  = model.encode(all_right_texts, batch_size=512,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True)
    embs_l  = model.encode(test_left_texts, batch_size=512,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True)
    zs = eval_embs(embs_l, embs_r)
    print(f"  {ds_name} SapBERT-ZS: H@1={zs:.4f}")

    # Fine-tuned
    examples = [
        InputExample(texts=[_sapbert_text(profs_left[a]),
                             _sapbert_text(profs_right[b])])
        for a, b in train_pairs
        if a in profs_left and b in profs_right
    ]
    print(f"  Fine-tuning on {len(examples):,} pairs...")
    model[0].auto_model.gradient_checkpointing_enable()
    loader  = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = MultipleNegativesRankingLoss(model)
    ft_path = Path(cache_dir) / f"model_sapbert_{ds_name}"
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=max(100, len(examples) // 20),
        show_progress_bar=True,
        output_path=str(ft_path),
        optimizer_params={"lr": lr},
    )
    embs_r = model.encode(all_right_texts, batch_size=512,
                           normalize_embeddings=True,
                           show_progress_bar=False,
                           convert_to_numpy=True)
    embs_l = model.encode(test_left_texts, batch_size=512,
                           normalize_embeddings=True,
                           show_progress_bar=False,
                           convert_to_numpy=True)
    ft = eval_embs(embs_l, embs_r)
    print(f"  {ds_name} SapBERT-FT: H@1={ft:.4f}")

    out = {"zeroshot": zs, "finetuned": ft}
    save(out, cache_key)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out


# ── GraphEmbeddingAligner (OntoAligner) ──────────────────────────────────────

def _build_onto_dict(attr_triples: list, rel_triples: list,
                      gold_set: set) -> dict:
    """Build OntoAligner-compatible ontology dict."""
    LABEL_PREDS = {
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
        "http://id.nlm.nih.gov/mesh/vocab#prefLabel",
    }
    ent2name = {}
    for h, r, v in attr_triples:
        if r in LABEL_PREDS and h not in ent2name:
            val = extract_literal(v)
            if val: ent2name[h] = val

    all_uris = set()
    for h, r, t in rel_triples:
        all_uris.add(h); all_uris.add(t)
    all_uris |= gold_set

    seen = {}; entity2iri = {}; iri2key = {}
    for uri in all_uris:
        name = ent2name.get(uri, uri.split("/")[-1])
        key  = name; idx = 0
        while key in seen and seen[key] != uri:
            key = f"{name}__{idx}"; idx += 1
        seen[key] = uri
        entity2iri[key] = uri
        iri2key[uri]    = key

    triplets = set()
    for h, r, t in rel_triples:
        hk = iri2key.get(h); tk = iri2key.get(t)
        rk = r.split("#")[-1].split("/")[-1]
        if hk and tk:
            triplets.add((hk, rk, tk))

    gold_keys   = {iri2key[u] for u in gold_set if u in iri2key}
    missing     = gold_keys - {t[0] for t in triplets} - {t[2] for t in triplets}
    for key in missing:
        triplets.add((key, "self_loop", key))

    return {
        "entity2iri": {iri2key[u]: u for u in gold_set if u in iri2key},
        "triplets":   list(triplets),
        "iri2key":    iri2key,
    }


def gea_baseline(ds_name: str, data: dict,
                  cache_dir: str, epochs: int = 20,
                  dim: int = 200,
                  device: str = "cuda") -> dict:
    """
    Run GraphEmbeddingAligner (TransE/DistMult/ConvE) via OntoAligner.
    Returns {TransE: {h1,p,r,f1}, DistMult: ..., ConvE: ...}
    Skips MeSH-SNOMED (scale too large for KGE training on 6.9M triples).
    """
    cache_key = Path(cache_dir) / f"gea_{ds_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_name}: GEA loaded from cache")
        return load(cache_key)

    try:
        from ontoaligner.aligner.graph.models import (
            TransEAligner, DistMultAligner, ConvEAligner)
    except ImportError:
        print("  OntoAligner not installed. pip install ontoaligner")
        return {}

    source = _build_onto_dict(data["attr1"], data["rel1"],
                               data["gold_left"])
    target = _build_onto_dict(data["attr2"], data["rel2"],
                               data["gold_right"])
    gt = dict(data["gold"])
    n  = len(data["gold"])

    results = {}
    for AlignerClass, mname in [
        (TransEAligner,   "TransE"),
        (DistMultAligner, "DistMult"),
        (ConvEAligner,    "ConvE"),
    ]:
        try:
            aligner = AlignerClass(
                model=mname, device=device,
                embedding_dim=dim, num_epochs=epochs,
                train_batch_size=512, eval_batch_size=256,
                num_negs_per_pos=5, top_k=1,
            )
            aligner.fit(source["triplets"] + target["triplets"])
            matchings = aligner.predict(source, target)
            preds = {m["source"]: m["target"]
                     for m in matchings
                     if "source" in m and "target" in m}
            tp = sum(1 for s, t in gt.items() if preds.get(s) == t)
            fp = max(0, len(preds) - tp)
            fn = n - tp
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            h1 = tp / n
            results[mname] = {"h1": round(h1, 4), "p": round(p, 4),
                               "r": round(r, 4), "f1": round(f1, 4)}
            print(f"  {ds_name} GEA-{mname}: H@1={h1:.4f} F1={f1:.4f}")
        except Exception as e:
            print(f"  {ds_name} GEA-{mname} failed: {e}")
            results[mname] = None

    save(results, cache_key)
    return results
