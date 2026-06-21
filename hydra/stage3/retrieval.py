"""
HyDRA Stage 3 — Asymmetry-Aware Candidate Retrieval
====================================================
Encodes entity profiles with BGE-large-en-v1.5 and retrieves
top-k candidates per source entity via cosine similarity.
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from utils.helpers import save, load, exists
from stage2.profiling import profile_to_text


# ── BGE encoding ──────────────────────────────────────────────────────────────

def encode_profiles(profiles: dict, uri_list: list,
                    model: SentenceTransformer,
                    bge_prefix: str = "",
                    batch_size: int = 512) -> np.ndarray:
    """
    Encode a list of entity URIs using their profiles.
    Returns L2-normalised embeddings array of shape (n, dim).
    """
    texts = [
        profile_to_text(profiles[u], bge_prefix=bge_prefix)
        if u in profiles
        else bge_prefix + (u.split("/")[-1].replace("_", " "))
        for u in uri_list
    ]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs


def encode_dataset_side(profiles: dict, ds_side_name: str,
                         cache_dir: str, model: SentenceTransformer,
                         bge_prefix: str = "",
                         batch_size: int = 512) -> dict:
    """
    Encode all profiles for one dataset side, with caching.

    Returns dict: {uris: list, embs: np.ndarray}
    """
    cache_key = Path(cache_dir) / f"bge_embs_{ds_side_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_side_name}: embeddings loaded from cache")
        return load(cache_key)

    all_uris = list(profiles.keys())
    print(f"  Encoding {len(all_uris):,} entities ({ds_side_name})...")
    embs = encode_profiles(profiles, all_uris, model,
                            bge_prefix, batch_size)
    out = {"uris": all_uris, "embs": embs}
    save(out, cache_key)
    print(f"  {ds_side_name}: {len(all_uris):,} encoded, dim={embs.shape[1]}")
    return out


# ── Cosine retrieval ──────────────────────────────────────────────────────────

def cosine_retrieve(left_embs: dict, right_embs: dict,
                    gold_dict: dict, k: int = 10,
                    batch_size: int = 256) -> tuple:
    """
    Retrieve top-k target candidates for each gold source entity
    using cosine similarity.

    Returns:
        metrics    : dict with hits@1/5/10 and MRR
        incorrect  : {src_uri: {true_match, all_candidates}} for rank > 1
    """
    src_uris   = left_embs["uris"]
    tgt_uris   = right_embs["uris"]
    src_mat    = left_embs["embs"]
    tgt_mat    = right_embs["embs"]

    src_idx    = {u: i for i, u in enumerate(src_uris)}
    tgt_idx    = {u: i for i, u in enumerate(tgt_uris)}

    gold_srcs  = [u for u in src_uris if u in gold_dict]
    gold_rows  = src_mat[[src_idx[u] for u in gold_srcs]]

    hits = {1: 0, 5: 0, 10: 0}
    rr_sum  = 0.0
    incorrect = {}
    n = 0

    for i in range(0, len(gold_srcs), batch_size):
        b_srcs = gold_srcs[i: i + batch_size]
        b_mat  = gold_rows[i: i + batch_size]
        sims   = b_mat @ tgt_mat.T

        for j, src in enumerate(b_srcs):
            tgt = gold_dict.get(src)
            if not tgt or tgt not in tgt_idx:
                continue
            n += 1
            row      = sims[j]
            topk_idx = np.argpartition(row, -k)[-k:]
            topk_idx = topk_idx[np.argsort(row[topk_idx])[::-1]]
            ranked   = [tgt_uris[ii] for ii in topk_idx]
            ranked_s = [float(row[ii]) for ii in topk_idx]

            if tgt in ranked:
                rank = ranked.index(tgt) + 1
                rr_sum += 1.0 / rank
                for kk in [1, 5, 10]:
                    if rank <= kk:
                        hits[kk] += 1
            else:
                rank = len(tgt_uris) + 1

            if rank > 1:
                incorrect[src] = {
                    "true_match":     tgt,
                    "all_candidates": list(zip(ranked, ranked_s)),
                }

    metrics = {
        "hits1":  round(hits[1]  / n, 4),
        "hits5":  round(hits[5]  / n, 4),
        "hits10": round(hits[10] / n, 4),
        "mrr":    round(rr_sum   / n, 4),
        "n":      n,
    }
    return metrics, incorrect


# ── Stage 3 entrypoint ────────────────────────────────────────────────────────

def run_stage3(all_profiles: dict, raw: dict,
               cache_dir: str, bge_prefix: str,
               encoder_name: str = "BAAI/bge-large-en-v1.5",
               top_k: int = 10, device: str = "cuda") -> tuple:
    """
    Encode all profiles with BGE and run cosine retrieval.

    Returns:
        bge_embs      : {ds_name: {left: {uris,embs}, right: {uris,embs}}}
        zeroshot_bge  : {ds_name: metrics_dict}
        incorrect_bge : {ds_name: incorrect_dict}
    """
    print("\n" + "=" * 60)
    print("Stage 3: BGE Encoding + Cosine Retrieval")
    print("=" * 60)

    model = SentenceTransformer(encoder_name, device=device)

    bge_embs     = {}
    zeroshot_bge = {}
    incorrect_bge = {}

    for ds_name, data in all_profiles.items():
        print(f"\n{ds_name}:")
        left_emb = encode_dataset_side(
            data["left"], f"{ds_name}_left", cache_dir, model, bge_prefix)
        right_emb = encode_dataset_side(
            data["right"], f"{ds_name}_right", cache_dir, model, bge_prefix)
        bge_embs[ds_name] = {"left": left_emb, "right": right_emb}

        gold_dict = dict(raw[ds_name]["gold"])
        metrics, incorrect = cosine_retrieve(
            left_emb, right_emb, gold_dict, k=top_k)

        zeroshot_bge[ds_name]  = metrics
        incorrect_bge[ds_name] = incorrect

        save(incorrect, Path(cache_dir) / f"incorrect_{ds_name}_bge.pkl")
        print(f"  BGE H@1={metrics['hits1']:.4f}  "
              f"H@5={metrics['hits5']:.4f}  "
              f"H@10={metrics['hits10']:.4f}  "
              f"MRR={metrics['mrr']:.4f}  n={metrics['n']:,}")

    save(bge_embs,      Path(cache_dir) / "bge_embs.pkl")
    save(zeroshot_bge,  Path(cache_dir) / "zeroshot_bge.pkl")
    save(incorrect_bge, Path(cache_dir) / "incorrect_bge.pkl")
    print("\n✓ Stage 3 complete")
    return bge_embs, zeroshot_bge, incorrect_bge
