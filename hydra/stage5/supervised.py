"""
HyDRA Stage 5 — Supervised Fine-Tuning (MNRL)
==============================================
Fine-tunes BGE-large-en-v1.5 with MultipleNegativesRankingLoss
on gold alignment pairs, then applies lexical reranking and
LLM reasoning on the held-out test split.

All supervised results use an 80/20 train/test split
(prior to the recommended 80/10/10 benchmark split release).
"""

import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from utils.helpers import save, load, exists
from stage2.profiling import profile_to_text
from stage3.retrieval import cosine_retrieve
from stage4.reranking import (rerank_lex, get_still_wrong,
                               llm_rerank, compute_stage4_metrics)


# ── MNRL fine-tuning ──────────────────────────────────────────────────────────

def finetune_mnrl(ds_name: str, gold_pairs: list,
                   profs_left: dict, profs_right: dict,
                   bge_prefix: str, encoder_name: str,
                   cache_dir: str, device: str,
                   test_size: float = 0.20,
                   epochs: int = 5, batch_size: int = 16,
                   lr: float = 2e-5) -> SentenceTransformer:
    """
    Fine-tune BGE with MNRL on training gold pairs.
    Returns the fine-tuned SentenceTransformer model.
    """
    model_path = Path(cache_dir) / f"model_mnrl_{ds_name}"

    if model_path.exists():
        print(f"  {ds_name}: fine-tuned model loaded from {model_path}")
        return SentenceTransformer(str(model_path), device=device)

    train_pairs, _ = train_test_split(
        gold_pairs, test_size=test_size, random_state=42)
    print(f"  {ds_name}: {len(train_pairs):,} training pairs")

    examples = []
    for a, b in train_pairs:
        pa = profs_left.get(a)
        pb = profs_right.get(b)
        if pa and pb:
            examples.append(InputExample(
                texts=[profile_to_text(pa, bge_prefix=bge_prefix),
                       profile_to_text(pb, bge_prefix=bge_prefix)]
            ))
    print(f"  {ds_name}: {len(examples):,} training examples")

    model = SentenceTransformer(encoder_name, device=device)
    model[0].auto_model.gradient_checkpointing_enable()

    loader  = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=max(100, len(examples) // 20),
        show_progress_bar=True,
        output_path=str(model_path),
        optimizer_params={"lr": lr},
    )
    print(f"  {ds_name}: fine-tuning complete → {model_path}")
    return model


# ── Supervised evaluation ─────────────────────────────────────────────────────

def eval_supervised(ds_name: str, model: SentenceTransformer,
                     gold_pairs: list, profs_left: dict,
                     profs_right: dict, bge_prefix: str,
                     cache_dir: str, client, llm_model: str,
                     test_size: float = 0.20,
                     top_k: int = 10,
                     batch_size: int = 512) -> dict:
    """
    Evaluate fine-tuned model on held-out test split with
    lex reranking and LLM reasoning.
    """
    _, test_pairs = train_test_split(
        gold_pairs, test_size=test_size, random_state=42)
    gt_test = {a: b for a, b in test_pairs}
    print(f"  {ds_name}: test set = {len(test_pairs):,}")

    # Encode test left and ALL right entities
    all_right_uris = [
        u for u in profs_right
        if profs_right[u].get("has_label") or profs_right[u].get("label")
    ]
    all_right_texts = [profile_to_text(profs_right[u], bge_prefix)
                       for u in all_right_uris]

    test_left_uris  = [a for a, b in test_pairs if a in profs_left]
    test_left_texts = [profile_to_text(profs_left[u], bge_prefix)
                       for u in test_left_uris]

    print(f"  Encoding {len(all_right_uris):,} right entities...")
    embs_r = model.encode(all_right_texts, batch_size=batch_size,
                           normalize_embeddings=True,
                           show_progress_bar=True,
                           convert_to_numpy=True)

    print(f"  Encoding {len(test_left_uris):,} test entities...")
    embs_l = model.encode(test_left_texts, batch_size=batch_size,
                           normalize_embeddings=True,
                           show_progress_bar=True,
                           convert_to_numpy=True)

    right_idx = {u: i for i, u in enumerate(all_right_uris)}
    scores    = embs_l @ embs_r.T
    top10     = np.argsort(-scores, axis=1)[:, :top_k]

    hits = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    incorrect = {}
    n = 0

    for i, src in enumerate(test_left_uris):
        tgt = gt_test.get(src)
        if not tgt or tgt not in right_idx:
            continue
        n += 1
        ranked   = [all_right_uris[j] for j in top10[i]]
        ranked_s = [float(scores[i, j]) for j in top10[i]]

        if tgt in ranked:
            rank = ranked.index(tgt) + 1
            rr_sum += 1.0 / rank
            for kk in [1, 5, 10]:
                if rank <= kk:
                    hits[kk] += 1

        if tgt not in ranked or ranked.index(tgt) > 0:
            incorrect[src] = {
                "true_match":     tgt,
                "all_candidates": list(zip(ranked, ranked_s)),
            }

    mnrl_metrics = {
        "hits1":  round(hits[1]  / n, 4),
        "hits5":  round(hits[5]  / n, 4),
        "hits10": round(hits[10] / n, 4),
        "mrr":    round(rr_sum   / n, 4),
        "n":      n,
    }
    print(f"  MNRL H@1={mnrl_metrics['hits1']:.4f}  "
          f"H@5={mnrl_metrics['hits5']:.4f}  "
          f"H@10={mnrl_metrics['hits10']:.4f}  "
          f"MRR={mnrl_metrics['mrr']:.4f}")

    # Lex reranking on top of MNRL
    rr = rerank_lex(incorrect, profs_left, profs_right)
    rr_correct   = sum(1 for s, p in rr.items() if gt_test.get(s) == p)
    h1_lex = (n - len(incorrect) + rr_correct) / n
    print(f"  MNRL+lex H@1={h1_lex:.4f}")

    # LLM on still-wrong
    still_wrong = get_still_wrong(rr, incorrect, gt_test)
    print(f"  Still wrong after lex: {len(still_wrong):,}")
    llm_key = f"mnrl_llm_{ds_name}"
    llm_preds = llm_rerank(
        still_wrong, profs_left, profs_right,
        llm_key, cache_dir, client, llm_model, top_k)

    final_metrics = compute_stage4_metrics(
        incorrect, rr, llm_preds, gt_test, n)
    print(f"  MNRL+lex+LLM H@1={final_metrics['hits1']:.4f}")

    return {
        "mnrl":         mnrl_metrics,
        "h1_mnrl_lex":  h1_lex,
        "h1_mnrl_llm":  final_metrics["hits1"],
        "final":        final_metrics,
        "train_n":      len(gold_pairs) - len(test_pairs),
        "test_n":       len(test_pairs),
    }


# ── Stage 5 entrypoint ────────────────────────────────────────────────────────

def run_stage5(all_profiles: dict, raw: dict,
               cache_dir: str, bge_prefix: str,
               encoder_name: str, client, llm_model: str,
               device: str = "cuda",
               test_size: float = 0.20,
               epochs: int = 5, batch_size: int = 16,
               lr: float = 2e-5) -> dict:
    """
    Fine-tune BGE with MNRL and evaluate with lex+LLM on test split.

    Returns mnrl_results: {ds_name: result_dict}
    """
    print("\n" + "=" * 60)
    print("Stage 5: MNRL Supervised Fine-Tuning")
    print("=" * 60)

    mnrl_results = {}

    for ds_name, data in all_profiles.items():
        print(f"\n{ds_name}:")
        gold_pairs = raw[ds_name]["gold"]

        cache_key = Path(cache_dir) / f"mnrl_results_{ds_name}.pkl"
        if exists(cache_key):
            print(f"  {ds_name}: results loaded from cache")
            mnrl_results[ds_name] = load(cache_key)
            continue

        model = finetune_mnrl(
            ds_name, gold_pairs, data["left"], data["right"],
            bge_prefix, encoder_name, cache_dir, device,
            test_size, epochs, batch_size, lr)

        result = eval_supervised(
            ds_name, model, gold_pairs,
            data["left"], data["right"], bge_prefix,
            cache_dir, client, llm_model,
            test_size=test_size)

        mnrl_results[ds_name] = result
        save(result, cache_key)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    save(mnrl_results, Path(cache_dir) / "mnrl_results.pkl")
    print("\n✓ Stage 5 complete")
    return mnrl_results
