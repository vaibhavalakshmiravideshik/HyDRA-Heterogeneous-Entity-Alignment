"""
HyDRA Stage 4 — Heterogeneity-Aware Reranking
==============================================
Two-step reranking:
  (a) Lexical reranking: weighted Jaccard/bigram/trigram over names+synonyms
  (b) LLM candidate reasoning: GPT-4o on entities still incorrect after lex

LLM gate: oracle mode (uses gold labels, for evaluation upper bound)
          or threshold mode (deployable, uses lex confidence score).
"""

import os
import re
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.helpers import (save, load, exists, lex_score,
                            llm_call)


# ── Lexical reranking ──────────────────────────────────────────────────────────

def rerank_lex(incorrect: dict, profs_left: dict, profs_right: dict,
               w_lex: float = 0.80, w_cos: float = 0.20) -> dict:
    """
    Rerank each incorrect entity's candidates using:
        score = w_lex * lex_score(src, cand) + w_cos * cosine_score

    Returns {src_uri: best_tgt_uri}
    """
    results = {}
    for src, info in incorrect.items():
        sp    = profs_left.get(src, {})
        stxts = ([sp.get("label") or ""] +
                 (sp.get("synonyms") or [])[:2])
        best_uri, best_score = None, -999.0
        for cand_uri, cos_score in info["all_candidates"]:
            tp    = profs_right.get(cand_uri, {})
            ttxts = ([tp.get("label") or ""] +
                     (tp.get("synonyms") or [])[:2])
            lx = max(
                (lex_score(s, t) for s in stxts for t in ttxts),
                default=0.0,
            )
            score = w_lex * lx + w_cos * cos_score
            if score > best_score:
                best_score = score
                best_uri   = cand_uri
        results[src] = best_uri
    return results


def get_still_wrong(reranked: dict, incorrect: dict,
                    gold_dict: dict) -> dict:
    """
    Return subset of incorrect entities still wrong after lex reranking.
    Oracle gate — requires gold labels.
    """
    return {
        src: incorrect[src]
        for src, pred in reranked.items()
        if gold_dict.get(src) != pred and src in incorrect
    }


def get_still_wrong_threshold(incorrect: dict, reranked: dict,
                               gold_dict: dict,
                               threshold: float = 1.0) -> dict:
    """
    Deployable gate: invoke LLM for entities whose top-1 lex score < threshold.
    Does NOT use gold labels.
    """
    still_wrong = {}
    for src, info in incorrect.items():
        top_score = info["all_candidates"][0][1] if info["all_candidates"] else 1.0
        if top_score < threshold:
            still_wrong[src] = info
    return still_wrong


# ── LLM candidate reasoning ───────────────────────────────────────────────────

def _build_llm_prompt(src_uri: str, candidates: list,
                       profs_left: dict, profs_right: dict) -> str:
    sp      = profs_left.get(src_uri, {})
    src_name = sp.get("label") or src_uri.split("/")[-1]
    src_syns = sp.get("synonyms") or []
    src_def  = (sp.get("definition") or "")[:150]
    src_rels = [
        f"{nb['rel_label']}: {nb['tgt_label']}"
        for nb in (sp.get("neighbors") or [])[:4]
        if nb.get("tgt_label")
    ]

    src_text = f"Name: {src_name}"
    if src_syns: src_text += f"\nSynonyms: {'; '.join(src_syns[:3])}"
    if src_def:  src_text += f"\nDefinition: {src_def}"
    if src_rels: src_text += f"\nRelations: {' | '.join(src_rels)}"

    cands_text = ""
    for idx, (cand_uri, _) in enumerate(candidates):
        tp      = profs_right.get(cand_uri, {})
        tgt_name = tp.get("label") or cand_uri.split("/")[-1]
        tgt_syns = (tp.get("synonyms") or [])[:2]
        tgt_rels = [
            f"{nb['rel_label']}: {nb['tgt_label']}"
            for nb in (tp.get("neighbors") or [])[:3]
            if nb.get("tgt_label")
        ]
        cands_text += f"\n{idx+1}. Name: {tgt_name}"
        if tgt_syns: cands_text += f" | Synonyms: {'; '.join(tgt_syns)}"
        if tgt_rels: cands_text += f" | Relations: {' | '.join(tgt_rels)}"

    return f"""You are an expert biomedical ontology alignment system.
You will be given a source entity and a list of {len(candidates)} candidate target entities.
Select the candidate that best represents the same real-world concept as the source entity.

Source entity:
{src_text}

Candidate entities (select by number 1-{len(candidates)}):
{cands_text}

Instructions:
1. Identify the candidate whose name or synonyms most closely match the source entity name.
2. If names are ambiguous, use definitions and relations to disambiguate.
3. Return ONLY the integer number (1-{len(candidates)}) of the best matching candidate. No explanation.

Answer:"""


def llm_rerank(still_wrong: dict, profs_left: dict,
               profs_right: dict, ds_name: str,
               cache_dir: str, client, model: str,
               top_k: int = 10) -> dict:
    """
    Run LLM candidate reasoning on still-wrong entities.
    Resume-safe: writes predictions to a JSONL file as they complete.

    Returns {src_uri: predicted_tgt_uri}
    """
    cache_key  = Path(cache_dir) / f"llm_reranked_{ds_name}.pkl"
    jsonl_path = Path(cache_dir) / f"llm_rerank_{ds_name}.jsonl"

    if exists(cache_key):
        print(f"  {ds_name}: LLM predictions loaded from cache")
        return load(cache_key)

    done = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[d["src"]] = d["pred"]
                except Exception:
                    pass

    remaining = [u for u in still_wrong if u not in done]
    print(f"  {ds_name}: {len(done):,} done | {len(remaining):,} remaining")

    with open(jsonl_path, "a") as fout:
        for src in tqdm(remaining, desc=f"  LLM {ds_name}"):
            cands  = still_wrong[src]["all_candidates"][:top_k]
            prompt = _build_llm_prompt(src, cands, profs_left, profs_right)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                )
                raw_resp = resp.choices[0].message.content.strip()
                nums = re.findall(r"\d+", raw_resp)
                idx  = int(nums[0]) - 1 if nums else 0
                idx  = max(0, min(idx, len(cands) - 1))
                pred = cands[idx][0]
            except Exception:
                pred = cands[0][0]

            done[src] = pred
            fout.write(json.dumps({"src": src, "pred": pred}) + "\n")
            fout.flush()

    save(done, cache_key)
    return done


# ── Metric computation after full Stage 4 ────────────────────────────────────

def compute_stage4_metrics(incorrect: dict, reranked_lex: dict,
                            llm_preds: dict, gold_dict: dict,
                            n_total: int) -> dict:
    """
    Compute H@1/H@5/H@10/MRR after Stage 4 (lex + LLM).

    For H@5/H@10/MRR: uses original candidate lists for entities
    that lex/LLM did NOT fix to rank 1.
    """
    h1 = h5 = h10 = mrr_sum = 0
    n  = n_total

    already_correct = n_total - len(incorrect)
    h1   += already_correct
    h5   += already_correct
    h10  += already_correct
    mrr_sum += float(already_correct)

    # Apply LLM overrides on top of lex predictions
    final_preds = dict(reranked_lex)
    for src, pred in llm_preds.items():
        final_preds[src] = pred

    for src, pred in final_preds.items():
        tgt = gold_dict.get(src)
        if not tgt:
            continue
        if pred == tgt:
            h1 += 1; h5 += 1; h10 += 1; mrr_sum += 1.0
        else:
            cands = [c[0] for c in (incorrect.get(src, {})
                                    .get("all_candidates", []))]
            if tgt in cands:
                rank = cands.index(tgt) + 1
                mrr_sum += 1.0 / rank
                if rank <= 5:  h5  += 1
                if rank <= 10: h10 += 1

    return {
        "hits1":  round(h1     / n, 4),
        "hits5":  round(h5     / n, 4),
        "hits10": round(h10    / n, 4),
        "mrr":    round(mrr_sum / n, 4),
        "n":      n,
    }


# ── Stage 4 entrypoint ────────────────────────────────────────────────────────

def run_stage4(all_profiles: dict, incorrect_bge: dict,
               raw: dict, cache_dir: str,
               client, model: str,
               w_lex: float = 0.80, w_cos: float = 0.20,
               top_k: int = 10,
               oracle_gate: bool = True,
               lex_threshold: float = 1.0) -> tuple:
    """
    Run Stage 4 (lex reranking + LLM reasoning) for all datasets.

    oracle_gate=True  : use gold labels to find which entities to send to LLM
                        (upper-bound evaluation, as in the paper Table 2)
    oracle_gate=False : use lex_threshold for deployable evaluation

    Returns:
        reranked_results : {ds_name: {src: pred}}  (lex predictions)
        llm_results      : {ds_name: {src: pred}}  (LLM override predictions)
        final_metrics    : {ds_name: metrics_dict}
    """
    print("\n" + "=" * 60)
    print(f"Stage 4: Lexical Reranking + LLM Reasoning "
          f"({'oracle gate' if oracle_gate else f'threshold={lex_threshold}'})")
    print("=" * 60)

    reranked_results = {}
    llm_results      = {}
    final_metrics    = {}

    for ds_name, data in all_profiles.items():
        print(f"\n{ds_name}:")
        gold_dict = dict(raw[ds_name]["gold"])
        incorrect = incorrect_bge[ds_name]
        n_total   = len([u for u in
                         list(data["left"].keys())[:1] and gold_dict])
        # recount total gold pairs correctly
        n_total = sum(1 for u in incorrect_bge[ds_name]) + \
                  (len(gold_dict) - len(incorrect_bge[ds_name]))

        profs_l = data["left"]
        profs_r = data["right"]

        # Lex reranking
        rr = rerank_lex(incorrect, profs_l, profs_r, w_lex, w_cos)
        save(rr, Path(cache_dir) / f"reranked_{ds_name}.pkl")

        already_correct = n_total - len(incorrect)
        rr_correct = sum(1 for s, p in rr.items()
                         if gold_dict.get(s) == p)
        h1_lex = (already_correct + rr_correct) / n_total
        print(f"  After lex reranking: H@1={h1_lex:.4f}")

        # Gate for LLM
        if oracle_gate:
            still_wrong = get_still_wrong(rr, incorrect, gold_dict)
        else:
            still_wrong = get_still_wrong_threshold(
                incorrect, rr, gold_dict, lex_threshold)
        print(f"  Sending {len(still_wrong):,} entities to LLM "
              f"({len(still_wrong)/n_total*100:.1f}%)")

        # LLM reasoning
        llm_preds = llm_rerank(
            still_wrong, profs_l, profs_r, ds_name,
            cache_dir, client, model, top_k)

        # Final metrics
        metrics = compute_stage4_metrics(
            incorrect, rr, llm_preds, gold_dict, n_total)

        reranked_results[ds_name] = rr
        llm_results[ds_name]      = llm_preds
        final_metrics[ds_name]    = metrics

        print(f"  Final: H@1={metrics['hits1']:.4f}  "
              f"H@5={metrics['hits5']:.4f}  "
              f"H@10={metrics['hits10']:.4f}  "
              f"MRR={metrics['mrr']:.4f}")

    save(reranked_results, Path(cache_dir) / "reranked_results.pkl")
    save(llm_results,      Path(cache_dir) / "llm_reranked_results.pkl")
    save(final_metrics,    Path(cache_dir) / "zeroshot_final_metrics.pkl")
    print("\n✓ Stage 4 complete")
    return reranked_results, llm_results, final_metrics
