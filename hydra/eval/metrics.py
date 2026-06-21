"""
HyDRA Evaluation
================
Metric computation: Hits@k, MRR, Precision, Recall, F1.
Threshold gating analysis (oracle vs deployable).
"""

import numpy as np
from pathlib import Path

from utils.helpers import save, load, exists, lex_score


# ── Per-ranking metrics ───────────────────────────────────────────────────────

def hits_mrr(ranked_lists: list, gold_targets: list, ks=(1, 5, 10)) -> dict:
    """
    Compute Hits@k and MRR from ranked prediction lists.

    Args:
        ranked_lists  : list of list-of-URIs (predictions per query)
        gold_targets  : list of gold target URIs (one per query)
        ks            : tuple of k values to compute Hits@k for

    Returns dict with keys hits1, hits5, hits10, mrr, n.
    """
    hits   = {k: 0 for k in ks}
    rr_sum = 0.0
    n      = 0

    for ranked, tgt in zip(ranked_lists, gold_targets):
        if not tgt:
            continue
        n += 1
        try:
            rank = ranked.index(tgt) + 1
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
        except ValueError:
            pass  # tgt not in ranked — counts as miss

    metrics = {f"hits{k}": round(hits[k] / n, 4) for k in ks}
    metrics["mrr"] = round(rr_sum / n, 4)
    metrics["n"]   = n
    return metrics


def prf1(predictions: dict, gold_dict: dict) -> dict:
    """
    Compute Precision, Recall, F1 for a top-1 prediction dict.

    Args:
        predictions : {src_uri: pred_tgt_uri}
        gold_dict   : {src_uri: gold_tgt_uri}

    Returns dict with p, r, f1, tp, fp, fn.
    """
    n  = len(gold_dict)
    tp = sum(1 for s, t in gold_dict.items() if predictions.get(s) == t)
    fp = max(0, len(predictions) - tp)
    fn = n - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ── Threshold gating analysis ─────────────────────────────────────────────────

def threshold_gating_analysis(incorrect: dict, llm_preds: dict,
                               gold_dict: dict, n_total: int,
                               thresholds: list = None,
                               cache_dir: str = None,
                               ds_name: str = "mondo_doid") -> list:
    """
    Quantify oracle vs deployable LLM gate gap.

    For each threshold T: invoke LLM if top-1 lex score < T.
    Reports H@1, LLM call count, and delta vs oracle.

    The oracle gate achieves the same LLM call count as T=1.0
    but uses gold labels to select only the 179 beneficial calls
    (vs 1,523 total incorrect), yielding 0.974 vs 0.885.

    Args:
        incorrect  : {src: {true_match, all_candidates}}  (lex scores available)
        llm_preds  : {src: tgt}  oracle LLM predictions
        gold_dict  : {src: tgt}
        n_total    : total gold pairs
        thresholds : list of T values to evaluate

    Returns list of result dicts.
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    already_correct = n_total - len(incorrect)

    # LLM precision on oracle-called entities
    llm_fixes = sum(1 for src, pred in llm_preds.items()
                    if src in incorrect
                    and gold_dict.get(src) == pred)
    llm_precision = llm_fixes / max(len(llm_preds), 1)

    oracle_h1 = (already_correct + llm_fixes) / n_total

    # Get top-1 lex scores for all incorrect entities
    scores_info = []
    for src, data in incorrect.items():
        top_score  = data["all_candidates"][0][1] \
                     if data["all_candidates"] else 1.0
        llm_correct = gold_dict.get(src) == llm_preds.get(src)
        scores_info.append({
            "src":        src,
            "top_score":  top_score,
            "llm_correct": llm_correct,
        })

    results = []
    for T in thresholds:
        called     = [s for s in scores_info if s["top_score"] < T]
        llm_gained = sum(1 for s in called if s["llm_correct"])
        h1 = (already_correct + llm_gained) / n_total
        results.append({
            "T":          T,
            "H@1":        round(h1, 4),
            "llm_calls":  len(called),
            "call_pct":   round(len(called) / n_total * 100, 1),
            "delta_vs_oracle_pp": round((h1 - oracle_h1) * 100, 2),
        })

    # Print table
    n_inc = len(incorrect)
    print(f"\n{'='*65}")
    print(f"Threshold Gating Analysis — {ds_name}")
    print(f"{'='*65}")
    print(f"  n_total={n_total}  incorrect_after_lex={n_inc}  "
          f"oracle_calls={n_inc} ({n_inc/n_total*100:.1f}%)")
    print(f"  LLM fixes: {llm_fixes}/{n_inc} = {llm_precision*100:.1f}% precision")
    print(f"  Oracle H@1: {oracle_h1:.4f}  "
          f"Lex-only H@1: {already_correct/n_total:.4f}")
    print(f"\n{'T':>12} {'H@1':>8} {'calls':>10} {'%':>8} {'vs oracle':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['T']:>12.1f} {r['H@1']:>8.4f} "
              f"{r['llm_calls']:>10,} {r['call_pct']:>7.1f}% "
              f"{r['delta_vs_oracle_pp']:>+9.2f}pp")
    print(f"{'oracle':>12} {oracle_h1:>8.4f} "
          f"{n_inc:>10,} {n_inc/n_total*100:>7.1f}%  {'0.00pp':>10}")
    print("=" * 65)

    if cache_dir:
        save({"results": results, "oracle_h1": oracle_h1,
              "llm_precision": llm_precision, "llm_fixes": llm_fixes},
             Path(cache_dir) / f"threshold_gating_{ds_name}.pkl")

    return results


# ── Xref leakage audit ────────────────────────────────────────────────────────

def audit_xref_leakage(all_profiles: dict) -> bool:
    """
    Verify that cross-ontology xref values do NOT appear in
    the profile text fields used for retrieval (label/synonyms/definition).

    Returns True if clean (no leakage), False if leakage detected.
    """
    checks = [
        ("mondo_doid",   "left",  ["DOID", "doid.owl"]),
        ("mondo_doid",   "right", ["MONDO", "mondo.owl"]),
        ("uberon_emapa", "left",  ["EMAPA", "emapa"]),
        ("uberon_emapa", "right", ["UBERON", "uberon"]),
        ("mesh_snomed",  "left",  ["SNOMEDCT", "snomed"]),
        ("mesh_snomed",  "right", ["mesh", "MeSH"]),
    ]

    print("\nCross-ontology xref leakage audit:")
    print("=" * 65)
    leakage_found = False

    for ds, side, targets in checks:
        if ds not in all_profiles or side not in all_profiles[ds]:
            continue
        count = 0
        for uri, p in all_profiles[ds][side].items():
            profile_text = " ".join([
                p.get("label") or "",
                " ".join(p.get("synonyms") or []),
                p.get("definition") or "",
            ])
            for xref in (p.get("xrefs") or []):
                xref_val = xref.split(":")[-1] if ":" in xref else xref
                if any(t.lower() in xref.lower() for t in targets):
                    if xref_val in profile_text:
                        count += 1
        status = "⚠ LEAKAGE" if count > 0 else "✓ clean"
        print(f"  {ds} {side:<8}: {count:>6,} leaking xrefs → {status}")
        if count > 0:
            leakage_found = True

    print("=" * 65)
    if leakage_found:
        print("⚠ LEAKAGE DETECTED — rebuild profiles excluding xref fields")
    else:
        print("✓ NO LEAKAGE — profiles are clean")
    return not leakage_found


# ── Results table printer ─────────────────────────────────────────────────────

def print_results_table(results: dict):
    """Pretty-print a nested results dict."""
    print(f"\n{'='*80}")
    print(f"{'Dataset':<22} {'System':<28} "
          f"{'H@1':>7} {'H@5':>7} {'H@10':>7} {'MRR':>7}")
    print("-" * 80)
    for ds, systems in results.items():
        first = True
        for sys_name, r in systems.items():
            label = ds if first else ""
            first = False

            def fmt(v): return f"{v:.4f}" if v is not None else "—"
            print(f"{label:<22} {sys_name:<28} "
                  f"{fmt(r.get('h1')):>7} {fmt(r.get('h5')):>7} "
                  f"{fmt(r.get('h10')):>7} {fmt(r.get('mrr')):>7}")
        print()
    print("=" * 80)
