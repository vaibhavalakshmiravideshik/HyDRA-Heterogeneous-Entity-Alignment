"""
HyDRA — Main Pipeline
=====================
End-to-end runner for all five stages.

Usage:
    python run_hydra.py \
        --cache_dir /path/to/cache \
        --azure_key YOUR_KEY \
        --azure_url YOUR_URL \
        --mode zeroshot           # or: supervised, baselines, all

Environment variables (alternative to flags):
    AZURE_OPENAI_KEY
    AZURE_OPENAI_URL
"""

import os
import argparse
import torch
from pathlib import Path

from config import (DATASETS, BGE_ENCODER, BGE_PREFIX, LLM_MODEL,
                    TOP_K, RERANK_W_LEX, RERANK_W_COS,
                    MNRL_EPOCHS, MNRL_BATCH_SIZE, MNRL_LR,
                    MNRL_TEST_SIZE, LABEL_PREDS, SYN_PREDS, DEF_PREDS)

from utils.helpers  import save, load, exists, get_llm_client, build_label_map
from utils.data_loader import load_all_datasets, make_official_splits
from stage1.rotate_families  import run_stage1
from stage2.profiling        import run_stage2
from stage3.retrieval        import run_stage3
from stage4.reranking        import run_stage4
from stage5.supervised       import run_stage5
from eval.metrics            import (threshold_gating_analysis,
                                      audit_xref_leakage,
                                      print_results_table)
from baselines.baselines     import lexical_baseline, sapbert_eval, gea_baseline


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HyDRA Pipeline")
    parser.add_argument("--cache_dir", default="./cache",
                        help="Directory for cached intermediates")
    parser.add_argument("--azure_key", default=None)
    parser.add_argument("--azure_url", default=None)
    parser.add_argument("--llm_model", default=LLM_MODEL)
    parser.add_argument("--mode", default="all",
                        choices=["zeroshot", "supervised", "baselines",
                                 "all", "eval_only"])
    parser.add_argument("--oracle_gate", action="store_true",
                        default=True,
                        help="Use oracle gating for LLM (upper bound)")
    parser.add_argument("--lex_threshold", type=float, default=1.0,
                        help="Lex confidence threshold for deployable gate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # LLM client
    client, llm_model = get_llm_client(
        args.azure_key, args.azure_url, args.llm_model)

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        import torch
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load datasets ──────────────────────────────────────────────────────
    raw_path = cache_dir / "raw_triples.pkl"
    if exists(raw_path):
        print("Loading cached raw triples...")
        raw = load(raw_path)
    else:
        raw = load_all_datasets(DATASETS)
        save(raw, raw_path)

    # ── Build global label map ─────────────────────────────────────────────
    label_path = cache_dir / "uri2label.pkl"
    if exists(label_path):
        uri2label = load(label_path)
        print(f"Loaded uri2label: {len(uri2label):,} entries")
    else:
        print("Building URI label map...")
        uri2label, uri2syns, uri2def = build_label_map(
            raw, LABEL_PREDS, SYN_PREDS, DEF_PREDS)
        save(uri2label, label_path)
        save(uri2syns,  cache_dir / "uri2syns.pkl")
        save(uri2def,   cache_dir / "uri2def.pkl")
        print(f"Labels={len(uri2label):,}  Syns={len(uri2syns):,}  "
              f"Defs={len(uri2def):,}")

    # ── Official splits ────────────────────────────────────────────────────
    splits_path = cache_dir / "official_splits.pkl"
    if not exists(splits_path):
        print("\nCreating official 80/10/10 splits...")
        splits = make_official_splits(raw, test_size=0.20, seed=42)
        save(splits, splits_path)
        for ds, sp in splits.items():
            print(f"  {ds}: train={len(sp['train'])}  "
                  f"val={len(sp['val'])}  test={len(sp['test'])}")

    # ── Stage 1: Relation family induction ────────────────────────────────
    print("\n" + "=" * 60)
    rotate_results, family_mappings = run_stage1(
        raw, DATASETS, str(cache_dir),
        client, llm_model, device)

    # ── Stage 2: Entity profiling ──────────────────────────────────────────
    all_profiles = run_stage2(
        raw, rotate_results, DATASETS, str(cache_dir),
        client, llm_model, uri2label)

    # ── Xref leakage audit ─────────────────────────────────────────────────
    audit_xref_leakage(all_profiles)

    # ── Stage 3: BGE retrieval ─────────────────────────────────────────────
    bge_embs, zeroshot_bge, incorrect_bge = run_stage3(
        all_profiles, raw, str(cache_dir),
        BGE_PREFIX, BGE_ENCODER, TOP_K, device)

    print("\nBGE-only zero-shot results:")
    for ds, m in zeroshot_bge.items():
        print(f"  {ds}: H@1={m['hits1']:.4f}  H@5={m['hits5']:.4f}  "
              f"H@10={m['hits10']:.4f}  MRR={m['mrr']:.4f}")

    # ── Stage 4: Reranking (zero-shot) ────────────────────────────────────
    if args.mode in ("zeroshot", "all"):
        reranked, llm_preds, zs_metrics = run_stage4(
            all_profiles, incorrect_bge, raw, str(cache_dir),
            client, llm_model,
            w_lex=RERANK_W_LEX, w_cos=RERANK_W_COS,
            top_k=TOP_K,
            oracle_gate=args.oracle_gate,
            lex_threshold=args.lex_threshold)

        print("\n" + "=" * 60)
        print("Zero-shot final results (BGE + lex + LLM):")
        for ds, m in zs_metrics.items():
            print(f"  {ds}: H@1={m['hits1']:.4f}  H@5={m['hits5']:.4f}  "
                  f"H@10={m['hits10']:.4f}  MRR={m['mrr']:.4f}")

        # Threshold gating analysis (MONDO-DOID)
        if "mondo_doid" in incorrect_bge:
            threshold_gating_analysis(
                incorrect=incorrect_bge["mondo_doid"],
                llm_preds=llm_preds.get("mondo_doid", {}),
                gold_dict=dict(raw["mondo_doid"]["gold"]),
                n_total=zeroshot_bge["mondo_doid"]["n"],
                cache_dir=str(cache_dir),
                ds_name="mondo_doid",
            )

    # ── Stage 5: Supervised fine-tuning ───────────────────────────────────
    if args.mode in ("supervised", "all"):
        mnrl_results = run_stage5(
            all_profiles, raw, str(cache_dir),
            BGE_PREFIX, BGE_ENCODER,
            client, llm_model, device,
            test_size=MNRL_TEST_SIZE,
            epochs=MNRL_EPOCHS,
            batch_size=MNRL_BATCH_SIZE,
            lr=MNRL_LR)

        print("\nSupervised results (MNRL + lex + LLM):")
        for ds, r in mnrl_results.items():
            print(f"  {ds}: H@1={r['h1_mnrl_llm']:.4f}  "
                  f"(MNRL: {r['mnrl']['hits1']:.4f}  "
                  f"+lex: {r['h1_mnrl_lex']:.4f})")

    # ── Baselines ──────────────────────────────────────────────────────────
    if args.mode in ("baselines", "all"):
        print("\n" + "=" * 60)
        print("Running baseline systems...")

        for ds_name, data in raw.items():
            print(f"\n--- {ds_name} ---")
            profs_l = all_profiles[ds_name]["left"]
            profs_r = all_profiles[ds_name]["right"]

            lexical_baseline(data["gold"], profs_l, profs_r,
                             ds_name, str(cache_dir))
            sapbert_eval(ds_name, profs_l, profs_r,
                         data["gold"], str(cache_dir), device)

            if ds_name != "mesh_snomed":   # too large for KGE
                gea_baseline(ds_name, data, str(cache_dir),
                             device=device)

    print("\n✓ HyDRA pipeline complete")


if __name__ == "__main__":
    main()
