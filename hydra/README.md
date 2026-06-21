# HyDRA: Heterogeneity-Aware Dynamic Retrieval and Alignment

Implementation of **HyDRA**, the five-stage zero-shot baseline for heterogeneous biomedical entity alignment, accompanying the paper:

> *HyDRA: Heterogeneity-Aware Dynamic Retrieval and Alignment for Biomedical Entity Alignment with Benchmark Evaluation*  
> Vaibhava Lakshmi Ravideshik and Mayank Kejriwal  
> USC Information Sciences Institute  
> Journal of Web Semantics (under review)

---

## Benchmarks

Three large-scale heterogeneous biomedical EA benchmarks on HuggingFace:

| Benchmark | Pairs | Background | Heterogeneity |
|-----------|-------|------------|---------------|
| [MONDO-DOID-12K](https://huggingface.co/datasets/vaibhavalakshmiravideshik/mondo-doid-12k) | 11,812 | 12K | Schema simplicity mismatch |
| [EMAPA-UBERON-4K](https://huggingface.co/datasets/vaibhavalakshmiravideshik/emapa-uberon-4k) | 4,079 | 8K | Cross-species/temporal |
| [MeSH-SNOMED-15K](https://huggingface.co/datasets/vaibhavalakshmiravideshik/mesh-snomed-entity-alignment-15k) | 15,000 | 530K | Cross-purpose schema |

---

## Pipeline

```
KG₁, KG₂ (RDF triples)
    │
    ▼
Stage 1 — Unsupervised Relation Family Induction
    RotatE + KMeans + Procrustes/OT
    │  family labels / relation anchors
    ▼
Stage 2 — Schema-Agnostic Entity Profiling
    LLM predicate classifier + [MISSING] tokens
    │
    ▼
Stage 3 — Asymmetry-Aware Candidate Retrieval
    BGE-large-en-v1.5 · cosine top-k=10
    │  top-k candidates
    ▼
Stage 4 — Heterogeneity-Aware Reranking
    Lex (Jaccard, bigram, trigram) + LLM reasoning
    │     4–22% of entities invoke GPT-4o
    ▼
Stage 5 — Global Resolution
    Greedy top-1 assignment
    │
    ▼
Alignments {(e₁, e₂)}
```

**Supervised variant**: Stage 3 uses BGE fine-tuned with MNRL on 80% of gold pairs.

---

## Key Results

### Gold-filtered pool (zero-shot, transductive)

| Dataset | H@1 | H@5 | H@10 | MRR |
|---------|-----|-----|------|-----|
| MONDO-DOID-12K | **0.974** | 0.980 | 0.981 | 0.976 |
| EMAPA-UBERON-4K | **0.975** | 0.983 | 0.984 | 0.978 |
| MeSH-SNOMED-15K | **0.803** | 0.829 | 0.831 | 0.814 |

> **Note on LLM gating**: Zero-shot H@1 figures use oracle gating (gold labels identify incorrectly ranked entities). The deployable lex-reranking H@1 is 0.870/0.946/0.659; the oracle upper bound is 0.974/0.975/0.803. The 8.9pp gap on MONDO-DOID arises from 37% LLM precision on incorrectly ranked entities — see Table 5 (threshold gating analysis) in the paper.

### Open full-pool (zero-shot)

| Dataset | Pool size | SapBERT-ZS H@1 | HyDRA-ZS H@1 |
|---------|-----------|----------------|--------------|
| MONDO-DOID-12K | 12,127 | **0.919** | 0.895 |
| EMAPA-UBERON-4K | 8,078 | **0.886** | 0.853 |
| MeSH-SNOMED-15K | 530,013 | 0.573 | **0.595** |

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Azure OpenAI credentials:
```bash
export AZURE_OPENAI_KEY="your_key"
export AZURE_OPENAI_URL="your_endpoint"
```

> **MeSH-SNOMED-15K**: SNOMED CT requires a license from [SNOMED International](https://www.snomed.org). The benchmark cannot be redistributed directly. Reconstruction scripts will be provided with the camera-ready release.

---

## Usage

### Full pipeline (zero-shot + supervised + baselines)

```bash
python run_hydra.py \
    --cache_dir ./cache \
    --mode all
```

### Zero-shot only

```bash
python run_hydra.py --cache_dir ./cache --mode zeroshot
```

### Deployable threshold gate (no oracle)

```bash
python run_hydra.py \
    --cache_dir ./cache \
    --mode zeroshot \
    --oracle_gate False \
    --lex_threshold 0.9
```

### Supervised fine-tuning only

```bash
python run_hydra.py --cache_dir ./cache --mode supervised
```

---

## Repository structure

```
hydra/
├── config.py                   # Central configuration
├── run_hydra.py                # Main pipeline runner
├── requirements.txt
│
├── utils/
│   ├── helpers.py              # I/O, URI handling, LLM client, lex scoring
│   └── data_loader.py          # HuggingFace triple loading, splits
│
├── stage1/
│   └── rotate_families.py      # RotatE + KMeans + Procrustes/OT
│
├── stage2/
│   └── profiling.py            # LLM predicate classification, profile builder
│
├── stage3/
│   └── retrieval.py            # BGE encoding, cosine top-k retrieval
│
├── stage4/
│   └── reranking.py            # Lex reranking, LLM candidate reasoning
│
├── stage5/
│   └── supervised.py           # MNRL fine-tuning, supervised evaluation
│
├── eval/
│   └── metrics.py              # Hits@k, MRR, P/R/F1, threshold analysis
│
└── baselines/
    └── baselines.py            # Lexical, SapBERT, GraphEmbeddingAligner
```

---

## Compute

All experiments run on a single NVIDIA A100-SXM4-80GB GPU.

| Step | Time |
|------|------|
| RotatE (MONDO-DOID) | ~10 min |
| RotatE (MeSH-SNOMED, 30 epochs) | ~28 min |
| BGE encoding | ~1 min per ontology side |
| MNRL fine-tuning (MONDO-DOID) | ~31 min |
| LLM reasoning (MONDO-DOID, 483 calls) | ~8 min |
| LLM reasoning (MeSH-SNOMED, 2,911 calls) | ~49 min |

---

## Citation

```bibtex
@article{ravideshik2026hydra,
  title={{HyDRA}: Heterogeneity-Aware Dynamic Retrieval and Alignment
         for Biomedical Entity Alignment with Benchmark Evaluation},
  author={Ravideshik, Vaibhava Lakshmi and Kejriwal, Mayank},
  journal={Journal of Web Semantics},
  year={2026},
  note={Under review}
}
```

---

## License

Code: MIT  
Datasets: see individual ontology licenses (MeSH: NLM; SNOMED CT: SNOMED International; Uberon/EMAPA: CC BY; MONDO/DOID: CC BY/CC0).
