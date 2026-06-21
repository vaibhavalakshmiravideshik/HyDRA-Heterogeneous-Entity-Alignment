# HyDRA: Heterogeneity-Aware Dynamic Retrieval and Alignment for Biomedical Entity Alignment

This repository contains the code for **HyDRA**, a heterogeneity-aware biomedical entity alignment framework, together with links to the three benchmark datasets introduced in the accompanying paper:

**HyDRA: Heterogeneity-Aware Dynamic Retrieval and Alignment for Biomedical Entity Alignment with Benchmark Evaluation**  
Vaibhava Lakshmi Ravideshik and Mayank Kejriwal  
USC Information Sciences Institute

## Overview

Entity alignment (EA) identifies equivalent entities across knowledge graphs and ontologies. Most standard EA benchmarks focus on relatively homogeneous graph pairs. HyDRA is built for **heterogeneous biomedical alignment**, where source and target ontologies differ substantially in schema, purpose, graph structure, lexical conventions, and attribute coverage.

The project has two main contributions:

1. **Three heterogeneous biomedical EA benchmarks** spanning disease, anatomy, and medical terminology.
2. **HyDRA**, a five-stage alignment pipeline combining unsupervised relation-family induction, schema-agnostic entity profiling, dense retrieval, heterogeneity-aware lexical reranking, and targeted LLM reasoning.

## Benchmarks

The three benchmark datasets are hosted on Hugging Face:

| Benchmark | Hugging Face | Gold alignments | Heterogeneity pattern |
|---|---|---:|---|
| **MeSH-SNOMED-15K** | https://huggingface.co/datasets/vaibhavalakshmiravideshik/mesh-snomed-entity-alignment-15k | 15,000 | Cross-purpose mismatch: literature indexing vs clinical terminology |
| **EMAPA-UBERON-4K** | https://huggingface.co/datasets/vaibhavalakshmiravideshik/emapa-uberon-4k | 4,079 | Cross-species and cross-temporal anatomy alignment |
| **MONDO-DOID-12K** | https://huggingface.co/datasets/vaibhavalakshmiravideshik/mondo-doid-12k | 11,812 | Schema simplicity mismatch in disease ontologies |

### Benchmark summaries

#### 1. MeSH-SNOMED-15K
- Aligns **MeSH** descriptors to **SNOMED CT** concepts.
- Designed to capture **cross-purpose heterogeneity**.
- MeSH is optimized for biomedical literature indexing; SNOMED CT is optimized for clinical documentation.
- This is the hardest large-scale benchmark in the suite because the graphs differ in purpose, structure, and granularity, and the target-side background context is very large.

#### 2. EMAPA-UBERON-4K
- Aligns **EMAPA** developmental mouse anatomy entities to **Uberon** cross-species anatomy entities.
- Captures **cross-species** and **cross-temporal** mismatch.
- EMAPA includes stage-specific developmental structure; Uberon is more broadly cross-species and not grounded in the same temporal representation.

#### 3. MONDO-DOID-12K
- Aligns **MONDO** disease entities to **DOID** disease entities.
- Captures **schema asymmetry** and **rich-vs-simple ontology mismatch**.
- MONDO contains richer cross-source harmonization and more diverse relations, while DOID is comparatively simpler.

## What makes these benchmarks different

These datasets are intended to move beyond homogeneous EA settings.

- They embed gold alignments in **full ontological context**.
- They include large **background candidate pools** rather than only closed-world aligned subsets.
- They require systems to bridge **schema mismatch**, **lexical variation**, **granularity differences**, and **graph asymmetry**.
- They target realistic biomedical interoperability problems rather than generic graph-matching alone.

Approximate benchmark scale from the paper:

- **MeSH-SNOMED-15K**: 15,000 gold pairs, with SNOMED-side background at roughly **530K** entities.
- **EMAPA-UBERON-4K**: 4,079 gold pairs, with target background around **8K** entities.
- **MONDO-DOID-12K**: 11,812 gold pairs, with target background around **12K** entities.

## HyDRA pipeline

HyDRA is implemented as a five-stage pipeline:

1. **Unsupervised Relation Family Induction**  
   Learns relation embeddings independently per KG using RotatE, clusters them with KMeans, and aligns relation families across KGs using anchor detection plus Procrustes / optimal transport.

2. **Schema-Agnostic Entity Profiling**  
   Uses LLM-based predicate role classification to build entity profiles from labels, synonyms, definitions, and resolved neighborhood information.

3. **Asymmetry-Aware Candidate Retrieval**  
   Encodes entity profiles with **BAAI/bge-large-en-v1.5** and retrieves top-k candidates by cosine similarity.

4. **Heterogeneity-Aware Reranking**  
   Applies lexical reranking using word overlap, character bigrams, and character trigrams; optionally invokes targeted LLM reasoning for ambiguous cases.

5. **Global Resolution**  
   Produces final top-1 alignments.

## Main results reported in the paper

### Gold-filtered pool, zero-shot, transductive

| Benchmark | Hits@1 | Hits@5 | Hits@10 | MRR |
|---|---:|---:|---:|---:|
| MONDO-DOID-12K | 0.974 | 0.980 | 0.981 | 0.976 |
| EMAPA-UBERON-4K | 0.975 | 0.983 | 0.984 | 0.978 |
| MeSH-SNOMED-15K | 0.803 | 0.829 | 0.831 | 0.814 |

### Open full-pool, zero-shot

| Benchmark | Pool size | SapBERT H@1 | HyDRA H@1 |
|---|---:|---:|---:|
| MONDO-DOID-12K | 12,127 | 0.919 | 0.895 |
| EMAPA-UBERON-4K | 8,078 | 0.886 | 0.853 |
| MeSH-SNOMED-15K | 530,013 | 0.573 | 0.595 |

### Important evaluation note

The paper distinguishes between:

- **oracle-gated zero-shot upper bounds**, where gold labels identify which cases should receive LLM reasoning, and
- **deployable settings**, where that oracle is unavailable.

This distinction is especially important for interpreting the Stage 4 LLM component.

## Repository structure

After extraction, the repository contains the implementation under `hydra/`:

```text
HyDRA-Hetergeneous-Entity-Alignment/
├── README.md
└── hydra/
    ├── README.md
    ├── __init__.py
    ├── config.py
    ├── requirements.txt
    ├── run_hydra.py
    ├── baselines/
    ├── eval/
    ├── stage1/
    ├── stage2/
    ├── stage3/
    ├── stage4/
    ├── stage5/
    └── utils/
```

### Key files

- `hydra/run_hydra.py` — main entry point for zero-shot, supervised, and baseline runs.
- `hydra/config.py` — central configuration for models, datasets, and hyperparameters.
- `hydra/stage1/rotate_families.py` — relation family induction.
- `hydra/stage2/profiling.py` — predicate classification and entity profile construction.
- `hydra/stage3/retrieval.py` — dense retrieval.
- `hydra/stage4/reranking.py` — lexical and LLM reranking.
- `hydra/stage5/supervised.py` — supervised fine-tuning logic.
- `hydra/eval/metrics.py` — Hits@k, MRR, P/R/F1, and evaluation helpers.
- `hydra/baselines/baselines.py` — lexical, SapBERT, and graph-embedding baselines.

## Installation

From the `hydra/` directory:

```bash
pip install -r requirements.txt
```

Core dependencies listed in the repository include:

- `torch`
- `sentence-transformers`
- `pykeen`
- `POT`
- `scikit-learn`
- `openai`
- `datasets`
- `huggingface-hub`
- `ontoaligner`

## Running the code

From inside `hydra/`:

### Full pipeline

```bash
python run_hydra.py --cache_dir ./cache --mode all
```

### Zero-shot only

```bash
python run_hydra.py --cache_dir ./cache --mode zeroshot
```

### Supervised only

```bash
python run_hydra.py --cache_dir ./cache --mode supervised
```

### Baselines only

```bash
python run_hydra.py --cache_dir ./cache --mode baselines
```

## LLM / API configuration

The code expects Azure OpenAI credentials, either through arguments or environment variables.

Environment variables:

```bash
export AZURE_OPENAI_KEY="your_key"
export AZURE_OPENAI_URL="your_endpoint"
```

The default LLM configured in the code is `gpt-4o`.

## Dataset loading

Dataset repository identifiers are currently configured in `hydra/config.py` as:

- `vaibhavalakshmiravideshik/mondo-doid-12k`
- `vaibhavalakshmiravideshik/emapa-uberon-4k`
- `vaibhavalakshmiravideshik/mesh-snomed-entity-alignment-15k`

These correspond to the public Hugging Face dataset pages linked above.

## Notes on splits and evaluation

The manuscript discusses a recommended **80/10/10** benchmark split for future work, while the current code configuration also reflects the earlier **80/20** supervised setting used in parts of the paper. When reporting results from this repository, it is important to state clearly:

- whether the experiment is **zero-shot** or **supervised**,
- whether it is **transductive** or **inductive**,
- whether it uses the **gold-filtered pool** or the **open full pool**, and
- which split protocol was used.

## Licensing and data access

### Code

Please add your preferred repository license if you plan to distribute the code formally.

### Datasets and ontology licensing

- **MeSH** is maintained by the U.S. National Library of Medicine.
- **SNOMED CT** has jurisdiction-specific licensing requirements through **SNOMED International**: https://www.snomed.org
- **Uberon** and **EMAPA** are distributed under their respective open licenses.
- **MONDO** and **DOID** are distributed under their respective open licenses.

### Important note on MeSH-SNOMED-15K

Because **SNOMED CT** is license-restricted, the complete benchmark may not always be directly redistributable in the same way as the fully open datasets. Please follow the source ontology licensing terms when reconstructing, redistributing, or using this benchmark.

## Suggested citation

```bibtex
@article{ravideshik2026hydra,
  title={HyDRA: Heterogeneity-Aware Dynamic Retrieval and Alignment for Biomedical Entity Alignment with Benchmark Evaluation},
  author={Ravideshik, Vaibhava Lakshmi and Kejriwal, Mayank},
  journal={Journal of Web Semantics},
  year={2026},
  note={Under review}
}
```

## Links

- **GitHub repository**: https://github.com/vaibhavalakshmiravideshik/HyDRA-Hetergeneous-Entity-Alignment
- **MeSH-SNOMED-15K**: https://huggingface.co/datasets/vaibhavalakshmiravideshik/mesh-snomed-entity-alignment-15k
- **EMAPA-UBERON-4K**: https://huggingface.co/datasets/vaibhavalakshmiravideshik/emapa-uberon-4k
- **MONDO-DOID-12K**: https://huggingface.co/datasets/vaibhavalakshmiravideshik/mondo-doid-12k
- **SNOMED International**: https://www.snomed.org

## Status

The repository archive originally stored in the repo has been extracted into the repository folder, and the zip file has been removed.
