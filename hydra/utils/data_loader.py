"""
HyDRA Data Loader
=================
Load RDF triples and gold alignments from HuggingFace datasets.
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from utils.helpers import strip_uri


# ── Triple loader ─────────────────────────────────────────────────────────────

def load_triples(repo: str, filename: str) -> list:
    """
    Download a TSV triple file from HuggingFace and return as
    list of (head, relation, tail) string tuples.
    """
    path = hf_hub_download(repo_id=repo, filename=filename,
                            repo_type="dataset")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                data.append((
                    parts[0].strip("<>"),
                    parts[1].strip("<>"),
                    parts[2],
                ))
    return data


# ── Full dataset loader ───────────────────────────────────────────────────────

def load_dataset_raw(ds_name: str, cfg: dict) -> dict:
    """
    Load one benchmark dataset from HuggingFace.

    Returns a dict with keys:
        gold         : list of (src_uri, tgt_uri) tuples
        gold_left    : set of src URIs
        gold_right   : set of tgt URIs
        attr1        : attribute triples for KG1
        attr2        : attribute triples for KG2
        rel1         : relation triples for KG1
        rel2         : relation triples for KG2
    """
    repo   = cfg["repo"]
    col_a  = cfg["col_a"]
    col_b  = cfg["col_b"]

    print(f"  Loading {ds_name} from {repo}...")

    ds = load_dataset(repo, "alignments")
    df = pd.DataFrame(ds["ent_links"])
    gold = list(zip(
        df[col_a].str.strip("<>"),
        df[col_b].str.strip("<>"),
    ))

    attr1 = load_triples(repo, "attr_triples_1")
    attr2 = load_triples(repo, "attr_triples_2")
    rel1  = load_triples(repo, "rel_triples_1")
    rel2  = load_triples(repo, "rel_triples_2")

    print(f"    gold={len(gold):,}  "
          f"attr1={len(attr1):,}  rel1={len(rel1):,}")

    return {
        "gold":        gold,
        "gold_left":   set(p[0] for p in gold),
        "gold_right":  set(p[1] for p in gold),
        "attr1":       attr1,
        "attr2":       attr2,
        "rel1":        rel1,
        "rel2":        rel2,
    }


def load_all_datasets(datasets_cfg: dict) -> dict:
    """Load all benchmark datasets."""
    raw = {}
    for ds_name, cfg in datasets_cfg.items():
        raw[ds_name] = load_dataset_raw(ds_name, cfg)
    return raw


# ── Official splits ───────────────────────────────────────────────────────────

def make_official_splits(raw: dict, test_size: float = 0.20,
                         seed: int = 42) -> dict:
    """
    Create 80/10/10 stratified splits for each dataset.
    Returns dict: ds_name -> {train, val, test} DataFrames.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    splits = {}
    for ds_name, data in raw.items():
        gold = data["gold"]
        df   = pd.DataFrame(gold, columns=["src", "tgt"])
        idx  = np.arange(len(df))

        train_idx, rest_idx = train_test_split(
            idx, test_size=test_size * 2, random_state=seed)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=0.50, random_state=seed)

        splits[ds_name] = {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "val":   df.iloc[val_idx].reset_index(drop=True),
            "test":  df.iloc[test_idx].reset_index(drop=True),
        }
        tr = len(splits[ds_name]["train"])
        va = len(splits[ds_name]["val"])
        te = len(splits[ds_name]["test"])
        print(f"  {ds_name}: train={tr}  val={va}  test={te}")

    return splits
