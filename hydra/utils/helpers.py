"""
HyDRA Utilities
===============
Shared helpers: I/O, URI handling, label maps, LLM client.
"""

import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from openai import OpenAI


# ── I/O ───────────────────────────────────────────────────────────────────────

def save(obj, path):
    """Pickle-save an object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    """Pickle-load an object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def exists(path):
    return Path(path).exists()


# ── URI / value helpers ───────────────────────────────────────────────────────

def strip_uri(uri: str) -> str:
    """Remove < > angle brackets and whitespace from a URI string."""
    return uri.strip("<>").strip()


def local_name(uri: str) -> str:
    """Return the local fragment of a URI (after # or last /)."""
    uri = strip_uri(uri)
    return re.split(r"[/#]", uri)[-1].replace("_", " ")


def extract_literal(val: str) -> str:
    """Strip surrounding quotes from a literal RDF value."""
    val = val.strip()
    if val.startswith('"'):
        val = val[1:]
        if '"' in val:
            val = val[: val.index('"')]
    return val.strip()


# ── Label map ─────────────────────────────────────────────────────────────────

def build_label_map(raw_datasets: dict, label_preds: set,
                    syn_preds: set, def_preds: set):
    """
    Build three global maps from all loaded RDF triples:
        uri2label  : URI -> primary label string
        uri2syns   : URI -> list of synonym strings
        uri2def    : URI -> definition string
    """
    uri2label: dict = {}
    uri2syns:  dict = defaultdict(list)
    uri2def:   dict = {}

    for ds_name, data in raw_datasets.items():
        for triples in [data["attr1"], data["attr2"],
                        data["rel1"],  data["rel2"]]:
            for h, r, v in triples:
                val = extract_literal(v)
                if not val or len(val) < 2:
                    continue
                if r in label_preds and h not in uri2label:
                    uri2label[h] = val
                elif r in syn_preds and len(uri2syns[h]) < 5:
                    uri2syns[h].append(val)
                elif r in def_preds and h not in uri2def:
                    uri2def[h] = val

    return uri2label, dict(uri2syns), uri2def


def resolve(uri: str, uri2label: dict) -> str:
    """Return human-readable label for a URI, falling back to local name."""
    return uri2label.get(uri, local_name(uri))


# ── LLM client ────────────────────────────────────────────────────────────────

def get_llm_client(azure_key: str = None, azure_url: str = None,
                   model: str = "gpt-4o") -> tuple:
    """
    Return (client, model_name).
    Reads AZURE_OPENAI_KEY and AZURE_OPENAI_URL from environment
    if not passed directly.
    """
    key = azure_key or os.environ.get("AZURE_OPENAI_KEY", "")
    url = azure_url or os.environ.get("AZURE_OPENAI_URL", "")
    client = OpenAI(api_key=key, base_url=url)
    return client, model


def llm_call(client, model: str, prompt: str,
             max_tokens: int = 2000,
             temperature: float = 0.1) -> str:
    """Single LLM call, returns text response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def parse_json_response(text: str):
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Lexical scoring ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) if text else set()


def _ngrams(text: str, n: int) -> set:
    t = (text or "").lower()
    return set(t[i: i + n] for i in range(len(t) - n + 1))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def lex_score(name_a: str, name_b: str,
              w_word: float = 0.30,
              w_bi:   float = 0.35,
              w_tri:  float = 0.35) -> float:
    """
    Weighted lexical similarity:
      word Jaccard + bigram Jaccard + trigram Jaccard.
    """
    if not name_a or not name_b:
        return 0.0
    return (w_word * _jaccard(_tokenize(name_a),  _tokenize(name_b)) +
            w_bi   * _jaccard(_ngrams(name_a, 2), _ngrams(name_b, 2)) +
            w_tri  * _jaccard(_ngrams(name_a, 3), _ngrams(name_b, 3)))
