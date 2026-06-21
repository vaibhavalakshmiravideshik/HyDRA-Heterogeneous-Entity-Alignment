"""
HyDRA Stage 2 — Schema-Agnostic Entity Profiling
=================================================
Classifies attribute predicates into semantic roles via LLM,
then constructs entity profiles with resolved family neighborhoods
and [MISSING] tokens for absent fields.
"""

import json
from pathlib import Path
from collections import defaultdict

from utils.helpers import (save, load, exists, extract_literal,
                            llm_call, parse_json_response, local_name)


# ── LLM predicate classification ──────────────────────────────────────────────

def classify_predicates(attr_triples: list, ds_side_name: str,
                         cache_dir: str, client, model: str) -> dict:
    """
    Classify each attribute predicate into one of:
        label / synonym / definition / xref / metadata / ignore

    Returns dict: {predicate_uri: role_string}
    """
    cache_key = Path(cache_dir) / f"pred_roles_{ds_side_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_side_name}: roles loaded from cache")
        return load(cache_key)

    # Collect up to 5 sample values per predicate
    pred_samples = defaultdict(list)
    for h, r, v in attr_triples:
        val = extract_literal(v)
        if val and len(pred_samples[r]) < 5:
            pred_samples[r].append(val[:80])

    pred_list = "\n".join([
        f"- Predicate: {p}\n  Samples: {pred_samples[p]}"
        for p in pred_samples
    ])

    prompt = f"""You are analyzing RDF attribute triples from a biomedical ontology.
Classify each predicate into exactly one of:
label / synonym / definition / xref / metadata / ignore

Rules:
- label: primary name, typically one per entity, short text
- synonym: alternative names, multiple per entity
- definition: textual description, longer text
- xref: cross-references like DB:ID patterns (e.g. DOID:1234)
- metadata: dates, booleans, numbers, internal IDs
- ignore: anything not useful for entity alignment

Return ONLY a valid JSON object mapping each predicate URI to its role.

Predicates:
{pred_list}"""

    try:
        raw_resp = llm_call(client, model, prompt,
                            max_tokens=2000, temperature=0.1)
        roles = parse_json_response(raw_resp)
        if not isinstance(roles, dict):
            raise ValueError("Expected a dict")
    except Exception as e:
        print(f"  LLM fallback for {ds_side_name}: {e}")
        roles = {p: "ignore" for p in pred_samples}

    from collections import Counter
    counts = Counter(roles.values())
    print(f"  {ds_side_name} roles: {dict(counts)}")

    save(roles, cache_key)
    return roles


# ── Profile builder ───────────────────────────────────────────────────────────

def build_profiles(attr_triples: list, rel_triples: list,
                   roles: dict, fam_map: dict,
                   gold_set: set, ds_side_name: str,
                   cache_dir: str, uri2label: dict) -> dict:
    """
    Build entity profiles for all entities.

    Each profile dict has:
        label      : str or None
        synonyms   : list[str]
        definition : str or None
        xrefs      : list[str]   (stored but NOT used in retrieval text)
        neighbors  : list of {fam, rel_label, tgt_label, tgt_uri}
        has_label  : bool
        has_def    : bool
        has_syn    : bool
    """
    cache_key = Path(cache_dir) / f"profiles_{ds_side_name}.pkl"
    if exists(cache_key):
        print(f"  {ds_side_name}: profiles loaded from cache")
        return load(cache_key)

    label_preds = {p for p, r in roles.items() if r == "label"}
    syn_preds   = {p for p, r in roles.items() if r == "synonym"}
    def_preds   = {p for p, r in roles.items() if r == "definition"}
    xref_preds  = {p for p, r in roles.items() if r == "xref"}

    # Collect text attributes
    text = defaultdict(lambda: {
        "label": None, "synonyms": [], "definition": None, "xrefs": []
    })
    for h, r, v in attr_triples:
        val = extract_literal(v)
        if not val:
            continue
        if r in label_preds and text[h]["label"] is None:
            text[h]["label"] = val
        elif r in syn_preds and len(text[h]["synonyms"]) < 5:
            text[h]["synonyms"].append(val)
        elif r in def_preds and text[h]["definition"] is None:
            text[h]["definition"] = val[:300]
        elif r in xref_preds and len(text[h]["xrefs"]) < 3:
            text[h]["xrefs"].append(val)

    # Fallback labels from global uri2label map
    for uri in gold_set:
        if text[uri]["label"] is None and uri in uri2label:
            text[uri]["label"] = uri2label[uri]

    # Build relation neighborhoods with resolved labels
    neighbors = defaultdict(list)
    for h, r, t in rel_triples:
        fam = fam_map.get(r, -1)
        rel_label = uri2label.get(r, local_name(r))
        tgt_label = uri2label.get(t, None)
        neighbors[h].append({
            "fam":       fam,
            "rel_label": rel_label,
            "tgt_label": tgt_label,
            "tgt_uri":   t,
        })

    # Assemble profiles
    all_ents = gold_set | set(text.keys()) | set(neighbors.keys())
    profiles = {}
    for uri in all_ents:
        t = text.get(uri, {"label": None, "synonyms": [],
                            "definition": None, "xrefs": []})
        profiles[uri] = {
            "label":      t["label"],
            "synonyms":   t["synonyms"],
            "definition": t["definition"],
            "xrefs":      t["xrefs"],
            "neighbors":  neighbors.get(uri, []),
            "has_label":  t["label"] is not None,
            "has_def":    t["definition"] is not None,
            "has_syn":    len(t["synonyms"]) > 0,
        }

    # Coverage stats
    n = max(len(gold_set), 1)
    n_lbl = sum(1 for u in gold_set if profiles.get(u, {}).get("has_label"))
    n_def = sum(1 for u in gold_set if profiles.get(u, {}).get("has_def"))
    n_syn = sum(1 for u in gold_set if profiles.get(u, {}).get("has_syn"))
    n_rel = sum(1 for u in gold_set if profiles.get(u, {}).get("neighbors"))
    print(f"  {ds_side_name}: {len(gold_set):,} gold | "
          f"label={n_lbl/n*100:.0f}% def={n_def/n*100:.0f}% "
          f"syn={n_syn/n*100:.0f}% rel={n_rel/n*100:.0f}%")

    save(profiles, cache_key)
    return profiles


# ── Profile → text string ─────────────────────────────────────────────────────

def profile_to_text(p: dict, bge_prefix: str = "",
                    use_relations: bool = True) -> str:
    """
    Convert a profile dict into an encoder-ready text string.
    Absent fields are encoded as [MISSING].
    """
    parts = []
    parts.append(f"name: {p['label']}" if p.get("has_label")
                 else "name: [MISSING]")
    parts.append(
        f"synonyms: {' | '.join((p.get('synonyms') or [])[:3])}"
        if p.get("has_syn") else "synonyms: [MISSING]")
    defn = p.get("definition")
    parts.append(f"definition: {defn[:200]}" if defn
                 else "definition: [MISSING]")

    if use_relations and p.get("neighbors"):
        rel_strs = []
        for nb in p["neighbors"][:5]:
            if nb.get("tgt_label"):
                rel_strs.append(
                    f"{nb['rel_label']}: {nb['tgt_label']}")
        if rel_strs:
            parts.append(f"relations: {' | '.join(rel_strs)}")
        else:
            parts.append("relations: [MISSING]")
    elif use_relations:
        parts.append("relations: [MISSING]")

    return bge_prefix + " | ".join(parts)


# ── Stage 2 entrypoint ────────────────────────────────────────────────────────

def run_stage2(raw: dict, rotate_results: dict,
               datasets_cfg: dict, cache_dir: str,
               client, model: str, uri2label: dict) -> dict:
    """
    Run Stage 2 (predicate classification + profile building) for all datasets.

    Returns all_profiles: {ds_name: {left: profiles, right: profiles}}
    """
    print("\n" + "=" * 60)
    print("Stage 2: Schema-Agnostic Entity Profiling")
    print("=" * 60)

    all_profiles = {}

    for ds_name, data in raw.items():
        print(f"\n{ds_name}:")
        rot = rotate_results[ds_name]

        roles_left = classify_predicates(
            data["attr1"], f"{ds_name}_left", cache_dir, client, model)
        roles_right = classify_predicates(
            data["attr2"], f"{ds_name}_right", cache_dir, client, model)

        profs_left = build_profiles(
            data["attr1"], data["rel1"],
            roles_left, rot["left"]["fam_map"],
            data["gold_left"], f"{ds_name}_left",
            cache_dir, uri2label)

        profs_right = build_profiles(
            data["attr2"], data["rel2"],
            roles_right, rot["right"]["fam_map"],
            data["gold_right"], f"{ds_name}_right",
            cache_dir, uri2label)

        all_profiles[ds_name] = {
            "left":  profs_left,
            "right": profs_right,
        }

    save(all_profiles, Path(cache_dir) / "all_profiles.pkl")
    print("\n✓ Stage 2 complete")
    return all_profiles
