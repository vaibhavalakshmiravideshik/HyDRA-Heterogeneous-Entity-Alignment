"""
HyDRA Configuration
===================
Central configuration for all pipeline stages.
"""

# ── LLM / API ─────────────────────────────────────────────────────────────────
AZURE_KEY  = ""   # set via environment variable AZURE_OPENAI_KEY
AZURE_URL  = ""   # set via environment variable AZURE_OPENAI_URL
LLM_MODEL  = "gpt-4o"

# ── Encoders ──────────────────────────────────────────────────────────────────
BGE_ENCODER = "BAAI/bge-large-en-v1.5"
BGE_PREFIX  = "Represent this biomedical entity for retrieval: "
SAPBERT     = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = {
    "mondo_doid": {
        "repo":           "vaibhavalakshmiravideshik/mondo-doid-12k",
        "col_a":          "mondo_uri",
        "col_b":          "doid_uri",
        "k_rotate":       4,
        "rotate_epochs":  50,
        "rotate_dim":     64,
    },
    "uberon_emapa": {
        "repo":           "vaibhavalakshmiravideshik/emapa-uberon-4k",
        "col_a":          "uberon_uri",
        "col_b":          "emapa_uri",
        "k_rotate":       4,
        "rotate_epochs":  50,
        "rotate_dim":     64,
    },
    "mesh_snomed": {
        "repo":           "vaibhavalakshmiravideshik/mesh-snomed-entity-alignment-15k",
        "col_a":          "mesh_uri",
        "col_b":          "snomed_uri",
        "k_rotate":       4,
        "rotate_epochs":  30,
        "rotate_dim":     64,
    },
}

# ── RDF predicate sets ────────────────────────────────────────────────────────
LABEL_PREDS = {
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://id.nlm.nih.gov/mesh/vocab#prefLabel",
}
SYN_PREDS = {
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
}
DEF_PREDS = {
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://purl.obolibrary.org/obo/IAO_0000115",
}

# ── Stage 3 retrieval ─────────────────────────────────────────────────────────
TOP_K = 10

# ── Stage 4 lexical reranking ─────────────────────────────────────────────────
LEX_WEIGHTS = {
    "word_jaccard":   0.30,
    "bigram_jaccard": 0.35,
    "trigram_jaccard": 0.35,
}
RERANK_W_LEX = 0.80
RERANK_W_COS = 0.20

# ── Stage 5 MNRL fine-tuning ──────────────────────────────────────────────────
MNRL_EPOCHS     = 5
MNRL_BATCH_SIZE = 16
MNRL_LR         = 2e-5
MNRL_TEST_SIZE  = 0.20
