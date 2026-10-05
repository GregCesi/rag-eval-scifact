"""Tests des étiquettes d'origine par paire (claim, document) du split test (EXE-124).

Données fabriquées à la main, au format exact de `claims_dev.jsonl` et du
corpus sentence-splitté d'origine (`data/scifact/origine/data/corpus.jsonl`) —
jamais les vraies données de `data/scifact/`.
"""

from __future__ import annotations

import pytest

from rag_eval_scifact import origin_labels

# Claim 1 : un document SUPPORT (deux groupes de preuve, cinq phrases en tout).
# Claim 2 : un document CONTRADICT, un document SANS_PREUVE (cité sans preuve).
# Claim 3 : aucune preuve du tout (tous ses documents cités sont SANS_PREUVE).
CLAIMS = {
    "1": {
        "id": 1,
        "claim": "claim un",
        "evidence": {
            "100": [
                {"sentences": [0, 2], "label": "SUPPORT"},
                {"sentences": [3], "label": "SUPPORT"},
            ]
        },
        "cited_doc_ids": [100],
    },
    "2": {
        "id": 2,
        "claim": "claim deux",
        "evidence": {"200": [{"sentences": [1], "label": "CONTRADICT"}]},
        "cited_doc_ids": [200, 201],
    },
    "3": {
        "id": 3,
        "claim": "claim trois",
        "evidence": {},
        "cited_doc_ids": [300],
    },
}

CORPUS = {
    "100": {
        "doc_id": 100,
        "title": "Doc 100",
        "abstract": ["phrase 0", "phrase 1", "phrase 2", "phrase 3"],
    },
    "200": {
        "doc_id": 200,
        "title": "Doc 200",
        "abstract": ["phrase a", "phrase b"],
    },
    "201": {
        "doc_id": 201,
        "title": "Doc 201",
        "abstract": ["phrase x"],
    },
    "300": {
        "doc_id": 300,
        "title": "Doc 300",
        "abstract": ["phrase y"],
    },
}

QRELS_PAIRS = [("1", "100"), ("2", "200"), ("2", "201"), ("3", "300")]


# ---------------------------------------------------------------------------
# Critère 1, 2 — étiquette et phrases-preuve par paire
# ---------------------------------------------------------------------------


def test_label_for_pair_is_support_when_any_evidence_group_says_support():
    assert origin_labels.label_for_pair(CLAIMS["1"], "100") == "SUPPORT"


def test_label_for_pair_is_contradict_when_evidence_says_contradict():
    assert origin_labels.label_for_pair(CLAIMS["2"], "200") == "CONTRADICT"


def test_label_for_pair_is_sans_preuve_when_cited_without_evidence():
    assert origin_labels.label_for_pair(CLAIMS["2"], "201") == "SANS_PREUVE"
    assert origin_labels.label_for_pair(CLAIMS["3"], "300") == "SANS_PREUVE"


def test_evidence_sentences_concatenates_all_groups_in_document_order():
    sentences = origin_labels.evidence_sentences_for_pair(CLAIMS["1"], "100", CORPUS)
    assert sentences == ["phrase 0", "phrase 2", "phrase 3"]


def test_evidence_sentences_is_empty_for_sans_preuve():
    assert origin_labels.evidence_sentences_for_pair(CLAIMS["2"], "201", CORPUS) == []


# ---------------------------------------------------------------------------
# Critère 3 — catégorie par claim
# ---------------------------------------------------------------------------


def test_categorie_for_claim_is_confirme_when_a_support_document_exists():
    records = [{"label": "SUPPORT"}, {"label": "SANS_PREUVE"}]
    assert origin_labels.categorie_for_claim(records) == "confirme"


def test_categorie_for_claim_is_contredit_when_a_contradict_document_exists_and_no_support():
    records = [{"label": "CONTRADICT"}, {"label": "SANS_PREUVE"}]
    assert origin_labels.categorie_for_claim(records) == "contredit"


def test_categorie_for_claim_is_sans_preuve_otherwise():
    records = [{"label": "SANS_PREUVE"}]
    assert origin_labels.categorie_for_claim(records) == "sans_preuve"


# ---------------------------------------------------------------------------
# Critère 4 — vérification contre les qrels
# ---------------------------------------------------------------------------


def test_verify_pairs_match_qrels_passes_silently_when_identical():
    claim_pairs = origin_labels.claim_doc_pairs_from_claims(CLAIMS)
    origin_labels.verify_pairs_match_qrels(claim_pairs, QRELS_PAIRS)  # ne lève pas


def test_verify_pairs_match_qrels_raises_when_a_pair_is_missing_from_qrels():
    claim_pairs = [("1", "100"), ("9", "999")]
    with pytest.raises(ValueError, match="999"):
        origin_labels.verify_pairs_match_qrels(claim_pairs, [("1", "100")])


def test_verify_pairs_match_qrels_raises_when_a_pair_is_missing_from_claims():
    claim_pairs = [("1", "100")]
    with pytest.raises(ValueError, match="200"):
        origin_labels.verify_pairs_match_qrels(
            claim_pairs, [("1", "100"), ("2", "200")]
        )


# ---------------------------------------------------------------------------
# Document complet
# ---------------------------------------------------------------------------


def test_build_origin_labels_document_counts_and_shape():
    document = origin_labels.build_origin_labels_document(CLAIMS, CORPUS, QRELS_PAIRS)

    assert len(document["pairs"]) == 4
    labels = {(p["query_id"], p["doc_id"]): p["label"] for p in document["pairs"]}
    assert labels == {
        ("1", "100"): "SUPPORT",
        ("2", "200"): "CONTRADICT",
        ("2", "201"): "SANS_PREUVE",
        ("3", "300"): "SANS_PREUVE",
    }

    categories = {c["query_id"]: c["categorie"] for c in document["claims"]}
    assert categories == {"1": "confirme", "2": "contredit", "3": "sans_preuve"}


def test_build_origin_labels_document_raises_before_writing_on_mismatch():
    bad_claims = {"1": {"id": 1, "claim": "x", "evidence": {}, "cited_doc_ids": [999]}}
    with pytest.raises(ValueError):
        origin_labels.build_origin_labels_document(bad_claims, CORPUS, QRELS_PAIRS)


# ---------------------------------------------------------------------------
# Critères 10 à 12 — ce que le dashboard consulte, sans navigateur
# ---------------------------------------------------------------------------


def test_label_in_clear_maps_the_three_labels():
    assert origin_labels.label_in_clear("SUPPORT") == "confirme"
    assert origin_labels.label_in_clear("CONTRADICT") == "contredit"
    assert origin_labels.label_in_clear("SANS_PREUVE") == "sans preuve"


def test_build_label_lookup_derives_the_three_structures():
    document = origin_labels.build_origin_labels_document(CLAIMS, CORPUS, QRELS_PAIRS)

    label_by_pair, evidence_by_pair, categorie_by_qid = (
        origin_labels.build_label_lookup(document)
    )

    assert label_by_pair[("1", "100")] == "SUPPORT"
    assert evidence_by_pair[("1", "100")] == ["phrase 0", "phrase 2", "phrase 3"]
    assert categorie_by_qid == {"1": "confirme", "2": "contredit", "3": "sans_preuve"}


def test_filter_rows_by_categorie_keeps_only_selected_categories():
    rows = [{"query_id": "1"}, {"query_id": "2"}, {"query_id": "3"}]
    categorie_by_qid = {"1": "confirme", "2": "contredit", "3": "sans_preuve"}

    filtered = origin_labels.filter_rows_by_categorie(
        rows, {"confirme", "sans_preuve"}, categorie_by_qid
    )

    assert [r["query_id"] for r in filtered] == ["1", "3"]


def test_filter_rows_by_categorie_returns_empty_when_no_category_selected():
    rows = [{"query_id": "1"}]
    assert origin_labels.filter_rows_by_categorie(rows, set(), {"1": "confirme"}) == []
