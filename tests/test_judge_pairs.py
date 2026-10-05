"""Tests de la sélection des paires à juger (EXE-119, refondus par EXE-125,
critères 1 à 3).

Run de référence, étiquettes d'origine et corpus entièrement fabriqués dans la
majorité des tests : aucun fichier réel n'est lu, aucun modèle n'est chargé.
Une exception assumée (`test_pairs_file_has_the_exact_counts_of_the_ticket`,
critère 2) relit les artefacts réels déjà commités (run de référence,
étiquettes d'origine, corpus) : aucun modèle n'y est appelé non plus, seuls des
fichiers sont lus.
"""

from __future__ import annotations

import json

from rag_eval_scifact.ingest import CORPUS_PATH, load_corpus
from rag_eval_scifact.judge_pairs import (
    build_pairs,
    claims_with_evidence,
    load_pairs,
    write_pairs,
)
from rag_eval_scifact.run_judge_pairs import ORIGIN_LABELS_PATH, _load_reference_run

CORPUS = {
    doc_id: {"title": f"Titre {doc_id}", "text": f"Texte {doc_id}."}
    for doc_id in ["d0", "d1", "d2", "d3", "d4", "dX", "a1", "a2", "a3", "a4", "a5"]
}


def _query(query_id, query_text, retrieved):
    return {
        "query_id": query_id,
        "query_text": query_text,
        "retrieved_top100": retrieved,
    }


def _reference_run():
    return {
        "queries": [
            # claim 1 : « devant » commun (d0) et « devant » intrus (dX) avant
            # le document SUPPORT (d1, rang 3).
            _query(
                "1",
                "claim 1",
                [
                    {"doc_id": "d0", "rank": 1, "score": 0.9},
                    {"doc_id": "dX", "rank": 2, "score": 0.8},
                    {"doc_id": "d1", "rank": 3, "score": 0.7},
                ],
            ),
            # claim 2 : document SUPPORT au rang 1, aucun devant.
            _query("2", "claim 2", [{"doc_id": "d2", "rank": 1, "score": 0.9}]),
            # claim 3 : sans preuve, ne contribue aucun devant même si retrouvé.
            _query("3", "claim 3", [{"doc_id": "d3", "rank": 1, "score": 0.9}]),
            # claim 4 : avec preuve, mais aucun document SUPPORT/CONTRADICT
            # retrouvé dans le top 100 -> repli sur les 5 premiers retournés.
            _query(
                "4",
                "claim 4",
                [
                    {"doc_id": "a1", "rank": 1, "score": 0.9},
                    {"doc_id": "a2", "rank": 2, "score": 0.8},
                    {"doc_id": "a3", "rank": 3, "score": 0.7},
                    {"doc_id": "a4", "rank": 4, "score": 0.6},
                    {"doc_id": "a5", "rank": 5, "score": 0.5},
                ],
            ),
        ]
    }


def _origin_labels():
    return {
        "claims": [
            {"query_id": "1", "categorie": "confirme"},
            {"query_id": "2", "categorie": "confirme"},
            {"query_id": "3", "categorie": "sans_preuve"},
            {"query_id": "4", "categorie": "contredit"},
        ],
        "pairs": [
            {"query_id": "1", "doc_id": "d0", "label": "SANS_PREUVE"},
            {"query_id": "1", "doc_id": "d1", "label": "SUPPORT"},
            {"query_id": "2", "doc_id": "d2", "label": "SUPPORT"},
            {"query_id": "3", "doc_id": "d3", "label": "SANS_PREUVE"},
            {"query_id": "4", "doc_id": "d4", "label": "CONTRADICT"},
        ],
    }


# ---------------------------------------------------------------------------
# Critère 1b — claims « avec preuve »
# ---------------------------------------------------------------------------


def test_claims_with_evidence_are_confirme_or_contredit_only():
    assert claims_with_evidence(_origin_labels()) == ["1", "2", "4"]


# ---------------------------------------------------------------------------
# Critère 1a et 3 — famille « attendu » : toutes les paires, toute catégorie
# ---------------------------------------------------------------------------


def test_family_a_includes_every_origin_label_pair_regardless_of_claim_category():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    attendu_pairs = {p["doc_id"] for p in pairs if "attendu" in p["famille"]}

    assert attendu_pairs == {"d0", "d1", "d2", "d3", "d4"}

    d3 = next(p for p in pairs if p["doc_id"] == "d3")
    assert d3["document_attendu"] is True
    assert d3["famille"] == ["attendu"]
    assert d3["etiquette_origine"] == "SANS_PREUVE"


# ---------------------------------------------------------------------------
# Critère 1b — famille « devant » : avant le premier document avec preuve
# ---------------------------------------------------------------------------


def test_family_b_takes_docs_before_first_evidence_doc():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    claim_1_pairs = [p for p in pairs if p["claim_id"] == "1"]

    devant_ids = {p["doc_id"] for p in claim_1_pairs if "devant" in p["famille"]}
    assert devant_ids == {"d0", "dX"}
    # d1 est le document avec preuve lui-même : jamais « devant ».
    assert (
        "devant" not in next(p for p in claim_1_pairs if p["doc_id"] == "d1")["famille"]
    )


def test_family_b_is_empty_when_evidence_doc_is_at_rank_one():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    claim_2_pairs = [p for p in pairs if p["claim_id"] == "2"]

    assert all("devant" not in p["famille"] for p in claim_2_pairs)


def test_claim_without_evidence_contributes_no_devant_pair():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    claim_3_pairs = [p for p in pairs if p["claim_id"] == "3"]

    assert all("devant" not in p["famille"] for p in claim_3_pairs)


def test_family_b_falls_back_to_first_five_retrieved_when_no_evidence_doc_in_top100():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    claim_4_pairs = [p for p in pairs if p["claim_id"] == "4"]

    devant_ids = [p["doc_id"] for p in claim_4_pairs if "devant" in p["famille"]]
    assert devant_ids == ["a1", "a2", "a3", "a4", "a5"]

    d4 = next(p for p in claim_4_pairs if p["doc_id"] == "d4")
    assert d4["famille"] == ["attendu"]
    assert d4["rank"] is None  # jamais retourné par le run de référence


# ---------------------------------------------------------------------------
# Critère 3 — une paire peut appartenir aux deux familles, une seule fois
# ---------------------------------------------------------------------------


def test_pair_common_to_both_families_appears_once_with_both():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    d0_pairs = [p for p in pairs if p["doc_id"] == "d0" and p["claim_id"] == "1"]

    assert len(d0_pairs) == 1
    d0 = d0_pairs[0]
    assert d0["famille"] == ["attendu", "devant"]
    assert d0["document_attendu"] is True
    assert d0["etiquette_origine"] == "SANS_PREUVE"
    assert d0["rank"] == 1


def test_devant_only_pair_has_no_origin_label_and_is_not_attendu():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    dx = next(p for p in pairs if p["doc_id"] == "dX")

    assert dx["famille"] == ["devant"]
    assert dx["document_attendu"] is False
    assert dx["etiquette_origine"] is None
    assert dx["rank"] == 2


def test_pair_carries_claim_and_document_identity_and_text():
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)
    d1 = next(p for p in pairs if p["doc_id"] == "d1")

    assert d1["claim_id"] == "1"
    assert d1["claim_text"] == "claim 1"
    assert d1["doc_title"] == "Titre d1"
    assert d1["doc_text"] == "Texte d1."
    assert d1["pair_id"] == "1:d1"


# ---------------------------------------------------------------------------
# Critère 2 — effectifs exacts sur les artefacts réels déjà commités
# ---------------------------------------------------------------------------


def test_pairs_file_has_the_exact_counts_of_the_ticket():
    reference_run = _load_reference_run()
    origin_labels = json.loads(ORIGIN_LABELS_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus(CORPUS_PATH)
    corpus_by_id = {doc["_id"]: doc for doc in corpus}

    pairs = build_pairs(reference_run, origin_labels, corpus_by_id)

    assert len(pairs) == 438
    assert sum(1 for p in pairs if "attendu" in p["famille"]) == 339
    assert sum(1 for p in pairs if "devant" in p["famille"]) == 105
    assert sum(1 for p in pairs if set(p["famille"]) == {"attendu", "devant"}) == 6
    assert len({p["claim_id"] for p in pairs if "devant" in p["famille"]}) == 46


# ---------------------------------------------------------------------------
# Écriture / lecture du fichier de paires
# ---------------------------------------------------------------------------


def test_write_then_load_pairs_round_trips(tmp_path, monkeypatch):
    import rag_eval_scifact.judge_pairs as judge_pairs_module

    monkeypatch.setattr(judge_pairs_module, "RESULTS_DIR", tmp_path)
    pairs = build_pairs(_reference_run(), _origin_labels(), CORPUS)

    path = write_pairs(pairs, "dev")

    assert path == tmp_path / "dev" / "paires.json"
    assert load_pairs(path) == json.loads(path.read_text(encoding="utf-8"))
    assert load_pairs(path) == pairs
