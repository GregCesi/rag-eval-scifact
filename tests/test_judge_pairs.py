"""Tests de la sélection des paires à juger (EXE-119, critères 1 à 3).

Runs et corpus entièrement fabriqués : aucun fichier réel de v2-grid n'est lu,
aucun modèle n'est chargé.
"""

from __future__ import annotations

import json

from rag_eval_scifact.judge_pairs import (
    build_pairs,
    load_pairs,
    select_claims_to_judge,
    write_pairs,
)

CORPUS = {
    "d1": {"title": "Titre D1", "text": "Texte D1."},
    "dX": {"title": "Titre DX", "text": "Texte DX."},
    "dY": {"title": "Titre DY", "text": "Texte DY."},
    "d9": {"title": "Titre D9", "text": "Texte D9."},
    **{f"a{i}": {"title": f"Titre A{i}", "text": f"Texte A{i}."} for i in range(1, 7)},
}


def _query(query_id, found_at_10):
    return {
        "query_id": query_id,
        "query_text": f"claim {query_id}",
        "per_query_metrics": {"found@10": found_at_10},
        "expected_docs": [],
        "retrieved_top100": [],
    }


def _grid_runs():
    # claim "1" : mixte (trouvé par A, pas par B, trouvé par C) -> retenu
    # claim "2" : trouvé par les trois -> pas retenu
    # claim "3" : jamais trouvé -> retenu
    run_a = {"queries": [_query("1", True), _query("2", True), _query("3", False)]}
    run_b = {"queries": [_query("1", False), _query("2", True), _query("3", False)]}
    run_c = {"queries": [_query("1", True), _query("2", True), _query("3", False)]}
    return [run_a, run_b, run_c]


# ---------------------------------------------------------------------------
# Critère 1 — sélection des claims
# ---------------------------------------------------------------------------


def test_claim_is_retained_when_found_at_10_is_not_unanimous_across_grid_runs():
    retained = select_claims_to_judge(_grid_runs())

    assert retained == {"1", "3"}


# ---------------------------------------------------------------------------
# Critères 2 et 3 — documents lus dans le run de référence
# ---------------------------------------------------------------------------


def _reference_run():
    claim_1 = {
        "query_id": "1",
        "query_text": "0-dimensional biomaterials show inductive properties.",
        "expected_docs": [{"doc_id": "d1", "token_count": 10}],
        "retrieved_top100": [
            {"doc_id": "dX", "rank": 1, "score": 0.9},
            {"doc_id": "d1", "rank": 2, "score": 0.8},
            {"doc_id": "dY", "rank": 3, "score": 0.7},
        ],
        "per_query_metrics": {"found@10": True, "best_rank": 2},
    }
    claim_3 = {
        "query_id": "3",
        "query_text": "claim jamais trouvé",
        "expected_docs": [{"doc_id": "d9", "token_count": 5}],
        "retrieved_top100": [
            {"doc_id": f"a{i}", "rank": i, "score": 1.0 / i} for i in range(1, 7)
        ],
        "per_query_metrics": {"found@10": False, "best_rank": None},
    }
    return {"queries": [claim_1, claim_3]}


def test_pairs_include_every_expected_doc_and_up_to_5_non_expected_before_best_rank():
    pairs = build_pairs(_grid_runs(), _reference_run(), CORPUS)
    by_claim = {}
    for p in pairs:
        by_claim.setdefault(p["claim_id"], []).append(p)

    claim_1_pairs = by_claim["1"]
    assert {p["doc_id"] for p in claim_1_pairs} == {"d1", "dX"}
    dy_ids = [p["doc_id"] for p in claim_1_pairs if p["doc_id"] == "dY"]
    assert dy_ids == []  # dY est classé après le document attendu : exclu

    d1 = next(p for p in claim_1_pairs if p["doc_id"] == "d1")
    assert d1["document_attendu"] is True
    assert d1["rank"] == 2
    dx = next(p for p in claim_1_pairs if p["doc_id"] == "dX")
    assert dx["document_attendu"] is False
    assert dx["rank"] == 1


def test_pairs_take_first_5_retrieved_when_no_expected_doc_in_top_100():
    pairs = build_pairs(_grid_runs(), _reference_run(), CORPUS)
    claim_3_pairs = [p for p in pairs if p["claim_id"] == "3"]

    non_expected = [p for p in claim_3_pairs if not p["document_attendu"]]
    assert [p["doc_id"] for p in non_expected] == ["a1", "a2", "a3", "a4", "a5"]

    expected = [p for p in claim_3_pairs if p["document_attendu"]]
    assert len(expected) == 1
    assert expected[0]["doc_id"] == "d9"
    assert expected[0]["rank"] is None  # jamais retourné par le run de référence


def test_pair_carries_claim_and_document_identity_and_text():
    pairs = build_pairs(_grid_runs(), _reference_run(), CORPUS)
    d1 = next(p for p in pairs if p["doc_id"] == "d1")

    assert d1["claim_id"] == "1"
    assert d1["claim_text"] == "0-dimensional biomaterials show inductive properties."
    assert d1["doc_title"] == "Titre D1"
    assert d1["doc_text"] == "Texte D1."
    assert d1["pair_id"] == "1:d1"


def test_claim_fully_found_everywhere_contributes_no_pair():
    pairs = build_pairs(_grid_runs(), _reference_run(), CORPUS)

    assert all(p["claim_id"] != "2" for p in pairs)


# ---------------------------------------------------------------------------
# Écriture / lecture du fichier de paires
# ---------------------------------------------------------------------------


def test_write_then_load_pairs_round_trips(tmp_path, monkeypatch):
    import rag_eval_scifact.judge_pairs as judge_pairs_module

    monkeypatch.setattr(judge_pairs_module, "RESULTS_DIR", tmp_path)
    pairs = build_pairs(_grid_runs(), _reference_run(), CORPUS)

    path = write_pairs(pairs, "dev")

    assert path == tmp_path / "dev" / "paires.json"
    assert load_pairs(path) == json.loads(path.read_text(encoding="utf-8"))
    assert load_pairs(path) == pairs
