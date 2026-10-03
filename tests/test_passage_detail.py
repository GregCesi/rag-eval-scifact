"""Tests de `rag_eval_scifact.passage_detail` (EXE-112, critères 1, 2, 3, 7, 9).

Logique pure sur des passages et des scores fabriqués : aucun modèle, aucun
corpus réel, aucune similarité recalculée par un retriever réel. Le
chargement de modèle et l'indexation BM25 restent dans
`generate_passage_detail.py` (point d'entrée CLI, hors de ce fichier).
"""

from __future__ import annotations

from rag_eval_scifact.chunking import Passage
from rag_eval_scifact.passage_detail import (
    build_query_passage_detail,
    build_run_passage_detail,
    eligible_for_passage_detail,
    find_query_passage_detail,
    group_passages_by_doc,
    load_passage_detail,
    needed_doc_ids,
    passage_detail_path,
    verify_max_grouping_scores,
    write_passage_detail,
)


def _passage(doc_id: str, index: int, score_hint: int = 0) -> Passage:
    return Passage(
        passage_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        text=f"texte {doc_id} {index}",
        token_count=5,
        char_start=index * 10,
        char_end=index * 10 + 10,
    )


# ---------------------------------------------------------------------------
# Critère 2 / « ce qui ne doit pas arriver » — éligibilité (passages, pas
# hybride, pas de reranker)
# ---------------------------------------------------------------------------


def _run_with(unit: str, name: str, rerank: dict | None) -> dict:
    return {
        "config": {
            "retriever": {"unit": unit, "name": name, "grouping": "max"},
            "rerank": rerank,
        },
        "queries": [],
    }


def test_document_unit_is_not_eligible():
    assert not eligible_for_passage_detail(_run_with("document", "dense", None))


def test_hybrid_retriever_is_not_eligible():
    assert not eligible_for_passage_detail(_run_with("passages", "hybrid", None))


def test_passages_with_cross_encoder_rerank_is_not_eligible():
    rerank = {"name": "cross-encoder", "model": "x", "top_n": 100}
    assert not eligible_for_passage_detail(_run_with("passages", "dense", rerank))


def test_passages_dense_without_rerank_is_eligible():
    assert eligible_for_passage_detail(_run_with("passages", "dense", None))


def test_passages_bm25_with_rerank_name_none_is_eligible():
    rerank = {"name": "none", "model": "x", "top_n": 100}
    assert eligible_for_passage_detail(_run_with("passages", "bm25", rerank))


# ---------------------------------------------------------------------------
# group_passages_by_doc
# ---------------------------------------------------------------------------


def test_group_passages_by_doc_preserves_order_within_a_document():
    passages = [_passage("d1", 0), _passage("d2", 0), _passage("d1", 1)]
    by_doc = group_passages_by_doc(passages)

    assert [p.passage_id for p in by_doc["d1"]] == ["d1::0", "d1::1"]
    assert [p.passage_id for p in by_doc["d2"]] == ["d2::0"]


# ---------------------------------------------------------------------------
# Critère 1 / 7 — docs nécessaires : attendus puis top 10, sans doublon,
# y compris un doc attendu absent du top 100
# ---------------------------------------------------------------------------


def _query_with_top_n(n: int) -> dict:
    return {
        "query_id": "q1",
        "expected_docs": [{"doc_id": "d_expected", "token_count": 10}],
        "retrieved_top100": [
            {"doc_id": f"d{i}", "rank": i + 1, "score": 1.0 - i / 100} for i in range(n)
        ],
    }


def test_needed_doc_ids_keeps_expected_doc_even_when_absent_from_top_100():
    query = _query_with_top_n(3)
    assert needed_doc_ids(query)[0] == "d_expected"


def test_needed_doc_ids_is_limited_to_the_top_10_of_retrieved_top100():
    query = _query_with_top_n(15)
    ids = needed_doc_ids(query)

    assert "d10" not in ids  # rang 11, hors top 10
    assert "d9" in ids  # rang 10


def test_needed_doc_ids_deduplicates_an_expected_doc_already_in_the_top_10():
    query = {
        "query_id": "q1",
        "expected_docs": [{"doc_id": "d0", "token_count": 10}],
        "retrieved_top100": [{"doc_id": "d0", "rank": 1, "score": 0.9}],
    }
    assert needed_doc_ids(query) == ["d0"]


# ---------------------------------------------------------------------------
# build_query_passage_detail / build_run_passage_detail
# ---------------------------------------------------------------------------


def test_build_query_passage_detail_lists_each_passage_with_its_score():
    query = _query_with_top_n(1)
    passages_by_doc = {
        "d_expected": [_passage("d_expected", 0), _passage("d_expected", 1)],
        "d0": [_passage("d0", 0)],
    }

    def score_doc_passages(doc_passages):
        return [0.5 for _ in doc_passages]

    detail = build_query_passage_detail(query, passages_by_doc, score_doc_passages)

    assert detail["query_id"] == "q1"
    assert [p["score"] for p in detail["docs"]["d_expected"]] == [0.5, 0.5]
    assert detail["docs"]["d_expected"][0]["char_start"] == 0
    assert detail["docs"]["d_expected"][1]["char_end"] == 20


def test_build_query_passage_detail_tolerates_a_document_with_no_passages():
    query = _query_with_top_n(1)
    detail = build_query_passage_detail(query, {}, lambda doc_passages: [])

    assert detail["docs"] == {"d_expected": [], "d0": []}


def test_build_run_passage_detail_scores_each_query_independently():
    run = {
        "campagne": "dev",
        "run_name": "fake-run",
        "queries": [_query_with_top_n(1), {**_query_with_top_n(1), "query_id": "q2"}],
    }
    passages = [_passage("d_expected", 0), _passage("d0", 0)]

    def score_doc_passages_for_query(query, doc_passages):
        return [1.0 if query["query_id"] == "q1" else 2.0 for _ in doc_passages]

    detail = build_run_passage_detail(run, passages, score_doc_passages_for_query)

    assert detail["campagne"] == "dev"
    assert detail["run_name"] == "fake-run"
    by_qid = {q["query_id"]: q for q in detail["queries"]}
    assert by_qid["q1"]["docs"]["d_expected"][0]["score"] == 1.0
    assert by_qid["q2"]["docs"]["d_expected"][0]["score"] == 2.0


# ---------------------------------------------------------------------------
# Critère 1 — chemin, écriture et lecture du fichier dérivé
# ---------------------------------------------------------------------------


def test_passage_detail_path_is_nested_under_a_passages_subfolder(tmp_path):
    """Hors du dossier plat de la campagne : `list_campaign_run_files`
    (dashboard) et les tests de EXE-109 listent `<campagne>/*.json.gz` sans
    récursion — un fichier dérivé à côté du run s'y ferait passer pour un
    second run (EXE-112)."""
    run_path = tmp_path / "dense-minilm-passages-sans-reranker-2026-10-03.json.gz"
    assert passage_detail_path(run_path) == (
        tmp_path
        / "passages"
        / "dense-minilm-passages-sans-reranker-2026-10-03.passages.json.gz"
    )


def test_write_then_load_passage_detail_roundtrips(tmp_path):
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"
    detail = {"campagne": "dev", "run_name": "fake-run", "queries": []}

    written_path = write_passage_detail(run_path, detail)

    assert written_path == passage_detail_path(run_path)
    assert load_passage_detail(run_path) == detail


def test_derived_file_is_invisible_to_a_flat_non_recursive_glob_of_the_campaign_dir(
    tmp_path,
):
    """Le dérivé n'apparaît jamais dans `<campagne>.glob("*.json.gz")`, utilisé
    sans récursion par `list_campaign_run_files` et par EXE-109 : seul un run
    réel doit s'y trouver."""
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"
    write_passage_detail(
        run_path, {"campagne": "dev", "run_name": "fake-run", "queries": []}
    )

    assert list(tmp_path.glob("*.json.gz")) == []


def test_load_passage_detail_returns_none_when_the_file_is_absent(tmp_path):
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"
    assert load_passage_detail(run_path) is None


# ---------------------------------------------------------------------------
# Critère 3 / H4 — vérification du regroupement max contre le score du run
# ---------------------------------------------------------------------------


def _run_and_detail_for_verification(
    grouping: str, doc_score: float, best_passage: float
):
    run = {
        "config": {"retriever": {"grouping": grouping}},
        "queries": [
            {
                "query_id": "q1",
                "retrieved_top100": [{"doc_id": "d1", "rank": 1, "score": doc_score}],
            }
        ],
    }
    detail = {
        "queries": [
            {
                "query_id": "q1",
                "docs": {
                    "d1": [{"score": best_passage}, {"score": best_passage - 0.1}]
                },
            }
        ]
    }
    return run, detail


def test_verify_max_grouping_reports_no_mismatch_when_scores_agree():
    run, detail = _run_and_detail_for_verification("max", 0.71046, 0.71046)
    assert verify_max_grouping_scores(run, detail) == []


def test_verify_max_grouping_reports_a_mismatch_beyond_the_4th_decimal():
    run, detail = _run_and_detail_for_verification("max", 0.7105, 0.7200)
    mismatches = verify_max_grouping_scores(run, detail)

    assert len(mismatches) == 1
    assert mismatches[0][0] == "q1"
    assert mismatches[0][1] == "d1"


def test_verify_max_grouping_is_skipped_for_sum_grouping():
    run, detail = _run_and_detail_for_verification("sum", 0.71, 0.99)
    assert verify_max_grouping_scores(run, detail) == []


def test_find_query_passage_detail_returns_none_when_detail_is_none():
    assert find_query_passage_detail(None, "q1") is None


def test_find_query_passage_detail_returns_none_when_qid_is_absent():
    detail = {"queries": [{"query_id": "q1", "docs": {}}]}
    assert find_query_passage_detail(detail, "q2") is None


def test_find_query_passage_detail_returns_the_matching_query():
    detail = {"queries": [{"query_id": "q1", "docs": {"d1": []}}]}
    assert find_query_passage_detail(detail, "q1") == {
        "query_id": "q1",
        "docs": {"d1": []},
    }


def test_verify_max_grouping_ignores_a_doc_absent_from_retrieved_top100():
    run = {
        "config": {"retriever": {"grouping": "max"}},
        "queries": [{"query_id": "q1", "retrieved_top100": []}],
    }
    detail = {"queries": [{"query_id": "q1", "docs": {"d_expected": [{"score": 0.5}]}}]}
    assert verify_max_grouping_scores(run, detail) == []
