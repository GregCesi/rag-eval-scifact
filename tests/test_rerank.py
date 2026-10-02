"""Tests du reranker cross-encoder (EXE-96 critères 2, 3, 6, 8).

Aucun modèle cross-encoder chargé : `score_fn` est toujours fabriqué.
"""

from __future__ import annotations

import pytest

import rag_eval_scifact.rerank as rerank_module
from rag_eval_scifact.rerank import rerank_campaign_results, rerank_ranking
from rag_eval_scifact.retrieve import RetrievalResult

# ---------------------------------------------------------------------------
# Critère 2 — chaque paire reclassée est (claim, titre + texte du document)
# ---------------------------------------------------------------------------


def test_pairs_combine_claim_text_and_document_text():
    captured: list[tuple[str, str]] = []

    def fake_score_fn(pairs):
        captured.extend(pairs)
        return [0.0] * len(pairs)

    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.5},
        {"doc_id": "d2", "rank": 2, "score": 0.4},
    ]
    doc_texts = {"d1": "Titre1 Texte1", "d2": "Titre2 Texte2"}

    rerank_ranking("mon claim", retrieved, doc_texts, top_n=10, score_fn=fake_score_fn)

    assert captured == [
        ("mon claim", "Titre1 Texte1"),
        ("mon claim", "Titre2 Texte2"),
    ]


def test_no_pairs_scored_when_retrieved_is_empty():
    calls = []

    def fake_score_fn(pairs):
        calls.append(pairs)
        return []

    reranked = rerank_ranking("claim", [], {}, top_n=10, score_fn=fake_score_fn)

    assert reranked == []
    assert calls == []


# ---------------------------------------------------------------------------
# Critère 3 — la liste reclassée contient exactement les mêmes documents
# ---------------------------------------------------------------------------


def test_reranked_list_contains_exactly_the_same_documents_reordered():
    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.9},
        {"doc_id": "d2", "rank": 2, "score": 0.8},
        {"doc_id": "d3", "rank": 3, "score": 0.7},
    ]
    doc_texts = {"d1": "t1", "d2": "t2", "d3": "t3"}

    def fake_score_fn(pairs):
        # Score fabriqué qui inverse l'ordre d'entrée.
        return list(range(len(pairs)))

    reranked = rerank_ranking(
        "claim", retrieved, doc_texts, top_n=3, score_fn=fake_score_fn
    )

    assert {d["doc_id"] for d in reranked} == {"d1", "d2", "d3"}
    assert [d["doc_id"] for d in reranked] == ["d3", "d2", "d1"]
    assert [d["rank"] for d in reranked] == [1, 2, 3]


def test_documents_beyond_top_n_are_appended_unchanged():
    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.9},
        {"doc_id": "d2", "rank": 2, "score": 0.8},
        {"doc_id": "d3", "rank": 3, "score": 0.1},
    ]
    doc_texts = {"d1": "t1", "d2": "t2", "d3": "t3"}

    def fake_score_fn(pairs):
        assert len(pairs) == 2  # seuls d1 et d2 entrent dans top_n=2
        return [0.0, 1.0]

    reranked = rerank_ranking(
        "claim", retrieved, doc_texts, top_n=2, score_fn=fake_score_fn
    )

    assert [d["doc_id"] for d in reranked] == ["d2", "d1", "d3"]
    assert [d["rank"] for d in reranked] == [1, 2, 3]
    assert reranked[2]["score"] == 0.1  # score original conservé pour la queue


# ---------------------------------------------------------------------------
# Critère 6 — la durée du reclassement est mesurée
# ---------------------------------------------------------------------------


def test_rerank_campaign_results_returns_reranked_results_and_duration(monkeypatch):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="claim1",
            retrieved=[
                {"doc_id": "d1", "rank": 1, "score": 0.5},
                {"doc_id": "d2", "rank": 2, "score": 0.4},
            ],
        ),
    ]
    doc_texts = {"d1": "t1", "d2": "t2"}

    def fake_score_fn(pairs):
        return [0.0, 1.0]  # inverse l'ordre

    clock = iter([10.0, 10.5])
    monkeypatch.setattr(rerank_module.time, "perf_counter", lambda: next(clock))

    reranked, duration = rerank_campaign_results(
        results, doc_texts, top_n=2, score_fn=fake_score_fn
    )

    assert [d["doc_id"] for d in reranked[0].retrieved] == ["d2", "d1"]
    assert duration == pytest.approx(0.5)


def test_rerank_campaign_results_reranks_every_query():
    results = [
        RetrievalResult(
            query_id=f"q{i}",
            query_text=f"claim{i}",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
        for i in range(3)
    ]
    calls: list[int] = []

    def fake_score_fn(pairs):
        calls.append(len(pairs))
        return [0.0] * len(pairs)

    reranked, _ = rerank_campaign_results(
        results, {"d1": "t1"}, top_n=100, score_fn=fake_score_fn
    )

    assert len(reranked) == 3
    assert calls == [1, 1, 1]
