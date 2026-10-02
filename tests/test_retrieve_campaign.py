"""Tests de `retrieve_campaign` (EXE-88, critères 7, 8, 9 ; EXE-90, critères 3, 4).

Utilise le vrai corpus/les vraies requêtes SciFact (lecture de fichiers, rapide)
mais un embedder et un compteur de tokens fabriqués : aucun modèle d'embedding
n'est chargé. Chaque test pointe `cache_dir` vers `tmp_path`, jamais le cache réel.
"""

from __future__ import annotations

import numpy as np

from rag_eval_scifact.retrieve import retrieve_campaign

TOP_K = 5
MODEL = "fake-model"
WINDOW = 256
SPLIT = "test"


def _fake_embedder(calls: list):
    def _embed(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        # vecteurs déterministes, jamais nuls (évite la division par zéro L2)
        return np.array(
            [[(hash(t) % 97) + 1, (len(t) % 53) + 1] for t in texts], dtype=np.float32
        )

    return _embed


def _fake_token_counter(token_count: int = 10):
    """Compteur de tokens fabriqué : un entier fixe par texte, aucun tokenizer chargé."""

    def _count(texts: list[str]) -> list[int]:
        return [token_count] * len(texts)

    return _count


def test_first_run_calls_embedder_for_docs_and_queries(tmp_path):
    calls: list[int] = []
    results, qrels, dataset_hash, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(calls),
        token_counter=_fake_token_counter(),
    )

    assert len(calls) == 2  # un appel docs, un appel requêtes
    assert len(results) == len(qrels) == 300
    assert dataset_hash.startswith("sha256:")
    assert len(results[0].retrieved) == TOP_K
    assert set(stats) == {
        "truncated_pct",
        "avg_retrieval_latency_ms",
        "indexing_duration_seconds",
    }


def test_second_identical_run_does_not_call_embedder(tmp_path):
    first_calls: list[int] = []
    results_1, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(first_calls),
        token_counter=_fake_token_counter(),
    )

    second_calls: list[int] = []
    results_2, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(second_calls),
        token_counter=_fake_token_counter(),
    )

    assert second_calls == []
    assert [r.retrieved for r in results_1] == [r.retrieved for r in results_2]


def test_second_run_prints_cache_hit_messages(tmp_path, capsys):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )
    capsys.readouterr()

    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )
    out = capsys.readouterr().out
    assert "aucune similarité recalculée" in out


def test_changing_window_recomputes_everything(tmp_path):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    other_window_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=2048,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(other_window_calls),
        token_counter=_fake_token_counter(),
    )

    assert len(other_window_calls) == 2


def test_changing_model_recomputes_everything(tmp_path):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    other_model_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name="other-fake-model",
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(other_model_calls),
        token_counter=_fake_token_counter(),
    )

    assert len(other_model_calls) == 2


# ---------------------------------------------------------------------------
# EXE-90 critère 3 — part tronquée, calculée via le compteur de tokens injecté
# ---------------------------------------------------------------------------


def test_truncated_pct_is_computed_from_injected_token_counts(tmp_path):
    # Tous les comptages (10) dépassent la fenêtre (5) -> 100% tronqué.
    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=5,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(token_count=10),
    )

    assert stats["truncated_pct"] == 100.0


def test_truncated_pct_is_zero_when_no_document_exceeds_the_window(tmp_path):
    # Tous les comptages (10) sont sous la fenêtre (50) -> 0% tronqué.
    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=50,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(token_count=10),
    )

    assert stats["truncated_pct"] == 0.0


def test_truncated_pct_is_recomputed_even_when_ranking_cache_serves(tmp_path):
    """La part tronquée n'est pas un calcul lourd mis en cache : elle se
    recalcule à chaque run, y compris quand le classement vient du cache."""
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(token_count=10),
    )

    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(token_count=10),
    )

    assert stats["truncated_pct"] == 0.0  # 10 tokens < fenêtre 256


# ---------------------------------------------------------------------------
# EXE-90 critère 4 — durée d'indexation et latence moyenne de retrieval
# ---------------------------------------------------------------------------


def test_indexing_duration_is_positive_on_a_cache_miss(tmp_path):
    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    assert stats["indexing_duration_seconds"] > 0.0
    assert stats["avg_retrieval_latency_ms"] > 0.0


def test_indexing_duration_is_zero_when_ranking_cache_serves(tmp_path):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    assert stats["indexing_duration_seconds"] == 0.0
