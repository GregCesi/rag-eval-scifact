"""Tests du cache disque des calculs lourds (EXE-88, critères 7, 8, 9).

N'écrivent jamais dans le cache réel (`~/.cache/rag-eval-scifact`) : chaque
test pointe `cache_dir` vers `tmp_path`. Ne chargent aucun modèle
d'embedding : `compute_fn` est une fonction fabriquée qui compte ses appels.
"""

from __future__ import annotations

import numpy as np

from rag_eval_scifact.cache import get_document_embeddings, get_ranking

DATASET_HASH = "sha256:abc123"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW = 256


def _counting_fn(return_value):
    calls = []

    def _fn():
        calls.append(1)
        return return_value

    _fn.calls = calls
    return _fn


# ---------------------------------------------------------------------------
# Critère 7 — embeddings de documents : cache absent puis présent
# ---------------------------------------------------------------------------


def test_document_embeddings_cache_miss_calls_compute_and_writes_cache(tmp_path):
    doc_ids = ["d1", "d2"]
    fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    compute = _counting_fn(fake_embeddings)

    embeddings, hit = get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, compute
    )

    assert hit is False
    assert len(compute.calls) == 1
    np.testing.assert_array_equal(embeddings, fake_embeddings)


def test_document_embeddings_cache_hit_does_not_call_compute(tmp_path):
    doc_ids = ["d1", "d2"]
    fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    compute = _counting_fn(fake_embeddings)
    get_document_embeddings(tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, compute)

    second_compute = _counting_fn(fake_embeddings)
    embeddings, hit = get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, second_compute
    )

    assert hit is True
    assert len(second_compute.calls) == 0
    np.testing.assert_array_equal(embeddings, fake_embeddings)


def test_document_embeddings_cache_prints_hit_message(tmp_path, capsys):
    doc_ids = ["d1"]
    fake_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, _counting_fn(fake_embeddings)
    )
    capsys.readouterr()

    get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, _counting_fn(fake_embeddings)
    )
    out = capsys.readouterr().out
    assert "cache" in out.lower()
    assert "aucun document ré-embeddé" in out


# ---------------------------------------------------------------------------
# Critère 9 — changer le modèle ou la fenêtre n'utilise jamais le cache de l'autre
# ---------------------------------------------------------------------------


def test_document_embeddings_cache_is_not_reused_across_models(tmp_path):
    doc_ids = ["d1"]
    fake_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, _counting_fn(fake_embeddings)
    )

    other_model_compute = _counting_fn(fake_embeddings)
    _, hit = get_document_embeddings(
        tmp_path,
        DATASET_HASH,
        "Qwen/Qwen3-Embedding-0.6B",
        WINDOW,
        doc_ids,
        other_model_compute,
    )

    assert hit is False
    assert len(other_model_compute.calls) == 1


def test_document_embeddings_cache_is_not_reused_across_windows(tmp_path):
    doc_ids = ["d1"]
    fake_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, WINDOW, doc_ids, _counting_fn(fake_embeddings)
    )

    other_window_compute = _counting_fn(fake_embeddings)
    _, hit = get_document_embeddings(
        tmp_path, DATASET_HASH, MODEL, 2048, doc_ids, other_window_compute
    )

    assert hit is False
    assert len(other_window_compute.calls) == 1


# ---------------------------------------------------------------------------
# Critère 8 — classement de premier étage : cache absent puis présent
# ---------------------------------------------------------------------------

RANKING = [
    {
        "query_id": "q1",
        "query_text": "une question",
        "retrieved": [{"doc_id": "d1", "rank": 1, "score": 0.9}],
    }
]


def test_ranking_cache_miss_calls_compute_and_writes_cache(tmp_path):
    compute = _counting_fn(RANKING)

    ranking, hit = get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", compute
    )

    assert hit is False
    assert len(compute.calls) == 1
    assert ranking == RANKING


def test_ranking_cache_hit_does_not_call_compute(tmp_path):
    get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", _counting_fn(RANKING)
    )

    second_compute = _counting_fn(RANKING)
    ranking, hit = get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", second_compute
    )

    assert hit is True
    assert len(second_compute.calls) == 0
    assert ranking == RANKING


def test_ranking_cache_prints_hit_message(tmp_path, capsys):
    get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", _counting_fn(RANKING)
    )
    capsys.readouterr()

    get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", _counting_fn(RANKING)
    )
    out = capsys.readouterr().out
    assert "aucune similarité recalculée" in out


def test_ranking_cache_is_not_reused_across_models(tmp_path):
    get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", _counting_fn(RANKING)
    )

    other_compute = _counting_fn(RANKING)
    _, hit = get_ranking(
        tmp_path,
        DATASET_HASH,
        "Qwen/Qwen3-Embedding-0.6B",
        WINDOW,
        100,
        "test",
        other_compute,
    )

    assert hit is False
    assert len(other_compute.calls) == 1


def test_ranking_cache_is_not_reused_across_windows(tmp_path):
    get_ranking(
        tmp_path, DATASET_HASH, MODEL, WINDOW, 100, "test", _counting_fn(RANKING)
    )

    other_compute = _counting_fn(RANKING)
    _, hit = get_ranking(
        tmp_path, DATASET_HASH, MODEL, 2048, 100, "test", other_compute
    )

    assert hit is False
    assert len(other_compute.calls) == 1
