"""Tests de `retrieve_campaign` (EXE-88, critères 7, 8, 9).

Utilise le vrai corpus/les vraies requêtes SciFact (lecture de fichiers, rapide)
mais un embedder fabriqué : aucun modèle d'embedding n'est chargé. Chaque test
pointe `cache_dir` vers `tmp_path`, jamais le cache réel.
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


def test_first_run_calls_embedder_for_docs_and_queries(tmp_path):
    calls: list[int] = []
    results, qrels, dataset_hash = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(calls),
    )

    assert len(calls) == 2  # un appel docs, un appel requêtes
    assert len(results) == len(qrels) == 300
    assert dataset_hash.startswith("sha256:")
    assert len(results[0].retrieved) == TOP_K


def test_second_identical_run_does_not_call_embedder(tmp_path):
    first_calls: list[int] = []
    results_1, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(first_calls),
    )

    second_calls: list[int] = []
    results_2, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(second_calls),
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
    )
    capsys.readouterr()

    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder([]),
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
    )

    other_window_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=2048,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(other_window_calls),
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
    )

    other_model_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name="other-fake-model",
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(other_model_calls),
    )

    assert len(other_model_calls) == 2
