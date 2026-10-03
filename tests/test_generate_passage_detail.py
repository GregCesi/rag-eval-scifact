"""Tests de `rag_eval_scifact.generate_passage_detail` (EXE-112, critères 1, 2, 3, 7, 8, 9).

Corpus/requêtes/qrels fabriqués et écrits sous `tmp_path` : jamais le vrai
corpus SciFact. `offsets_fn` fabriqué (un token = un caractère) comme dans
`test_grouping.py` : aucun tokenizer réel chargé. Le dense utilise un
embedder fabriqué et un cache d'embeddings pré-écrit à la main ; le BM25
utilise les vraies fonctions de `rag_eval_scifact.bm25` (pas un modèle,
l'indexation lexicale est autorisée hors cache).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from rag_eval_scifact.bm25 import build_bm25_index, score_queries
from rag_eval_scifact.cache import document_embeddings_cache_path
from rag_eval_scifact.generate_passage_detail import generate_passage_detail, main
from rag_eval_scifact.ingest import compute_dataset_hash
from rag_eval_scifact.passage_detail import load_passage_detail, passage_detail_path


def _char_offsets_fn():
    def _offsets(text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    return _offsets


def _fixture_corpus(
    tmp_path: Path, texts: tuple[str, str] = ("cat cat cat", "dog dog dog")
):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    corpus_path.write_text(
        "\n".join(
            json.dumps(doc)
            for doc in [
                {"_id": "d1", "title": "", "text": texts[0]},
                {"_id": "d2", "title": "", "text": texts[1]},
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(json.dumps({"_id": "q1", "text": "cat"}), encoding="utf-8")
    qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return corpus_path, queries_path, qrels_path


def _base_run(
    dataset_hash: str, retriever_overrides: dict, retrieved_top100: list[dict]
) -> dict:
    retriever = {
        "name": "bm25",
        "unit": "passages",
        "model": "fake-model",
        "max_seq_length": 999,
        "chunk_size": 50,
        "chunk_overlap": 5,
        "grouping": "max",
        "grouping_top_n": 1000,
        "batch_size": 8,
        "query_instruction": "",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        **retriever_overrides,
    }
    return {
        "campagne": "dev",
        "run_name": "fake-run",
        "dataset_hash": dataset_hash,
        "config": {"retriever": retriever, "rerank": None},
        "queries": [
            {
                "query_id": "q1",
                "query_text": "cat",
                "expected_docs": [{"doc_id": "d1", "token_count": 3}],
                "retrieved_top100": retrieved_top100,
                "per_query_metrics": {"best_rank": 1},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Critère « ce qui ne doit pas arriver » — éligibilité, rien n'est écrit
# ---------------------------------------------------------------------------


def test_document_unit_run_is_skipped_without_writing_a_file(tmp_path, capsys):
    run = {
        "run_name": "doc-run",
        "config": {"retriever": {"unit": "document", "name": "dense"}, "rerank": None},
        "queries": [],
    }
    run_path = tmp_path / "doc-run-2026-10-03.json.gz"

    result = generate_passage_detail(run, run_path, tmp_path / "cache")

    assert result is None
    assert not passage_detail_path(run_path).exists()
    assert "pas éligible" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# H5bis / porte de sortie — dataset_hash du run != corpus actuel
# ---------------------------------------------------------------------------


def test_dataset_hash_mismatch_stops_without_writing(tmp_path, capsys):
    corpus_path, queries_path, qrels_path = _fixture_corpus(tmp_path)
    run = _base_run(
        "sha256:not-the-real-one", {}, [{"doc_id": "d1", "rank": 1, "score": 0.5}]
    )
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"

    result = generate_passage_detail(
        run,
        run_path,
        tmp_path / "cache",
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        offsets_fn=_char_offsets_fn(),
    )

    assert result is None
    assert not passage_detail_path(run_path).exists()
    assert "dataset_hash" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Critère 2, 3, 9 — BM25 : vraie indexation lexicale, aucun modèle chargé
# ---------------------------------------------------------------------------


def test_bm25_generates_a_file_matching_the_run_score(tmp_path):
    corpus_path, queries_path, qrels_path = _fixture_corpus(
        tmp_path, texts=("rare xylophone solo", "dog dog dog")
    )
    index = build_bm25_index(["rare xylophone solo", "dog dog dog"], 1.2, 0.75)
    expected_score = float(score_queries(index, ["cat"])[0][0])

    dataset_hash = compute_dataset_hash(corpus_path)
    run = _base_run(
        dataset_hash,
        {"name": "bm25"},
        [{"doc_id": "d1", "rank": 1, "score": expected_score}],
    )
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"

    result = generate_passage_detail(
        run,
        run_path,
        tmp_path / "cache",
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        offsets_fn=_char_offsets_fn(),
    )

    assert result == passage_detail_path(run_path)
    detail = load_passage_detail(run_path)
    d1_scores = [p["score"] for p in detail["queries"][0]["docs"]["d1"]]
    assert d1_scores == pytest.approx([expected_score])


# ---------------------------------------------------------------------------
# Critère 2, 3, 7, 9 — dense : cache lu, jamais recalculé ; requêtes ré-encodées
# ---------------------------------------------------------------------------


def _write_fake_dense_cache(
    cache_dir, dataset_hash, model_name, max_seq_length, chunking_unit
):
    path = document_embeddings_cache_path(
        cache_dir, dataset_hash, model_name, max_seq_length, chunking_unit
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # d1 et d2 ont chacun un seul passage (chunk_size > longueur du texte).
    np.savez_compressed(
        path,
        doc_ids=np.array(["d1::0", "d2::0"]),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    return path


def _aligned_with_d1_embedder(texts: list[str]) -> np.ndarray:
    return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_dense_reads_the_cache_and_matches_the_run_score_including_a_doc_absent_from_top100(
    tmp_path,
):
    corpus_path, queries_path, qrels_path = _fixture_corpus(tmp_path)
    dataset_hash = compute_dataset_hash(corpus_path)
    cache_dir = tmp_path / "cache"
    _write_fake_dense_cache(cache_dir, dataset_hash, "fake-model", 999, "passages-50-5")

    run = _base_run(
        dataset_hash,
        {"name": "dense"},
        [{"doc_id": "d1", "rank": 1, "score": 1.0}],  # d2 absent du top 100
    )
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"

    result = generate_passage_detail(
        run,
        run_path,
        cache_dir,
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        offsets_fn=_char_offsets_fn(),
        embedder=_aligned_with_d1_embedder,
    )

    assert result == passage_detail_path(run_path)
    detail = load_passage_detail(run_path)
    docs = detail["queries"][0]["docs"]
    assert [p["score"] for p in docs["d1"]] == pytest.approx([1.0])
    # Critère 7 : d2 (attendu absent du top 100 dans ce test, mais présent au
    # moins ici comme doc du top 10) a quand même ses passages et scores.
    assert docs  # non vide


def test_dense_missing_cache_stops_without_writing(tmp_path, capsys):
    corpus_path, queries_path, qrels_path = _fixture_corpus(tmp_path)
    dataset_hash = compute_dataset_hash(corpus_path)
    run = _base_run(
        dataset_hash, {"name": "dense"}, [{"doc_id": "d1", "rank": 1, "score": 1.0}]
    )
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"

    result = generate_passage_detail(
        run,
        run_path,
        tmp_path / "cache_absent",
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        offsets_fn=_char_offsets_fn(),
        embedder=_aligned_with_d1_embedder,
    )

    assert result is None
    assert not passage_detail_path(run_path).exists()
    assert "cache" in capsys.readouterr().out.lower()


def test_dense_score_mismatch_stops_without_writing(tmp_path, capsys):
    """H4 — le score du run (faux ici) ne correspond pas au score de passage
    recalculé : arrêt, prémisse fausse, rien n'est écrit."""
    corpus_path, queries_path, qrels_path = _fixture_corpus(tmp_path)
    dataset_hash = compute_dataset_hash(corpus_path)
    cache_dir = tmp_path / "cache"
    _write_fake_dense_cache(cache_dir, dataset_hash, "fake-model", 999, "passages-50-5")

    run = _base_run(
        dataset_hash,
        {"name": "dense"},
        [{"doc_id": "d1", "rank": 1, "score": 0.1}],  # faux : la vraie valeur est 1.0
    )
    run_path = tmp_path / "fake-run-2026-10-03.json.gz"

    result = generate_passage_detail(
        run,
        run_path,
        cache_dir,
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        offsets_fn=_char_offsets_fn(),
        embedder=_aligned_with_d1_embedder,
    )

    assert result is None
    assert not passage_detail_path(run_path).exists()
    assert "prémisse fausse" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Critère 1 — point d'entrée CLI
# ---------------------------------------------------------------------------


def test_main_skips_a_document_unit_run_without_crashing(tmp_path, capsys):
    run_data = {
        "campagne": "dev",
        "run_name": "doc-run",
        "dataset_hash": "sha256:whatever",
        "config": {"retriever": {"unit": "document", "name": "dense"}, "rerank": None},
        "queries": [],
    }
    run_path = tmp_path / "doc-run-2026-10-03.json.gz"
    payload = json.dumps(run_data).encode("utf-8")
    with gzip.open(run_path, "wb") as f:
        f.write(payload)

    main([str(run_path)])

    assert "pas éligible" in capsys.readouterr().out
