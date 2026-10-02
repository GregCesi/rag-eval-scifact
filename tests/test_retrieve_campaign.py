"""Tests de `retrieve_campaign` (EXE-88, critères 7, 8, 9 ; EXE-90, critères 3, 4).

Utilise le vrai corpus/les vraies requêtes SciFact (lecture de fichiers, rapide)
mais un embedder et un compteur de tokens fabriqués : aucun modèle d'embedding
n'est chargé. Chaque test pointe `cache_dir` vers `tmp_path`, jamais le cache réel.
"""

from __future__ import annotations

import json

import numpy as np

import rag_eval_scifact.retrieve as retrieve_module
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


def _capturing_embedder(seen: list):
    """Comme `_fake_embedder`, mais garde les textes exacts vus par chaque appel."""

    def _embed(texts: list[str]) -> np.ndarray:
        seen.append(list(texts))
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
        "device",
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


# ---------------------------------------------------------------------------
# EXE-92 — unité passages : découpage, regroupement au niveau document, cache
# ---------------------------------------------------------------------------

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
PASSAGES_WINDOW = 5000  # unité de la fenêtre = caractères (offsets_fn fabriqué)


def _char_offsets_fn():
    """Un token = un caractère : aucun tokenizer réel chargé."""

    def _offsets(text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    return _offsets


def test_passages_unit_returns_only_deduplicated_document_ids_within_top_k(tmp_path):
    results, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    for r in results:
        doc_ids = [d["doc_id"] for d in r.retrieved]
        assert len(doc_ids) <= TOP_K
        assert len(doc_ids) == len(set(doc_ids))


def test_passages_unit_records_the_total_number_of_passages(tmp_path):
    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    assert "n_passages" in stats
    assert stats["n_passages"] >= 5183  # au moins un passage par document


def test_passages_unit_truncated_pct_is_zero_when_passages_fit_the_window(tmp_path):
    _, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    assert stats["truncated_pct"] == 0.0


def test_passages_unit_second_identical_run_does_not_call_embedder(tmp_path):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    second_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder(second_calls),
        offsets_fn=_char_offsets_fn(),
    )

    assert second_calls == []


def test_passages_unit_changing_chunk_size_recomputes_embeddings(tmp_path):
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    other_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=800,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder(other_calls),
        offsets_fn=_char_offsets_fn(),
    )

    assert len(other_calls) == 2  # un appel passages, un appel requêtes


def test_document_unit_is_unaffected_by_a_passages_run_sharing_the_cache_dir(tmp_path):
    """Un run document reproduit v1 à l'identique, même après un run passages
    sur le même `cache_dir` (segments de cache distincts — critère 6)."""
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    document_calls: list[int] = []
    results, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(document_calls),
        token_counter=_fake_token_counter(),
    )

    assert len(document_calls) == 2
    assert len(results[0].retrieved) == TOP_K


# ---------------------------------------------------------------------------
# EXE-94 — Qwen3-Embedding : instruction sur les requêtes, cache séparé
# ---------------------------------------------------------------------------


def test_query_instruction_prefixes_only_query_texts_document_unit(tmp_path):
    seen: list = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_capturing_embedder(seen),
        token_counter=_fake_token_counter(),
        query_instruction="INSTR: ",
    )

    doc_texts_seen, query_texts_seen = seen
    assert not any(t.startswith("INSTR: ") for t in doc_texts_seen)
    assert all(t.startswith("INSTR: ") for t in query_texts_seen)


def test_query_instruction_prefixes_only_query_texts_passages_unit(tmp_path):
    seen: list = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_capturing_embedder(seen),
        offsets_fn=_char_offsets_fn(),
        query_instruction="INSTR: ",
    )

    passage_texts_seen, query_texts_seen = seen
    assert not any(t.startswith("INSTR: ") for t in passage_texts_seen)
    assert all(t.startswith("INSTR: ") for t in query_texts_seen)


def test_no_instruction_by_default_leaves_query_texts_unchanged(tmp_path):
    seen: list = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_capturing_embedder(seen),
        token_counter=_fake_token_counter(),
    )

    _, query_texts_seen = seen
    assert not any(t.startswith("INSTR: ") for t in query_texts_seen)


def test_qwen_model_cache_is_separate_from_minilm_default_run(tmp_path):
    """Changer de modèle vers Qwen (avec instruction) n'affecte jamais le
    cache du run MiniLM par défaut, qui reste rejouable à l'identique
    (critère 6)."""
    minilm_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(minilm_calls),
        token_counter=_fake_token_counter(),
    )

    qwen_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_seq_length=2048,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(qwen_calls),
        token_counter=_fake_token_counter(),
        query_instruction="Instruct: ...\nQuery: ",
    )

    assert len(qwen_calls) == 2  # aucun hit sur le cache MiniLM

    minilm_replay: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        embedder=_fake_embedder(minilm_replay),
        token_counter=_fake_token_counter(),
    )

    assert minilm_replay == []  # MiniLM toujours servi par son propre cache


def test_default_embedder_forwards_configured_batch_size(monkeypatch):
    """H7 — un modèle à fenêtre longue (Qwen) sature la mémoire au batch_size
    par défaut (64) ; `batch_size` doit atteindre `SentenceTransformer.encode`
    tel que configuré, sans jamais charger de modèle réel (classe fabriquée)."""
    seen_batch_sizes: list[int] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.max_seq_length = None

        def encode(self, texts, show_progress_bar=False, batch_size=64):
            seen_batch_sizes.append(batch_size)
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(
        retrieve_module, "SentenceTransformer", _FakeSentenceTransformer
    )

    embed = retrieve_module._default_embedder("fake-model", 2048, batch_size=8)
    embed(["un texte", "un autre"])

    assert seen_batch_sizes == [8]


# ---------------------------------------------------------------------------
# EXE-93 — retriever_name="bm25" : lexical, jamais mis en cache
# ---------------------------------------------------------------------------


def test_bm25_document_unit_indexes_the_whole_corpus_without_truncation(tmp_path):
    results, qrels, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
    )

    assert stats["truncated_pct"] == 0.0
    assert len(results) == len(qrels) == 300
    assert len(results[0].retrieved) == TOP_K


def test_bm25_rare_term_query_ranks_the_matching_document_first(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    corpus_path.write_text(
        "\n".join(
            json.dumps(doc)
            for doc in [
                {"_id": "d1", "title": "", "text": "the cat sat on the mat"},
                {"_id": "d2", "title": "", "text": "the dog ran in the park"},
                {
                    "_id": "d3",
                    "title": "",
                    "text": "a rare xylophone solo echoed through the hall",
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps({"_id": "q1", "text": "xylophone"}), encoding="utf-8"
    )
    qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\td3\t1\n", encoding="utf-8")

    results, _, _, _ = retrieve_campaign(
        top_k=3,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )

    assert results[0].retrieved[0]["doc_id"] == "d3"


def test_bm25_passages_unit_uses_the_same_chunking_and_grouping_as_dense(tmp_path):
    results, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        offsets_fn=_char_offsets_fn(),
    )

    for r in results:
        doc_ids = [d["doc_id"] for d in r.retrieved]
        assert len(doc_ids) <= TOP_K
        assert len(doc_ids) == len(set(doc_ids))
    assert stats["n_passages"] >= 5183


def test_bm25_passages_unit_produces_the_same_passage_count_as_dense(tmp_path):
    _, _, _, dense_stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    _, _, _, bm25_stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        offsets_fn=_char_offsets_fn(),
    )

    assert bm25_stats["n_passages"] == dense_stats["n_passages"]


def test_bm25_does_not_call_the_embedder(tmp_path):
    calls: list[int] = []

    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        embedder=_fake_embedder(calls),
    )

    assert calls == []


def test_changing_bm25_k1_changes_the_results(tmp_path):
    results_default, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        bm25_k1=1.2,
    )

    results_other, _, _, _ = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="bm25",
        bm25_k1=100.0,
    )

    assert [r.retrieved for r in results_default] != [
        r.retrieved for r in results_other
    ]


# ---------------------------------------------------------------------------
# EXE-95 — retriever_name="hybrid" : combine dense et BM25 (union ou RRF)
# ---------------------------------------------------------------------------


def _hybrid_corpus(tmp_path):
    """Corpus de 3 docs où le dense et BM25 classent dans un ordre opposé.

    Un 3e document ("bird") évite que le terme "cat" apparaisse dans
    exactement la moitié du corpus — ce qui annulerait son idf BM25 (df = N/2
    -> idf = 0) et rendrait le classement lexical indéterminé.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"

    corpus_path.write_text(
        "\n".join(
            json.dumps(doc)
            for doc in [
                {"_id": "d1", "title": "", "text": "cat cat cat cat cat"},
                {"_id": "d2", "title": "", "text": "dog dog dog dog dog"},
                {"_id": "d3", "title": "", "text": "bird bird bird"},
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(json.dumps({"_id": "q1", "text": "cat"}), encoding="utf-8")
    qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return corpus_path, queries_path, qrels_path


def _dense_favors_d2_embedder():
    """Vecteur de requête aligné avec d2 : le dense classe d2 avant d1, alors
    que BM25 (terme littéral "cat") classe d1 avant d2 — les deux sens
    opposés, pour prouver que la fusion combine vraiment les deux classements."""
    vectors = {
        " cat cat cat cat cat": [0.0, 1.0],
        " dog dog dog dog dog": [1.0, 0.0],
        " bird bird bird": [0.1, 0.1],
        "cat": [1.0, 0.0],
    }

    def _embed(texts: list[str]) -> np.ndarray:
        return np.array([vectors[t] for t in texts], dtype=np.float32)

    return _embed


def test_hybrid_union_alternates_dense_and_bm25_rankings(tmp_path):
    corpus_path, queries_path, qrels_path = _hybrid_corpus(tmp_path)

    results, _, _, _ = retrieve_campaign(
        top_k=10,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        fusion_mode="union",
        embedder=_dense_favors_d2_embedder(),
        token_counter=_fake_token_counter(),
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )

    # Dense classe d2 avant d1 (vecteur de requête aligné sur d2) ; BM25
    # classe d1 en tête (terme "cat" littéral). L'union alterne 1er dense
    # (d2), 1er BM25 (d1), 2e dense (d3) ; le reste est déjà vu.
    assert [d["doc_id"] for d in results[0].retrieved] == ["d2", "d1", "d3"]


def test_hybrid_sub_rankings_are_exposed_for_tracing(tmp_path):
    corpus_path, queries_path, qrels_path = _hybrid_corpus(tmp_path)

    _, _, _, stats = retrieve_campaign(
        top_k=10,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        embedder=_dense_favors_d2_embedder(),
        token_counter=_fake_token_counter(),
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )

    assert set(stats["sub_rankings"]) == {"dense", "bm25"}
    dense_ranking = stats["sub_rankings"]["dense"]
    bm25_ranking = stats["sub_rankings"]["bm25"]
    assert dense_ranking[0].retrieved[0]["doc_id"] == "d2"
    assert bm25_ranking[0].retrieved[0]["doc_id"] == "d1"


def test_hybrid_reuses_the_dense_cache_on_a_second_run(tmp_path):
    first_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        embedder=_fake_embedder(first_calls),
        token_counter=_fake_token_counter(),
    )

    second_calls: list[int] = []
    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        embedder=_fake_embedder(second_calls),
        token_counter=_fake_token_counter(),
    )

    assert len(first_calls) == 2  # un appel docs, un appel requêtes (dense)
    assert second_calls == []  # cache du sous-retriever dense réutilisé


def test_hybrid_dispatches_to_union_fuse_by_default(tmp_path, monkeypatch):
    calls: list[str] = []
    original = retrieve_module.union_fuse
    monkeypatch.setattr(
        retrieve_module,
        "union_fuse",
        lambda *a, **kw: (calls.append("union"), original(*a, **kw))[1],
    )

    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    assert calls and all(c == "union" for c in calls)


def test_hybrid_rrf_mode_dispatches_to_rrf_fuse(tmp_path, monkeypatch):
    calls: list[str] = []
    original = retrieve_module.rrf_fuse
    monkeypatch.setattr(
        retrieve_module,
        "rrf_fuse",
        lambda *a, **kw: (calls.append("rrf"), original(*a, **kw))[1],
    )

    retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        fusion_mode="rrf",
        rrf_k=30,
        embedder=_fake_embedder([]),
        token_counter=_fake_token_counter(),
    )

    assert calls and all(c == "rrf" for c in calls)


def test_hybrid_works_on_the_passages_unit(tmp_path):
    results, _, _, stats = retrieve_campaign(
        top_k=TOP_K,
        model_name=MODEL,
        max_seq_length=PASSAGES_WINDOW,
        split=SPLIT,
        cache_dir=tmp_path,
        retriever_name="hybrid",
        unit="passages",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedder=_fake_embedder([]),
        offsets_fn=_char_offsets_fn(),
    )

    for r in results:
        doc_ids = [d["doc_id"] for d in r.retrieved]
        assert len(doc_ids) <= TOP_K
        assert len(doc_ids) == len(set(doc_ids))
    assert "n_passages" in stats


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
