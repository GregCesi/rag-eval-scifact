"""Retrieval dense brute-force cosinus sur le corpus SciFact.

Pour chaque requête test, embedde la requête avec MiniLM puis calcule la similarité
cosinus EXACTE (numpy, pas HNSW) contre les 5183 docs. Retourne le top-100 par
requête, trié par score décroissant.

Le retrieval exact garantit que le top-100 est le vrai top-100 (pas une approximation).
ChromaDB sert uniquement de store (embeddings, metadata, dataset_hash).
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_eval_scifact.bm25 import build_bm25_index, score_queries
from rag_eval_scifact.cache import get_document_embeddings, get_ranking
from rag_eval_scifact.chunking import chunk_text, default_offsets_fn
from rag_eval_scifact.ingest import CORPUS_PATH, compute_dataset_hash, load_corpus

# --- Constantes ---

QUERIES_PATH = Path("data/scifact/queries.jsonl")
QRELS_PATH = Path("data/scifact/qrels/test.tsv")
CHROMA_DIR = Path("chroma_data")
COLLECTION_NAME = "scifact_v1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
TOP_K = 100


@dataclass
class RetrievalResult:
    """Résultat de retrieval pour une requête (contrat figé avec le harness d'éval)."""

    query_id: str
    query_text: str
    retrieved: list[dict] = field(default_factory=list)
    # top-100, trié par score décroissant (similarité max en premier)
    # chaque dict = {"doc_id": str, "rank": int, "score": float}
    # score = similarité cosinus directe (brute-force numpy)


def load_test_queries(
    queries_path: Path, qrels_path: Path
) -> tuple[list[dict], dict[str, set[str]]]:
    """Charge les requêtes du split test et leurs qrels.

    Retourne:
        queries: liste de {"_id": str, "text": str} pour les 300 requêtes test
        qrels: {query_id: {doc_id, ...}} — docs pertinents par requête
    """
    # Charger les qrels pour identifier les query IDs du split test
    qrels: dict[str, set[str]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(did)

    test_qids = set(qrels.keys())

    # Charger les textes des requêtes test
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in test_qids:
                queries.append(q)

    assert len(queries) == len(test_qids), (
        f"Attendu {len(test_qids)} requêtes test, trouvé {len(queries)}"
    )

    return queries, qrels


def _top_k_from_matrix(
    queries: list[dict],
    doc_ids: list[str],
    sim_matrix: np.ndarray,
    top_k: int,
) -> list[RetrievalResult]:
    """Top-k par requête à partir d'une matrice (n_requêtes, n_docs) de scores déjà calculés.

    Partagé par le dense (cosinus) et BM25 (EXE-93) : seul le calcul de
    `sim_matrix` diffère entre les deux.
    """
    results: list[RetrievalResult] = []
    for i, query in enumerate(queries):
        scores = sim_matrix[i]
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        retrieved = [
            {"doc_id": doc_ids[idx], "rank": rank, "score": float(scores[idx])}
            for rank, idx in enumerate(top_indices, start=1)
        ]
        results.append(
            RetrievalResult(
                query_id=query["_id"],
                query_text=query["text"],
                retrieved=retrieved,
            )
        )
    return results


def _rank_top_k(
    queries: list[dict],
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> list[RetrievalResult]:
    """Cosinus brute-force (L2-normalise puis produit scalaire) + top-k par requête."""
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_norm = doc_embeddings / doc_norms

    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / query_norms

    sim_matrix = query_embeddings_norm @ doc_embeddings_norm.T
    return _top_k_from_matrix(queries, doc_ids, sim_matrix, top_k)


def group_passage_scores(
    passage_doc_ids: list[str],
    scores: np.ndarray,
    grouping: str,
    grouping_top_n: int,
) -> dict[str, float]:
    """Regroupe des scores de passages (une requête) en scores de documents.

    `max` (défaut) : le score d'un document est le meilleur score parmi
    TOUS ses passages (jamais restreint par `grouping_top_n` : le max sur
    l'ensemble complet est aussi simple à calculer que sur un sous-ensemble).
    `sum` : la somme des scores de ses passages présents parmi les
    `grouping_top_n` premiers passages de la requête (tous documents
    confondus) ; un document sans passage dans ce sous-ensemble n'apparaît pas.
    """
    if grouping == "sum":
        n = min(grouping_top_n, len(scores))
        if n <= 0:
            return {}
        top_indices = np.argpartition(scores, -n)[-n:]
        doc_scores: dict[str, float] = {}
        for idx in top_indices:
            doc_id = passage_doc_ids[idx]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + float(scores[idx])
        return doc_scores

    doc_scores = {}
    for idx, doc_id in enumerate(passage_doc_ids):
        score = float(scores[idx])
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score
    return doc_scores


def _top_k_grouped_from_matrix(
    queries: list[dict],
    passage_doc_ids: list[str],
    sim_matrix: np.ndarray,
    top_k: int,
    grouping: str,
    grouping_top_n: int,
) -> list[RetrievalResult]:
    """Regroupement passage -> document à partir d'une matrice de scores déjà calculés.

    Partagé par le dense (cosinus) et BM25 (EXE-93 critère 3 : même
    regroupement que le dense) ; seul le calcul de `sim_matrix` diffère. La
    liste classée retournée ne contient que des identifiants de documents,
    sans doublon (clés d'un dict), top_k au plus.
    """
    results: list[RetrievalResult] = []
    for i, query in enumerate(queries):
        doc_scores = group_passage_scores(
            passage_doc_ids, sim_matrix[i], grouping, grouping_top_n
        )
        ranked_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[
            :top_k
        ]
        retrieved = [
            {"doc_id": doc_id, "rank": rank, "score": score}
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1)
        ]
        results.append(
            RetrievalResult(
                query_id=query["_id"], query_text=query["text"], retrieved=retrieved
            )
        )
    return results


def _rank_top_k_grouped(
    queries: list[dict],
    passage_ids: list[str],
    passage_doc_ids: list[str],
    passage_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
    grouping: str,
    grouping_top_n: int,
) -> list[RetrievalResult]:
    """Cosinus brute-force sur les passages, puis regroupement au niveau document."""
    passage_norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    passage_embeddings_norm = passage_embeddings / passage_norms

    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / query_norms

    sim_matrix = query_embeddings_norm @ passage_embeddings_norm.T
    return _top_k_grouped_from_matrix(
        queries, passage_doc_ids, sim_matrix, top_k, grouping, grouping_top_n
    )


def retrieve(
    top_k: int = TOP_K,
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> tuple[list[RetrievalResult], dict[str, set[str]], str]:
    """Pipeline de retrieval complet.

    Les trois paramètres par défaut reproduisent v1 à l'identique ; une
    campagne les surcharge via sa config résolue, jamais en modifiant ce fichier.

    Retourne:
        results: liste de RetrievalResult (1 par requête test)
        qrels: {query_id: {doc_ids pertinents}}
        dataset_hash: hash relu depuis la collection ChromaDB
    """
    # Charger les requêtes test et qrels
    print("Chargement des requêtes test et qrels...")
    queries, qrels = load_test_queries(QUERIES_PATH, QRELS_PATH)
    print(
        f"  {len(queries)} requêtes test, {sum(len(v) for v in qrels.values())} paires qrel."
    )

    # Charger les embeddings de docs depuis ChromaDB
    print("Chargement des embeddings depuis ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    dataset_hash = collection.metadata["dataset_hash"]
    print(f"  dataset_hash = {dataset_hash[:30]}...")

    all_docs = collection.get(include=["embeddings"])
    doc_ids = all_docs["ids"]
    doc_embeddings = np.array(all_docs["embeddings"], dtype=np.float32)
    print(f"  {len(doc_ids)} embeddings chargés (shape={doc_embeddings.shape}).")

    # Embedder les requêtes
    print("Embedding des requêtes...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    query_texts = [q["text"] for q in queries]
    query_embeddings = model.encode(query_texts, show_progress_bar=False, batch_size=64)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"  {len(query_texts)} requêtes embeddées (shape={query_embeddings.shape}).")

    # L2-normaliser requêtes ET docs → cosinus = dot product
    print("Calcul cosinus brute-force (numpy)...")
    print(f"Extraction top-{top_k} par requête...")
    results = _rank_top_k(queries, doc_ids, doc_embeddings, query_embeddings, top_k)

    print(f"  {len(results)} résultats de retrieval générés.")
    return results, qrels, dataset_hash


def compute_truncated_pct(token_counts: list[int], max_seq_length: int) -> float:
    """Part (en pourcentage) des documents dont le comptage dépasse la fenêtre."""
    if not token_counts:
        return 0.0
    truncated = sum(1 for tc in token_counts if tc > max_seq_length)
    return 100.0 * truncated / len(token_counts)


def _default_token_counter(model_name: str) -> Callable[[list[str]], list[int]]:
    """Compte les tokens avec le tokenizer du modèle, chargé paresseusement, au plus une fois.

    Une campagne de test injecte son propre compteur (fonction fabriquée) pour ne
    jamais charger de tokenizer réel — voir `retrieve_campaign`.
    """
    tokenizer_box: dict[str, Any] = {}

    def _count(texts: list[str]) -> list[int]:
        if "tokenizer" not in tokenizer_box:
            from transformers import AutoTokenizer

            tokenizer_box["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        tokenizer = tokenizer_box["tokenizer"]
        return [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]

    return _count


def _default_embedder(
    model_name: str, max_seq_length: int
) -> Callable[[list[str]], np.ndarray]:
    """Encode avec un `SentenceTransformer` chargé paresseusement, au plus une fois.

    Une campagne de test injecte son propre embedder (fonction fabriquée) pour ne
    jamais charger de modèle réel — voir `retrieve_campaign`.
    """
    model_box: dict[str, SentenceTransformer] = {}

    def _embed(texts: list[str]) -> np.ndarray:
        if "model" not in model_box:
            model = SentenceTransformer(model_name)
            model.max_seq_length = max_seq_length
            model_box["model"] = model
        embeddings = model_box["model"].encode(
            texts, show_progress_bar=False, batch_size=64
        )
        return np.array(embeddings, dtype=np.float32)

    return _embed


def retrieve_campaign(
    top_k: int,
    model_name: str,
    max_seq_length: int,
    split: str,
    cache_dir: Path,
    unit: str = "document",
    chunk_size: int = 128,
    chunk_overlap: int = 32,
    grouping: str = "max",
    grouping_top_n: int = 1000,
    retriever_name: str = "dense",
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    token_counter: Callable[[list[str]], list[int]] | None = None,
    offsets_fn: Callable[[str], list[tuple[int, int]]] | None = None,
    corpus_path: Path = CORPUS_PATH,
    queries_path: Path = QUERIES_PATH,
    qrels_path: Path = QRELS_PATH,
) -> tuple[list[RetrievalResult], dict[str, set[str]], str, dict[str, float]]:
    """Retrieval dense d'un run de campagne, embeddings et classement mis en cache.

    Contrairement à `retrieve()` (v1 historique, lit `chroma_data/`), ce chemin
    embedde les documents lui-même — nécessaire pour qu'une stratégie puisse
    changer de modèle ou de fenêtre — et met en cache les deux calculs lourds
    (embeddings, classement de premier étage) sous `cache_dir`, séparé de
    `chroma_data/`. `embedder`, `token_counter` et `offsets_fn` sont injectables
    pour les tests : aucun modèle ni tokenizer n'est chargé tant qu'ils ne sont
    pas fournis par l'appelant.

    `unit` (EXE-92) : `"document"` (défaut, reproduit v1 à l'identique) ou
    `"passages"` — chaque document est découpé en fenêtres de `chunk_size`
    tokens avec `chunk_overlap` de chevauchement (`rag_eval_scifact.chunking`),
    le retrieval se fait sur les passages puis se regroupe au niveau document
    (`grouping` : `"max"` ou `"sum"`, voir `group_passage_scores`).

    `retriever_name` (EXE-93) : `"dense"` (défaut) ou `"bm25"` — lexical,
    scores calculés par `rag_eval_scifact.bm25` (`bm25_k1`, `bm25_b`), jamais
    mis en cache (indexation rapide, aucun modèle chargé). Sur `unit`
    `"passages"`, BM25 réutilise le même découpage (`chunk_text`) et le même
    regroupement (`group_passage_scores`) que le dense.

    En plus du triplet historique, retourne un dict `stats` (EXE-90, EXE-92) :
    - `truncated_pct` : part des documents (unité document) ou des passages
      (unité passages) dont le comptage dépasse la fenêtre, recalculée à
      chaque run (pas un calcul mis en cache).
    - `indexing_duration_seconds` : durée de l'embedding des documents ou des
      passages, 0.0 quand le cache (embeddings ou classement) a servi.
    - `avg_retrieval_latency_ms` : latence moyenne par requête de l'étape
      d'embedding des requêtes + classement, 0.0 quand le classement vient du cache.
    - `n_passages` (unité passages seulement) : nombre total de passages.
    """
    dataset_hash = compute_dataset_hash(corpus_path)
    print(f"  dataset_hash = {dataset_hash[:30]}...")

    queries, qrels = load_test_queries(queries_path, qrels_path)

    docs = load_corpus(corpus_path)
    doc_ids = [doc["_id"] for doc in docs]
    doc_texts = [doc["title"] + " " + doc["text"] for doc in docs]

    embed = embedder or _default_embedder(model_name, max_seq_length)
    timings = {"indexing_duration_seconds": 0.0, "retrieval_duration_seconds": 0.0}
    stats: dict[str, float] = {}

    if retriever_name == "bm25":
        query_texts = [q["text"] for q in queries]

        if unit == "passages":
            offsets = offsets_fn or default_offsets_fn(model_name)
            passages = [
                passage
                for doc_id, text in zip(doc_ids, doc_texts)
                for passage in chunk_text(
                    doc_id, text, chunk_size, chunk_overlap, offsets
                )
            ]
            passage_doc_ids = [p.doc_id for p in passages]
            passage_texts = [p.text for p in passages]
            stats["truncated_pct"] = compute_truncated_pct(
                [p.token_count for p in passages], max_seq_length
            )
            stats["n_passages"] = float(len(passages))

            start = time.perf_counter()
            index = build_bm25_index(passage_texts, bm25_k1, bm25_b)
            timings["indexing_duration_seconds"] = time.perf_counter() - start

            retrieval_start = time.perf_counter()
            sim_matrix = score_queries(index, query_texts)
            ranked = _top_k_grouped_from_matrix(
                queries, passage_doc_ids, sim_matrix, top_k, grouping, grouping_top_n
            )
            timings["retrieval_duration_seconds"] = (
                time.perf_counter() - retrieval_start
            )
        else:
            # BM25 indexe title + text sans fenêtre de tokens : jamais tronqué
            # (EXE-93 critère 2).
            stats["truncated_pct"] = 0.0

            start = time.perf_counter()
            index = build_bm25_index(doc_texts, bm25_k1, bm25_b)
            timings["indexing_duration_seconds"] = time.perf_counter() - start

            retrieval_start = time.perf_counter()
            sim_matrix = score_queries(index, query_texts)
            ranked = _top_k_from_matrix(queries, doc_ids, sim_matrix, top_k)
            timings["retrieval_duration_seconds"] = (
                time.perf_counter() - retrieval_start
            )

        ranking = [
            {
                "query_id": r.query_id,
                "query_text": r.query_text,
                "retrieved": r.retrieved,
            }
            for r in ranked
        ]
    elif unit == "passages":
        offsets = offsets_fn or default_offsets_fn(model_name)
        passages = [
            passage
            for doc_id, text in zip(doc_ids, doc_texts)
            for passage in chunk_text(doc_id, text, chunk_size, chunk_overlap, offsets)
        ]
        passage_ids = [p.passage_id for p in passages]
        passage_doc_ids = [p.doc_id for p in passages]
        passage_texts = [p.text for p in passages]
        stats["truncated_pct"] = compute_truncated_pct(
            [p.token_count for p in passages], max_seq_length
        )
        stats["n_passages"] = float(len(passages))

        chunking_unit = f"passages-{chunk_size}-{chunk_overlap}"
        ranking_unit = f"{chunking_unit}-{grouping}-{grouping_top_n}"

        def _compute_ranking() -> list[dict]:
            def _timed_embed_passages() -> np.ndarray:
                start = time.perf_counter()
                vectors = embed(passage_texts)
                timings["indexing_duration_seconds"] = time.perf_counter() - start
                return vectors

            passage_embeddings, _ = get_document_embeddings(
                cache_dir,
                dataset_hash,
                model_name,
                max_seq_length,
                passage_ids,
                _timed_embed_passages,
                unit=chunking_unit,
            )

            retrieval_start = time.perf_counter()
            query_texts = [q["text"] for q in queries]
            query_embeddings = embed(query_texts)
            ranked = _rank_top_k_grouped(
                queries,
                passage_ids,
                passage_doc_ids,
                passage_embeddings,
                query_embeddings,
                top_k,
                grouping,
                grouping_top_n,
            )
            timings["retrieval_duration_seconds"] = (
                time.perf_counter() - retrieval_start
            )

            return [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "retrieved": r.retrieved,
                }
                for r in ranked
            ]

        ranking, _ = get_ranking(
            cache_dir,
            dataset_hash,
            model_name,
            max_seq_length,
            top_k,
            split,
            _compute_ranking,
            unit=ranking_unit,
        )
    else:
        count_tokens = token_counter or _default_token_counter(model_name)
        stats["truncated_pct"] = compute_truncated_pct(
            count_tokens(doc_texts), max_seq_length
        )

        def _compute_ranking() -> list[dict]:
            def _timed_embed_docs() -> np.ndarray:
                start = time.perf_counter()
                vectors = embed(doc_texts)
                timings["indexing_duration_seconds"] = time.perf_counter() - start
                return vectors

            doc_embeddings, _ = get_document_embeddings(
                cache_dir,
                dataset_hash,
                model_name,
                max_seq_length,
                doc_ids,
                _timed_embed_docs,
                unit="document",
            )

            retrieval_start = time.perf_counter()
            query_texts = [q["text"] for q in queries]
            query_embeddings = embed(query_texts)
            ranked = _rank_top_k(
                queries, doc_ids, doc_embeddings, query_embeddings, top_k
            )
            timings["retrieval_duration_seconds"] = (
                time.perf_counter() - retrieval_start
            )

            return [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "retrieved": r.retrieved,
                }
                for r in ranked
            ]

        ranking, _ = get_ranking(
            cache_dir,
            dataset_hash,
            model_name,
            max_seq_length,
            top_k,
            split,
            _compute_ranking,
            unit="document",
        )

    results = [
        RetrievalResult(
            query_id=entry["query_id"],
            query_text=entry["query_text"],
            retrieved=entry["retrieved"],
        )
        for entry in ranking
    ]

    stats["avg_retrieval_latency_ms"] = (
        timings["retrieval_duration_seconds"] * 1000 / len(queries) if queries else 0.0
    )
    stats["indexing_duration_seconds"] = timings["indexing_duration_seconds"]
    return results, qrels, dataset_hash, stats


if __name__ == "__main__":
    results, qrels, dataset_hash = retrieve()

    # Résumé rapide
    print("\nRésumé :")
    print(f"  Requêtes : {len(results)}")
    print(f"  Top-k    : {TOP_K}")
    print(f"  Hash     : {dataset_hash}")

    # Vérif basique : une requête avec son premier résultat
    r = results[0]
    print(f"\n  Exemple — query '{r.query_id}': '{r.query_text[:60]}...'")
    print(
        f"    #1: doc_id={r.retrieved[0]['doc_id']}, score={r.retrieved[0]['score']:.4f}"
    )
    print(
        f"    #100: doc_id={r.retrieved[99]['doc_id']}, score={r.retrieved[99]['score']:.4f}"
    )
