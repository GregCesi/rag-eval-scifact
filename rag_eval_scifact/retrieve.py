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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

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


def load_test_queries(queries_path: Path, qrels_path: Path) -> tuple[list[dict], dict[str, set[str]]]:
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


def retrieve() -> tuple[list[RetrievalResult], dict[str, set[str]], str]:
    """Pipeline de retrieval complet.

    Retourne:
        results: liste de RetrievalResult (1 par requête test)
        qrels: {query_id: {doc_ids pertinents}}
        dataset_hash: hash relu depuis la collection ChromaDB
    """
    # Charger les requêtes test et qrels
    print("Chargement des requêtes test et qrels...")
    queries, qrels = load_test_queries(QUERIES_PATH, QRELS_PATH)
    print(f"  {len(queries)} requêtes test, {sum(len(v) for v in qrels.values())} paires qrel.")

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
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    query_texts = [q["text"] for q in queries]
    query_embeddings = model.encode(query_texts, show_progress_bar=False, batch_size=64)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"  {len(query_texts)} requêtes embeddées (shape={query_embeddings.shape}).")

    # L2-normaliser requêtes ET docs → cosinus = dot product
    print("Calcul cosinus brute-force (numpy)...")
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_norm = doc_embeddings / doc_norms

    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / query_norms

    # Matrice de similarité : (300 x 384) @ (384 x 5183) → (300 x 5183)
    sim_matrix = query_embeddings_norm @ doc_embeddings_norm.T
    print(f"  Matrice de similarité : {sim_matrix.shape}")

    # Top-100 par requête (tri décroissant)
    print("Extraction top-100 par requête...")
    results: list[RetrievalResult] = []

    for i, query in enumerate(queries):
        scores = sim_matrix[i]
        # argpartition pour efficacité, puis tri des top-k
        top_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        retrieved = []
        for rank, idx in enumerate(top_indices, start=1):
            retrieved.append({
                "doc_id": doc_ids[idx],
                "rank": rank,
                "score": float(scores[idx]),
            })

        results.append(RetrievalResult(
            query_id=query["_id"],
            query_text=query["text"],
            retrieved=retrieved,
        ))

    print(f"  {len(results)} résultats de retrieval générés.")
    return results, qrels, dataset_hash


if __name__ == "__main__":
    results, qrels, dataset_hash = retrieve()

    # Résumé rapide
    print(f"\nRésumé :")
    print(f"  Requêtes : {len(results)}")
    print(f"  Top-k    : {TOP_K}")
    print(f"  Hash     : {dataset_hash}")

    # Vérif basique : une requête avec son premier résultat
    r = results[0]
    print(f"\n  Exemple — query '{r.query_id}': '{r.query_text[:60]}...'")
    print(f"    #1: doc_id={r.retrieved[0]['doc_id']}, score={r.retrieved[0]['score']:.4f}")
    print(f"    #100: doc_id={r.retrieved[99]['doc_id']}, score={r.retrieved[99]['score']:.4f}")
