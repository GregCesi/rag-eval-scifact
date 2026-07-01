"""Génération des artefacts de run : JSON détaillé + append RESULTS.md.

Produit :
- results/{version}-{date}.json : résultats complets (métriques agrégées + per-query)
- RESULTS.md : une ligne ajoutée (append-only strict)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import chromadb

from rag_eval_scifact.retrieve import RetrievalResult

# --- Constantes ---

RESULTS_DIR = Path("results")
RESULTS_MD = Path("RESULTS.md")
CHROMA_DIR = Path("chroma_data")
COLLECTION_NAME = "scifact_v1"
VERSION = "v1-dense"
MAX_SEQ_LENGTH = 256


def compute_per_query_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> dict:
    """Calcule found@k et best_rank pour une requête.

    found@k (bool) = au moins un doc pertinent dans le top-k
    best_rank (int | None) = rang du premier doc pertinent dans le top-100
    """
    best_rank = None
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            best_rank = i
            break

    return {
        "found@1": best_rank is not None and best_rank <= 1,
        "found@5": best_rank is not None and best_rank <= 5,
        "found@10": best_rank is not None and best_rank <= 10,
        "found@100": best_rank is not None and best_rank <= 100,
        "best_rank": best_rank,
    }


def get_token_counts(doc_ids: list[str]) -> dict[str, int]:
    """Récupère les token_count depuis ChromaDB pour les docs donnés."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(ids=doc_ids, include=["metadatas"])
    return {
        doc_id: meta["token_count"]
        for doc_id, meta in zip(result["ids"], result["metadatas"])
    }


def generate_run_json(
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
    metrics: dict[str, float],
    dataset_hash: str,
    run_date: datetime,
) -> dict:
    """Assemble le JSON complet du run (niveau run + niveau requête)."""
    # Collecter tous les doc_ids pertinents pour récupérer leurs token_count
    all_relevant_ids = set()
    for relevant_set in qrels.values():
        all_relevant_ids.update(relevant_set)
    token_counts = get_token_counts(list(all_relevant_ids))

    # Assembler les requêtes
    queries_json = []
    for r in results:
        relevant_ids = qrels.get(r.query_id, set())
        retrieved_ids = [d["doc_id"] for d in r.retrieved]

        per_query = compute_per_query_metrics(retrieved_ids, relevant_ids)

        expected_docs = [
            {"doc_id": doc_id, "token_count": token_counts.get(doc_id, 0)}
            for doc_id in sorted(relevant_ids)
        ]

        queries_json.append({
            "query_id": r.query_id,
            "query_text": r.query_text,
            "expected_docs": expected_docs,
            "retrieved_top100": r.retrieved,
            "per_query_metrics": per_query,
        })

    return {
        "version": VERSION,
        "date": run_date.isoformat(timespec="seconds"),
        "dataset_hash": dataset_hash,
        "config": {
            "model": "all-MiniLM-L6-v2",
            "max_seq_length": MAX_SEQ_LENGTH,
            "dim": 384,
        },
        "metrics": metrics,
        "queries": queries_json,
    }


def write_run_json(run_data: dict) -> Path:
    """Écrit le JSON du run dans results/{version}-{date}.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    date_str = run_data["date"].replace(":", "-")
    filename = f"{run_data['version']}-{date_str}.json"
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)
    return path


def append_results_md(metrics: dict[str, float]) -> None:
    """Append une ligne au tableau RESULTS.md (append-only strict)."""
    header = (
        "| version | date | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR | dette | note d'analyse |\n"
        "|---------|------|-----|-----|------|-------|---------|-----|-------|----------------|\n"
    )

    if not RESULTS_MD.exists():
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(header)

    date_str = datetime.now().strftime("%Y-%m-%d")
    line = (
        f"| {VERSION} "
        f"| {date_str} "
        f"| {metrics['recall@1']:.4f} "
        f"| {metrics['recall@5']:.4f} "
        f"| {metrics['recall@10']:.4f} "
        f"| {metrics['recall@100']:.4f} "
        f"| {metrics['ndcg@10']:.4f} "
        f"| {metrics['mrr']:.4f} "
        f"| max_seq={MAX_SEQ_LENGTH} "
        f"| |\n"
    )

    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(line)
