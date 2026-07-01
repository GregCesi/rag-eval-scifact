"""Orchestrateur de run d'évaluation complet.

Enchaîne : retrieval dense → métriques fait-main → artefacts (JSON + RESULTS.md).
Un seul point d'entrée pour un run reproductible bout-en-bout.

Usage : python -m rag_eval_scifact.run_eval
"""

from __future__ import annotations

from datetime import datetime

from rag_eval_scifact.metrics import mrr, ndcg_at_k, recall_at_k
from rag_eval_scifact.retrieve import retrieve
from rag_eval_scifact.run_output import (
    append_results_md,
    generate_run_json,
    write_run_json,
)


def run_eval() -> None:
    """Exécute un run d'évaluation complet."""
    run_date = datetime.now()

    # --- Retrieval ---
    results, qrels, dataset_hash = retrieve()

    # --- Métriques agrégées (macro-average sur les 300 requêtes) ---
    print("\nCalcul des métriques...")
    n = len(results)
    agg: dict[str, float] = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@100": 0.0,
        "ndcg@10": 0.0,
        "mrr": 0.0,
    }

    for r in results:
        retrieved_ids = [d["doc_id"] for d in r.retrieved]
        relevant_ids = qrels.get(r.query_id, set())

        agg["recall@1"] += recall_at_k(retrieved_ids, relevant_ids, 1)
        agg["recall@5"] += recall_at_k(retrieved_ids, relevant_ids, 5)
        agg["recall@10"] += recall_at_k(retrieved_ids, relevant_ids, 10)
        agg["recall@100"] += recall_at_k(retrieved_ids, relevant_ids, 100)
        agg["ndcg@10"] += ndcg_at_k(retrieved_ids, relevant_ids, 10)
        agg["mrr"] += mrr(retrieved_ids, relevant_ids)

    metrics = {k: v / n for k, v in agg.items()}

    # --- Artefacts ---
    print("Génération des artefacts...")
    run_data = generate_run_json(results, qrels, metrics, dataset_hash, run_date)
    json_path = write_run_json(run_data)
    append_results_md(metrics)

    # --- Résumé console ---
    print("\n" + "=" * 60)
    print(f"  RUN COMPLET — {run_data['version']} — {run_data['date']}")
    print("=" * 60)
    print(f"  dataset_hash : {dataset_hash}")
    print(f"  config       : {run_data['config']}")
    print()
    for k, v in metrics.items():
        print(f"  {k:12s} = {v:.4f}")
    print()
    print(f"  JSON  : {json_path}")
    print(f"  TABLE : RESULTS.md (ligne ajoutée)")
    print("=" * 60)
    print("\n  Rappel : commiter les artefacts + taguer (run non commité = run inexistant)")


if __name__ == "__main__":
    run_eval()
