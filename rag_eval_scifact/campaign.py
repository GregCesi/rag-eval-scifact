"""Orchestration d'un run de campagne : config résolue + artefacts versionnés.

Sépare le calcul (métriques, écriture) du retrieval (embedding + ChromaDB) pour
que l'écriture des artefacts — non-écrasement, append RESULTS.md, format JSON —
se teste sans jamais charger de modèle d'embedding. Le chargement du modèle et
l'appel à `retrieve()` restent dans `run_campaign.py` (le point d'entrée Hydra).

Format des artefacts : .claude/rules/versioning.md.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

from rag_eval_scifact.metrics import mrr, ndcg_at_k, recall_at_k
from rag_eval_scifact.retrieve import RetrievalResult
from rag_eval_scifact.run_output import compute_per_query_metrics

RESULTS_DIR = Path("results")
RESULTS_MD = Path("RESULTS.md")


def compute_aggregate_metrics(
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    """Macro-average des 6 métriques fait-main sur les requêtes du run."""
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
    return {k: v / n for k, v in agg.items()}


def build_run_artifact(
    campagne: str,
    run_name: str,
    config: dict,
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
    metrics: dict[str, float],
    dataset_hash: str,
    run_date: datetime,
    token_counts: dict[str, int] | None = None,
) -> dict:
    """Assemble le JSON complet d'un run de campagne (format versioning.md)."""
    token_counts = token_counts or {}

    queries_json = []
    for r in results:
        relevant_ids = qrels.get(r.query_id, set())
        retrieved_ids = [d["doc_id"] for d in r.retrieved]
        per_query = compute_per_query_metrics(retrieved_ids, relevant_ids)
        expected_docs = [
            {"doc_id": doc_id, "token_count": token_counts.get(doc_id, 0)}
            for doc_id in sorted(relevant_ids)
        ]
        queries_json.append(
            {
                "query_id": r.query_id,
                "query_text": r.query_text,
                "expected_docs": expected_docs,
                "retrieved_top100": r.retrieved,
                "per_query_metrics": per_query,
            }
        )

    return {
        "campagne": campagne,
        "run_name": run_name,
        "date": run_date.isoformat(timespec="seconds"),
        "dataset_hash": dataset_hash,
        "config": config,
        "metrics": metrics,
        "queries": queries_json,
    }


def write_campaign_json(
    campagne: str,
    run_name: str,
    run_data: dict,
    run_date: datetime,
) -> Path:
    """Écrit results/{campagne}/{run_name}-{date}.json.gz sans jamais écraser un run existant."""
    campaign_dir = RESULTS_DIR / campagne
    campaign_dir.mkdir(parents=True, exist_ok=True)

    date_str = run_date.strftime("%Y-%m-%dT%H-%M-%S-%f")
    path = campaign_dir / f"{run_name}-{date_str}.json.gz"
    suffix = 0
    while path.exists():
        suffix += 1
        path = campaign_dir / f"{run_name}-{date_str}-{suffix}.json.gz"

    payload = json.dumps(run_data, indent=2, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return path


def append_results_md_campaign(
    campagne: str,
    run_name: str,
    metrics: dict[str, float],
    dette: str,
    run_date: datetime,
) -> None:
    """Append une ligne au tableau RESULTS.md pour un run de campagne (append-only strict)."""
    header = (
        "| version | date | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR | dette | note d'analyse |\n"
        "|---------|------|-----|-----|------|-------|---------|-----|-------|----------------|\n"
    )
    if not RESULTS_MD.exists():
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(header)

    date_str = run_date.strftime("%Y-%m-%d")
    line = (
        f"| {campagne}/{run_name} "
        f"| {date_str} "
        f"| {metrics['recall@1']:.4f} "
        f"| {metrics['recall@5']:.4f} "
        f"| {metrics['recall@10']:.4f} "
        f"| {metrics['recall@100']:.4f} "
        f"| {metrics['ndcg@10']:.4f} "
        f"| {metrics['mrr']:.4f} "
        f"| {dette} "
        f"| |\n"
    )
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(line)


def run_campaign(
    campagne: str,
    run_name: str,
    config: dict,
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
    dataset_hash: str,
    run_date: datetime,
    token_counts: dict[str, int] | None = None,
) -> dict:
    """Calcule les métriques d'un run déjà retrievé et écrit ses artefacts de campagne."""
    metrics = compute_aggregate_metrics(results, qrels)
    run_data = build_run_artifact(
        campagne,
        run_name,
        config,
        results,
        qrels,
        metrics,
        dataset_hash,
        run_date,
        token_counts,
    )
    json_path = write_campaign_json(campagne, run_name, run_data, run_date)

    retriever_cfg = config.get("retriever") or {}
    dette = f"max_seq={retriever_cfg.get('max_seq_length', '?')}"
    append_results_md_campaign(campagne, run_name, metrics, dette, run_date)

    return {"run_data": run_data, "json_path": json_path, "metrics": metrics}
