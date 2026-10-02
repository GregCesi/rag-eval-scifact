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
V1_BUCKETS_PATH = Path("results/v1-buckets.json")
BUCKET_NAMES = ("perfect", "near_miss", "deep_miss", "miss_100")


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


def load_query_buckets() -> dict[str, str]:
    """Charge `V1_BUCKETS_PATH` : query_id -> nom du bucket v1 (figé, lecture seule)."""
    data = json.loads(V1_BUCKETS_PATH.read_text(encoding="utf-8"))
    return {qid: info["bucket"] for qid, info in data["queries"].items()}


def compute_bucket_found_at_10(
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
    query_buckets: dict[str, str],
) -> dict[str, float]:
    """Pour chaque bucket v1, part des requêtes avec un doc attendu dans le top 10."""
    totals = dict.fromkeys(BUCKET_NAMES, 0)
    founds = dict.fromkeys(BUCKET_NAMES, 0)
    for r in results:
        bucket = query_buckets.get(r.query_id)
        if bucket not in totals:
            continue
        totals[bucket] += 1
        relevant_ids = qrels.get(r.query_id, set())
        retrieved_ids = [d["doc_id"] for d in r.retrieved]
        if compute_per_query_metrics(retrieved_ids, relevant_ids)["found@10"]:
            founds[bucket] += 1
    return {
        f"bucket_found_at_10_{name}": (
            founds[name] / totals[name] if totals[name] else 0.0
        )
        for name in BUCKET_NAMES
    }


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
    extended_metrics: dict[str, float] | None = None,
    device: str = "cpu",
) -> dict:
    """Assemble le JSON complet d'un run de campagne (format versioning.md).

    `extended_metrics` (EXE-90) va dans un champ séparé de `metrics` : les 6
    métriques historiques et leur ensemble de clés ne changent pas. `device`
    (EXE-94 critère 7) est une chaîne, donc hors d'`extended_metrics` (valeurs
    numériques, journalisées par `mlflow.log_metrics`).
    """
    token_counts = token_counts or {}
    extended_metrics = extended_metrics or {}

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
        "device": device,
        "config": config,
        "metrics": metrics,
        "extended_metrics": extended_metrics,
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
    truncated_pct: float = 0.0,
    avg_retrieval_latency_ms: float = 0.0,
    indexing_duration_seconds: float = 0.0,
    n_passages: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Calcule les métriques d'un run déjà retrievé et écrit ses artefacts de campagne.

    `truncated_pct`, `avg_retrieval_latency_ms`, `indexing_duration_seconds`
    et `device` viennent de `retrieve_campaign` (EXE-90, EXE-94) : ce module ne
    charge aucun modèle, il ne fait que les reporter. `n_passages` (EXE-92)
    vaut 0 pour un run unité document. La part par bucket v1
    (`compute_bucket_found_at_10`) est calculée ici, à partir de
    `V1_BUCKETS_PATH` (lecture seule).
    """
    metrics = compute_aggregate_metrics(results, qrels)
    query_buckets = load_query_buckets()
    extended_metrics = {
        **compute_bucket_found_at_10(results, qrels, query_buckets),
        "truncated_pct": truncated_pct,
        "avg_retrieval_latency_ms": avg_retrieval_latency_ms,
        "indexing_duration_seconds": indexing_duration_seconds,
        "n_passages": n_passages,
    }
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
        extended_metrics,
        device,
    )
    json_path = write_campaign_json(campagne, run_name, run_data, run_date)

    retriever_cfg = config.get("retriever") or {}
    dette = f"max_seq={retriever_cfg.get('max_seq_length', '?')}"
    append_results_md_campaign(campagne, run_name, metrics, dette, run_date)

    return {
        "run_data": run_data,
        "json_path": json_path,
        "metrics": metrics,
        "extended_metrics": extended_metrics,
        "device": device,
    }
