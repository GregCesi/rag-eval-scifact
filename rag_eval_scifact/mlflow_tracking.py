"""Suivi MLflow d'un run de campagne, local et sans serveur distant (EXE-86).

`mlflow.db` n'est pas la source de vérité (.claude/rules/versioning.md) : ce
module journalise dans MLflow un run déjà écrit par `campaign.run_campaign`,
il ne le remplace pas. Tracking SQLite (`mlflow.db` à la racine du dépôt), pas
le backend fichier `mlruns/` (en mode maintenance depuis MLflow 3.x) : aucune
variable d'environnement à poser pour l'obtenir.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.processor.base_mlflow import flush_all_batch_processors

from rag_eval_scifact.retrieve import RetrievalResult
from rag_eval_scifact.run_output import compute_per_query_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"

mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplati une config Hydra résolue en clés pointées pour mlflow.log_params."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        flat_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=f"{flat_key}."))
        else:
            flat[flat_key] = value
    return flat


def _sanitize_metric_name(name: str) -> str:
    """MLflow interdit `@` dans un nom de métrique (recall@1 -> recall_at_1)."""
    return name.replace("@", "_at_")


def log_campaign_run(
    campagne: str,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    json_path: Path,
    extended_metrics: dict[str, float] | None = None,
    device: str | None = None,
) -> str:
    """Journalise un run de campagne déjà écrit sur disque. Retourne le run_id MLflow.

    L'URI de tracking suit la configuration ambiante de mlflow
    (`mlflow.set_tracking_uri` ou `MLFLOW_TRACKING_URI`) ; par défaut, MLflow
    écrit sous `./mlruns`, local et sans serveur.

    `extended_metrics` (EXE-90 : bucket v1, part tronquée, latence, durée
    d'indexation) se journalise en plus des 6 métriques historiques, jamais à
    leur place. `device` (EXE-94 critère 7) est une chaîne : journalisé comme
    paramètre (`mlflow.log_param`), jamais comme métrique.
    """
    mlflow.set_experiment(campagne)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_config(config))
        if device is not None:
            mlflow.log_param("device", device)
        all_metrics = {**metrics, **(extended_metrics or {})}
        mlflow.log_metrics(
            {_sanitize_metric_name(name): value for name, value in all_metrics.items()}
        )
        mlflow.log_artifact(str(json_path))
        return run.info.run_id


def log_query_traces(
    run_id: str,
    results: list[RetrievalResult],
    qrels: dict[str, set[str]],
    titles: dict[str, str],
) -> None:
    """Journalise une trace MLflow par requête, rattachée à `run_id` (EXE-89).

    Une trace par requête, pas une trace à N spans : chaque requête se retrouve
    par sa propre recherche (`mlflow.search_traces(run_id=run_id)`). Les traces
    sont exportées par MLflow de façon asynchrone ; `flush_all_batch_processors`
    les rend visibles en recherche avant la fin de cet appel, sans attendre la
    sortie du process (nécessaire aux tests comme à la lecture immédiate du run_id).
    """
    for r in results:
        relevant_ids = qrels.get(r.query_id, set())
        retrieved_ids = [d["doc_id"] for d in r.retrieved]
        per_query = compute_per_query_metrics(retrieved_ids, relevant_ids)
        with mlflow.start_span(
            name=f"retrieve-{r.query_id}",
            span_type=SpanType.RETRIEVER,
            run_id=run_id,
        ) as span:
            span.set_inputs({"query_id": r.query_id, "query_text": r.query_text})
            span.set_outputs(
                {
                    "retrieved_top10": [
                        {
                            "doc_id": d["doc_id"],
                            "title": titles.get(d["doc_id"], ""),
                            "rank": d["rank"],
                            "score": d["score"],
                        }
                        for d in r.retrieved[:10]
                    ]
                }
            )
            span.set_attributes(
                {
                    "expected_docs": [
                        {"doc_id": doc_id, "title": titles.get(doc_id, "")}
                        for doc_id in sorted(relevant_ids)
                    ],
                    "best_rank_in_top_100": per_query["best_rank"],
                }
            )
    flush_all_batch_processors()
