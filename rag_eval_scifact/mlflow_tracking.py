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
) -> str:
    """Journalise un run de campagne déjà écrit sur disque. Retourne le run_id MLflow.

    L'URI de tracking suit la configuration ambiante de mlflow
    (`mlflow.set_tracking_uri` ou `MLFLOW_TRACKING_URI`) ; par défaut, MLflow
    écrit sous `./mlruns`, local et sans serveur.
    """
    mlflow.set_experiment(campagne)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_config(config))
        mlflow.log_metrics(
            {_sanitize_metric_name(name): value for name, value in metrics.items()}
        )
        mlflow.log_artifact(str(json_path))
        return run.info.run_id
