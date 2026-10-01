"""Tests du suivi MLflow d'un run de campagne (EXE-86, critères 1 à 4).

N'écrivent jamais dans le `mlruns/` du dépôt : chaque test pointe
`mlflow.set_tracking_uri` vers un dossier temporaire, restauré après coup. Ne
chargent aucun modèle d'embedding : l'artefact comparé est fabriqué à la main,
comme dans test_campaign.py.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from rag_eval_scifact import mlflow_tracking
from rag_eval_scifact.mlflow_tracking import log_campaign_run

CONFIG = {
    "campagne": "campagne-test",
    "run_name": "run-1",
    "retriever": {
        "name": "dense",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_seq_length": 256,
        "dim": 384,
    },
    "fusion": None,
    "rerank": None,
    "split": "test",
    "top_k": 100,
}

METRICS = {
    "recall@1": 0.4823,
    "recall@5": 0.7379,
    "recall@10": 0.7833,
    "recall@100": 0.9250,
    "ndcg@10": 0.6451,
    "mrr": 0.6110,
}


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking(tmp_path):
    """Pointe MLflow vers un SQLite temporaire : jamais le mlflow.db du dépôt."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(None)


def _write_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "campagne-test" / "run-1-2026-10-01T12-00-00-000000.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"campagne": "campagne-test"}, ensure_ascii=False).encode(
        "utf-8"
    )
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return path


# ---------------------------------------------------------------------------
# Critère 1 — l'expérience MLflow porte le nom de la campagne
# ---------------------------------------------------------------------------


def test_run_is_filed_under_an_experiment_named_after_the_campaign(tmp_path):
    json_path = _write_artifact(tmp_path)

    log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    client = MlflowClient()
    experiment = client.get_experiment_by_name("campagne-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_name == "run-1"


# ---------------------------------------------------------------------------
# Critère 2 — chaque valeur de la config résolue est un paramètre du run
# ---------------------------------------------------------------------------


def test_run_params_cover_every_resolved_config_value(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    params = MlflowClient().get_run(run_id).data.params
    assert params["campagne"] == "campagne-test"
    assert params["run_name"] == "run-1"
    assert params["retriever.name"] == "dense"
    assert params["retriever.model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert params["retriever.max_seq_length"] == "256"
    assert params["retriever.dim"] == "384"
    assert params["fusion"] == "None"
    assert params["rerank"] == "None"
    assert params["split"] == "test"
    assert params["top_k"] == "100"


# ---------------------------------------------------------------------------
# Critère 3 — les 6 métriques du run MLflow valent celles de RESULTS.md
# ---------------------------------------------------------------------------


def test_run_metrics_equal_the_six_results_md_metrics(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    mlflow_metrics = MlflowClient().get_run(run_id).data.metrics
    assert len(mlflow_metrics) == len(METRICS)
    for name, value in METRICS.items():
        assert mlflow_metrics[name.replace("@", "_at_")] == pytest.approx(value)


# ---------------------------------------------------------------------------
# Critère 4 — l'artefact attaché est identique octet pour octet au fichier results/
# ---------------------------------------------------------------------------


def test_artifact_attached_is_byte_identical_to_results_file(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    client = MlflowClient()
    downloaded = client.download_artifacts(run_id, json_path.name, str(tmp_path / "dl"))
    assert Path(downloaded).read_bytes() == json_path.read_bytes()


# ---------------------------------------------------------------------------
# Critère 5 (EXE-88) — tracking SQLite par défaut, sans variable d'environnement
# ---------------------------------------------------------------------------


def test_default_tracking_uri_is_sqlite_mlflow_db_at_repo_root():
    repo_root = Path(mlflow_tracking.__file__).resolve().parent.parent
    expected = f"sqlite:///{repo_root / 'mlflow.db'}"
    assert mlflow_tracking.DEFAULT_TRACKING_URI == expected


def test_no_mlflow_allow_file_store_env_var_is_set():
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ
