"""Tests du lanceur Hydra standard de `run_campaign` (EXE-88, critères 2 et 3).

`--help` est un vrai sous-processus : Hydra affiche l'aide et quitte sans
jamais appeler `main()`, donc sans charger de modèle. Le `--multirun` est
testé en process, avec `retrieve_campaign` et `get_token_counts` remplacés par
des fonctions fabriquées : aucun modèle n'est chargé, rien n'est écrit dans le
`mlflow.db` ni le `RESULTS.md` réels.
"""

from __future__ import annotations

import subprocess
import sys

import mlflow
import pytest
from hydra.core.global_hydra import GlobalHydra
from mlflow.tracking import MlflowClient

import rag_eval_scifact.run_campaign as run_campaign_module
from rag_eval_scifact import campaign
from rag_eval_scifact.retrieve import RetrievalResult


def test_help_shows_hydra_help_with_overridable_config():
    result = subprocess.run(
        [sys.executable, "-m", "rag_eval_scifact.run_campaign", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "powered by Hydra" in result.stdout
    assert "top_k: 100" in result.stdout
    assert "cache_dir:" in result.stdout


@pytest.fixture
def _isolated_cli(tmp_path, monkeypatch):
    """Isole RESULTS_DIR/RESULTS_MD et le tracking MLflow dans tmp_path."""
    monkeypatch.setattr(campaign, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(campaign, "RESULTS_MD", tmp_path / "RESULTS.md")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def _fake_retrieve_campaign(**kwargs):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une question",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.9}],
        )
    ]
    qrels = {"q1": {"d1"}}
    return results, qrels, "sha256:fake"


def test_multirun_produces_two_runs_two_files_two_mlflow_runs(
    _isolated_cli, monkeypatch
):
    calls: list[int] = []

    def tracking_retrieve_campaign(*, top_k, **kwargs):
        calls.append(top_k)
        return _fake_retrieve_campaign(top_k=top_k, **kwargs)

    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", tracking_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        sys, "argv", ["run_campaign.py", "--multirun", "campagne=dev", "top_k=10,20"]
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert sorted(calls) == [10, 20]

    files = sorted((_isolated_cli / "results" / "dev").glob("*.json.gz"))
    assert len(files) == 2

    client = MlflowClient()
    experiment = client.get_experiment_by_name("dev")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
