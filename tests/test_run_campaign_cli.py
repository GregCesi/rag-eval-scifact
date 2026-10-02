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
    """Isole RESULTS_DIR/RESULTS_MD, V1_BUCKETS_PATH et le tracking MLflow dans tmp_path."""
    monkeypatch.setattr(campaign, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(campaign, "RESULTS_MD", tmp_path / "RESULTS.md")
    buckets_path = tmp_path / "v1-buckets.json"
    buckets_path.write_text('{"counts": {}, "queries": {}}', encoding="utf-8")
    monkeypatch.setattr(campaign, "V1_BUCKETS_PATH", buckets_path)
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
    stats = {
        "truncated_pct": 0.0,
        "avg_retrieval_latency_ms": 0.0,
        "indexing_duration_seconds": 0.0,
    }
    return results, qrels, "sha256:fake", stats


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
        sys,
        "argv",
        [
            "run_campaign.py",
            "--multirun",
            "campagne=dev",
            "top_k=10,20",
            "tracing=false",
        ],
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


# ---------------------------------------------------------------------------
# Critère 6 (EXE-89) — tracing=false désactive les traces, métriques inchangées
# ---------------------------------------------------------------------------


def test_tracing_false_writes_no_trace_and_keeps_metrics_identical(
    _isolated_cli, monkeypatch
):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "Titre D1", "text": ""}],
    )

    for tracing_value, run_name in [
        ("true", "with-tracing"),
        ("false", "without-tracing"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_campaign.py",
                "campagne=dev",
                f"run_name={run_name}",
                f"tracing={tracing_value}",
            ],
        )
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        run_campaign_module._cli()

    client = MlflowClient()
    experiment = client.get_experiment_by_name("dev")
    runs = {r.info.run_name: r for r in client.search_runs([experiment.experiment_id])}

    traces_with = mlflow.search_traces(
        run_id=runs["with-tracing"].info.run_id, return_type="list"
    )
    traces_without = mlflow.search_traces(
        run_id=runs["without-tracing"].info.run_id, return_type="list"
    )
    assert len(traces_with) == 1
    assert len(traces_without) == 0
    assert runs["with-tracing"].data.metrics == runs["without-tracing"].data.metrics
