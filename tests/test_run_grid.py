"""Tests du CLI de grille v2-grid (EXE-91, critères 2, 3, 4, 5).

Mode liste : ne charge aucun modèle, ne lance rien (vérifié en empêchant tout
appel à `run_campaign.main`). Mode lancement : `retrieve_campaign`,
`get_token_counts` et `load_corpus` sont remplacés par des fonctions fabriquées,
comme dans test_run_campaign_cli.py — aucun modèle d'embedding n'est chargé.
"""

from __future__ import annotations

import mlflow
import pytest
from hydra.core.global_hydra import GlobalHydra

import rag_eval_scifact.run_campaign as run_campaign_module
import rag_eval_scifact.run_grid as run_grid_module
from rag_eval_scifact import campaign
from rag_eval_scifact.retrieve import RetrievalResult


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
        "device": "cpu",
    }
    return results, qrels, "sha256:fake", stats


@pytest.fixture
def _isolated_grid(tmp_path, monkeypatch):
    """Isole RESULTS_DIR/RESULTS_MD, V1_BUCKETS_PATH, MLflow ; aucun modèle chargé."""
    monkeypatch.setattr(campaign, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(campaign, "RESULTS_MD", tmp_path / "RESULTS.md")
    buckets_path = tmp_path / "v1-buckets.json"
    buckets_path.write_text('{"counts": {}, "queries": {}}', encoding="utf-8")
    monkeypatch.setattr(campaign, "V1_BUCKETS_PATH", buckets_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(run_campaign_module, "load_corpus", lambda path: [])
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    yield tmp_path
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# Critère 2 — mode liste : affiche, compte, ne lance rien
# ---------------------------------------------------------------------------


def test_list_mode_prints_run_names_and_count_without_launching(capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(run_campaign_module, "main", lambda cfg: calls.append(cfg))

    run_grid_module.main(["--list"])

    out = capsys.readouterr().out
    combos = run_grid_module.load_grid_combos()
    for combo in combos:
        assert combo["run_name"] in out
    assert f"{len(combos)} combinaison" in out
    assert calls == []


# ---------------------------------------------------------------------------
# Critère 3 — sans le mode liste, chaque combinaison est lancée comme un run
# ---------------------------------------------------------------------------


def test_launch_mode_runs_every_declared_combo_as_a_campaign_run(
    _isolated_grid, monkeypatch
):
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda: [
            {"run_name": "combo-a", "overrides": []},
            {"run_name": "combo-b", "overrides": ["top_k=10"]},
        ],
    )

    run_grid_module.main(["--campagne", "dev"])

    campaign_dir = _isolated_grid / "results" / "dev"
    assert len(list(campaign_dir.glob("combo-a-*.json.gz"))) == 1
    assert len(list(campaign_dir.glob("combo-b-*.json.gz"))) == 1


# ---------------------------------------------------------------------------
# Critère 4 — refus de lancer v2-grid sans PREDICTION.md committé
# ---------------------------------------------------------------------------


def test_launch_refuses_v2_grid_without_committed_prediction(capsys, monkeypatch):
    monkeypatch.setattr(run_grid_module, "_prediction_committed", lambda: False)
    calls = []
    monkeypatch.setattr(run_campaign_module, "main", lambda cfg: calls.append(cfg))

    with pytest.raises(SystemExit):
        run_grid_module.main(["--campagne", "v2-grid"])

    out = capsys.readouterr().out
    assert "PREDICTION.md" in out
    assert calls == []


def test_launch_does_not_check_prediction_for_other_campaigns(
    _isolated_grid, monkeypatch
):
    monkeypatch.setattr(run_grid_module, "_prediction_committed", lambda: False)

    run_grid_module.main(["--campagne", "dev"])

    files = list((_isolated_grid / "results" / "dev").glob("*.json.gz"))
    assert len(files) >= 1


# ---------------------------------------------------------------------------
# Critère 5 — une combinaison déjà résultante dans la campagne est sautée
# ---------------------------------------------------------------------------


def test_launch_skips_combo_with_existing_result_and_says_so(
    _isolated_grid, monkeypatch, capsys
):
    combos = run_grid_module.load_grid_combos()
    run_name = combos[0]["run_name"]
    campaign_dir = _isolated_grid / "results" / "dev"
    campaign_dir.mkdir(parents=True)
    for combo in combos:
        (
            campaign_dir / f"{combo['run_name']}-2026-10-02T00-00-00-000000.json.gz"
        ).touch()

    calls = []
    monkeypatch.setattr(run_campaign_module, "main", lambda cfg: calls.append(cfg))

    run_grid_module.main(["--campagne", "dev"])

    out = capsys.readouterr().out
    assert f"{run_name} : déjà fait, ignoré" in out
    assert calls == []
