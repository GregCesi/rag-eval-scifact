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
        lambda campagne: [
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
    monkeypatch.setattr(
        run_grid_module, "_prediction_committed", lambda campagne: False
    )
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
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [{"run_name": "combo-a", "overrides": []}],
    )
    monkeypatch.setattr(
        run_grid_module, "_prediction_committed", lambda campagne: False
    )

    run_grid_module.main(["--campagne", "dev"])

    files = list((_isolated_grid / "results" / "dev").glob("*.json.gz"))
    assert len(files) >= 1


# ---------------------------------------------------------------------------
# Critère 5 — une combinaison déjà résultante dans la campagne est sautée
# ---------------------------------------------------------------------------


def test_launch_skips_combo_with_existing_result_and_says_so(
    _isolated_grid, monkeypatch, capsys
):
    combos = [
        {"run_name": "combo-a", "overrides": []},
        {"run_name": "combo-b", "overrides": ["top_k=10"]},
    ]
    monkeypatch.setattr(run_grid_module, "load_grid_combos", lambda campagne: combos)
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


# ---------------------------------------------------------------------------
# EXE-140 critère 1 — mode liste d'une autre campagne que v2-grid
# ---------------------------------------------------------------------------


def test_list_mode_for_v4_leviers_shows_its_four_runs_and_count(capsys):
    run_grid_module.main(["--list", "--campagne", "v4-leviers"])

    out = capsys.readouterr().out
    for name in [
        "qwen3-passages-reference",
        "qwen3-passages-sans-instruction",
        "qwen3-4b-passages",
        "medcpt-passages",
    ]:
        assert name in out
    assert "4 combinaison(s)" in out


def test_list_mode_without_naming_a_campagne_still_shows_v2_grid(capsys):
    run_grid_module.main(["--list"])

    out = capsys.readouterr().out
    assert "34 combinaison(s)" in out


# ---------------------------------------------------------------------------
# EXE-140 critère 3 — prédiction exigée pour toute campagne sauf « dev »
# ---------------------------------------------------------------------------


def test_launch_refuses_any_non_dev_campaign_without_committed_prediction(
    capsys, monkeypatch
):
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [{"run_name": "x", "overrides": []}],
    )
    monkeypatch.setattr(
        run_grid_module, "_prediction_committed", lambda campagne: False
    )
    calls = []
    monkeypatch.setattr(run_campaign_module, "main", lambda cfg: calls.append(cfg))

    with pytest.raises(SystemExit):
        run_grid_module.main(["--campagne", "v4-leviers"])

    out = capsys.readouterr().out
    assert "results/v4-leviers/PREDICTION.md" in out
    assert calls == []


# ---------------------------------------------------------------------------
# EXE-140 critère 4 — fichier de grille absent : une phrase, pas de trace
# ---------------------------------------------------------------------------


def test_list_mode_missing_grid_file_prints_a_sentence_and_exits_cleanly(capsys):
    with pytest.raises(SystemExit):
        run_grid_module.main(["--list", "--campagne", "campagne-sans-grille"])

    out = capsys.readouterr().out
    assert "conf/grid/campagne-sans-grille.yaml" in out


def test_launch_missing_grid_file_prints_a_sentence_and_exits_cleanly(capsys):
    with pytest.raises(SystemExit):
        run_grid_module.main(["--campagne", "campagne-sans-grille"])

    out = capsys.readouterr().out
    assert "conf/grid/campagne-sans-grille.yaml" in out


# ---------------------------------------------------------------------------
# EXE-140 critère 5 — lancer un seul run par son nom
# ---------------------------------------------------------------------------


def test_launch_with_run_option_launches_only_that_run(_isolated_grid, monkeypatch):
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [
            {"run_name": "combo-a", "overrides": []},
            {"run_name": "combo-b", "overrides": ["top_k=10"]},
        ],
    )

    run_grid_module.main(["--campagne", "dev", "--run", "combo-b"])

    campaign_dir = _isolated_grid / "results" / "dev"
    assert list(campaign_dir.glob("combo-a-*.json.gz")) == []
    assert len(list(campaign_dir.glob("combo-b-*.json.gz"))) == 1


def test_launch_with_unknown_run_name_lists_known_names(capsys, monkeypatch):
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [{"run_name": "combo-a", "overrides": []}],
    )

    with pytest.raises(SystemExit):
        run_grid_module.main(["--campagne", "dev", "--run", "inconnu"])

    out = capsys.readouterr().out
    assert "inconnu" in out
    assert "combo-a" in out


def test_essai_with_unknown_run_name_lists_known_names(capsys, monkeypatch):
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [{"run_name": "combo-a", "overrides": []}],
    )

    with pytest.raises(SystemExit):
        run_grid_module.main(["--campagne", "dev", "--essai", "inconnu"])

    out = capsys.readouterr().out
    assert "inconnu" in out
    assert "combo-a" in out


# ---------------------------------------------------------------------------
# EXE-140 critère 12 — essai : encodage d'un échantillon, rien n'est écrit
# ---------------------------------------------------------------------------


def test_essai_mode_prints_sample_stats_and_writes_no_campaign_file(
    capsys, monkeypatch, tmp_path
):
    captured = {}

    def _fake_essai_embedding(**kwargs):
        captured.update(kwargs)
        return {"n_units": 42.0, "duration_seconds": 1.5, "extrapolated_minutes": 3.25}

    monkeypatch.setattr(run_grid_module, "essai_embedding", _fake_essai_embedding)
    monkeypatch.setattr(
        run_grid_module,
        "load_grid_combos",
        lambda campagne: [{"run_name": "combo-a", "overrides": []}],
    )
    results_dir = tmp_path / "results"
    monkeypatch.setattr(campaign, "RESULTS_DIR", results_dir)

    run_grid_module.main(["--campagne", "dev", "--essai", "combo-a"])

    out = capsys.readouterr().out
    assert "42" in out
    assert "1.5" in out
    assert "3.2" in out or "3.3" in out
    assert not results_dir.exists() or list(results_dir.rglob("*.json*")) == []
    assert captured["unit"] in {"document", "passages"}
