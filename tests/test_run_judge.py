"""Tests du CLI de jugement (EXE-119 critère 10 ; EXE-120 critères 1 et 5).
Aucun modèle n'est appelé : `run_judge.JUDGES` est remplacé par des juges fabriqués.
"""

from __future__ import annotations

import json

import mlflow
import pytest

import rag_eval_scifact.run_judge as run_judge_module
from rag_eval_scifact.judge_local import judge_pairs as judge_pairs_local


@pytest.fixture(autouse=True)
def _isolate_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(None)


def _fake_call_fn(model, system, user, seed):
    return json.dumps({"verdict": "SUPPORTS", "evidence": "", "reason": "ok"})


@pytest.fixture
def _isolated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run_judge_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        run_judge_module,
        "JUDGES",
        {
            "local": {
                "call_fn": _fake_call_fn,
                "judge_pairs": judge_pairs_local,
                "default_model": "modele-local-fabrique",
            }
        },
    )
    return tmp_path


def _write_pairs(results_dir, campagne, pairs):
    path = results_dir / campagne / "paires.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return path


PAIR = {
    "pair_id": "1:d1",
    "claim_id": "1",
    "claim_text": "claim",
    "doc_id": "d1",
    "doc_title": "titre",
    "doc_text": "texte",
    "rank": 1,
    "document_attendu": True,
}


# ---------------------------------------------------------------------------
# Critère 10 — refus de juger v3-juge sans PREDICTION.md committé
# ---------------------------------------------------------------------------


def test_refuses_v3_juge_without_committed_prediction(
    capsys, monkeypatch, _isolated_results
):
    monkeypatch.setattr(run_judge_module, "_prediction_committed", lambda: False)
    _write_pairs(_isolated_results, "v3-juge", [PAIR])

    with pytest.raises(SystemExit):
        run_judge_module.main(["--campagne", "v3-juge"])

    out = capsys.readouterr().out
    assert "PREDICTION.md" in out
    assert not (_isolated_results / "v3-juge" / "jugements-local.json").exists()


def test_does_not_check_prediction_for_other_campaigns(monkeypatch, _isolated_results):
    monkeypatch.setattr(run_judge_module, "_prediction_committed", lambda: False)
    _write_pairs(_isolated_results, "dev", [PAIR])

    run_judge_module.main(["--campagne", "dev"])

    judgments_path = _isolated_results / "dev" / "jugements-local.json"
    assert judgments_path.exists()
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert [j["pair_id"] for j in judgments] == ["1:d1"]


def test_rerun_does_not_rejudge_existing_pairs(monkeypatch, _isolated_results, capsys):
    monkeypatch.setattr(run_judge_module, "_prediction_committed", lambda: False)
    pairs = [PAIR, {**PAIR, "pair_id": "1:d2", "doc_id": "d2"}]
    _write_pairs(_isolated_results, "dev", pairs)

    run_judge_module.main(["--campagne", "dev"])
    capsys.readouterr()
    run_judge_module.main(["--campagne", "dev"])

    out = capsys.readouterr().out
    assert "1:d1" in out
    assert "déjà jugée" in out
    judgments_path = _isolated_results / "dev" / "jugements-local.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert {j["pair_id"] for j in judgments} == {"1:d1", "1:d2"}


# ---------------------------------------------------------------------------
# EXE-120, critères 1 et 5 — un second juge, « claude », avec son propre
# modèle par défaut et sa propre fonction de jugement.
# ---------------------------------------------------------------------------


def test_claude_judge_is_dispatched_to_its_own_call_fn_and_default_model(
    monkeypatch, tmp_path
):
    from rag_eval_scifact.judge_claude import judge_pairs as judge_pairs_claude

    monkeypatch.setattr(run_judge_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_judge_module, "_prediction_committed", lambda: False)

    requested_models = []

    def _fake_claude_call_fn(model, system, user):
        requested_models.append(model)
        return (
            json.dumps({"verdict": "SUPPORTS", "evidence": "", "reason": "ok"}),
            "claude-sonnet-5",
            42,
        )

    monkeypatch.setattr(
        run_judge_module,
        "JUDGES",
        {
            "claude": {
                "call_fn": _fake_claude_call_fn,
                "judge_pairs": judge_pairs_claude,
                "default_model": "modele-claude-fabrique",
            }
        },
    )
    _write_pairs(tmp_path, "dev", [PAIR])

    run_judge_module.main(["--campagne", "dev", "--juge", "claude"])

    assert requested_models == ["modele-claude-fabrique"]
    judgments_path = tmp_path / "dev" / "jugements-claude.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert judgments[0]["model"] == "claude-sonnet-5"
    assert judgments[0]["input_tokens"] == 42
