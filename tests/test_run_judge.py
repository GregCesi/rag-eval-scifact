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
    return json.dumps(
        {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "ok"}
    )


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
    assert "2 paires déjà jugées, non rejugées" in out
    judgments_path = _isolated_results / "dev" / "jugements-local.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert {j["pair_id"] for j in judgments} == {"1:d1", "1:d2"}


# ---------------------------------------------------------------------------
# EXE-131, critère 2 — chaque jugement est écrit dès qu'il est produit, pas
# seulement à la fin du run
# ---------------------------------------------------------------------------


def test_judgments_are_written_incrementally_as_they_are_produced(
    monkeypatch, _isolated_results
):
    pairs = [
        {**PAIR, "pair_id": "1:d1", "doc_id": "d1"},
        {**PAIR, "pair_id": "1:d2", "doc_id": "d2"},
    ]
    _write_pairs(_isolated_results, "dev", pairs)
    seen_on_disk = []

    judgments_path = _isolated_results / "dev" / "jugements-local.json"

    def _fake_judge_pairs(
        pairs,
        call_fn,
        already_judged_ids,
        model,
        limit=None,
        on_judgment=lambda j: None,
    ):
        for pair in pairs:
            judgment = {
                "pair_id": pair["pair_id"],
                "claim_id": pair["claim_id"],
                "doc_id": pair["doc_id"],
                "model": model,
                "duration_seconds": 0.01,
                "verdict": "SUPPORTS",
            }
            on_judgment(judgment)
            seen_on_disk.append(json.loads(judgments_path.read_text(encoding="utf-8")))
        return []

    monkeypatch.setattr(
        run_judge_module,
        "JUDGES",
        {
            "local": {
                "call_fn": _fake_call_fn,
                "judge_pairs": _fake_judge_pairs,
                "default_model": "modele-local-fabrique",
            }
        },
    )

    run_judge_module.main(["--campagne", "dev"])

    assert [len(snapshot) for snapshot in seen_on_disk] == [1, 2]


# ---------------------------------------------------------------------------
# EXE-131, critère 6 — le juge s'arrête sans trace Python, le MLflow run porte
# le nombre réellement gardé
# ---------------------------------------------------------------------------


def test_judge_call_error_stops_cleanly_and_mlflow_run_keeps_partial_count(
    monkeypatch, _isolated_results, capsys
):
    from rag_eval_scifact.judge_errors import JudgeCallError

    pairs = [
        {**PAIR, "pair_id": "1:d1", "doc_id": "d1"},
        {**PAIR, "pair_id": "1:d2", "doc_id": "d2"},
    ]
    _write_pairs(_isolated_results, "dev", pairs)

    def _fake_judge_pairs(
        pairs,
        call_fn,
        already_judged_ids,
        model,
        limit=None,
        on_judgment=lambda j: None,
    ):
        on_judgment(
            {
                "pair_id": "1:d1",
                "claim_id": "1",
                "doc_id": "d1",
                "model": model,
                "duration_seconds": 0.01,
                "verdict": "SUPPORTS",
            }
        )
        raise JudgeCallError("code de sortie 1", stdout="out", stderr="err")

    monkeypatch.setattr(
        run_judge_module,
        "JUDGES",
        {
            "local": {
                "call_fn": _fake_call_fn,
                "judge_pairs": _fake_judge_pairs,
                "default_model": "modele-local-fabrique",
            }
        },
    )

    with pytest.raises(JudgeCallError):
        run_judge_module.main(["--campagne", "dev"])

    judgments_path = _isolated_results / "dev" / "jugements-local.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert [j["pair_id"] for j in judgments] == ["1:d1"]

    run = mlflow.search_runs(experiment_names=["dev"], output_format="list")[0]
    assert run.data.metrics["n_judgments"] == 1


# ---------------------------------------------------------------------------
# EXE-131, critère 10 — un jugement « citation introuvable » (ancienne forme)
# est rejugé à la relance
# ---------------------------------------------------------------------------


def test_old_format_citation_introuvable_judgment_is_rejudged(
    monkeypatch, _isolated_results, capsys
):
    pairs = [{**PAIR, "pair_id": "1:d1", "doc_id": "d1"}]
    _write_pairs(_isolated_results, "dev", pairs)
    judgments_path = _isolated_results / "dev" / "jugements-local.json"
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    judgments_path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "1:d1",
                    "claim_id": "1",
                    "doc_id": "d1",
                    "verdict": "citation introuvable",
                    "level": "DIRECT",
                    "evidence": "x",
                    "reason": "y",
                    "duration_seconds": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    run_judge_module.main(["--campagne", "dev"])

    out = capsys.readouterr().out
    assert "déjà jugée" not in out
    new_judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert new_judgments[0]["verdict"] == "SUPPORTS"


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
            json.dumps(
                {
                    "verdict": "SUPPORTS",
                    "level": "DIRECT",
                    "evidence": "",
                    "reason": "ok",
                }
            ),
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


# ---------------------------------------------------------------------------
# EXE-121, critère 1 — un troisième juge, « etapes ».
# EXE-121, critère 10 — sa trace MLflow montre une étape par nœud appelé.
# ---------------------------------------------------------------------------


def test_etapes_judge_is_registered_by_default():
    assert "etapes" in run_judge_module.JUDGES


def test_etapes_judgment_trace_has_one_child_span_per_step_actually_called(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(run_judge_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_judge_module, "_prediction_committed", lambda: False)

    def _fake_judge_pairs(
        pairs,
        call_fn,
        already_judged_ids,
        model,
        limit=None,
        on_judgment=lambda j: None,
    ):
        return [
            {
                "pair_id": "1:d1",
                "claim_id": "1",
                "doc_id": "d1",
                "model": model,
                "duration_seconds": 0.01,
                "verdict": "SUPPORTS",
                "evidence": "phrase",
                "cause": "",
                "discarded_sentences": 0,
                "model_calls": 3,
                "steps": {
                    "claim": {
                        "input": {"claim_text": "x"},
                        "output": {"topic": "t", "effect": "e", "direction": "d"},
                    },
                    "document": {
                        "input": {"doc_text": "y"},
                        "output": {"sentences": ["phrase"], "discarded": 0},
                    },
                    "verdict": {
                        "input": {"sentences": ["phrase"]},
                        "output": {
                            "verdict": "SUPPORTS",
                            "decisive_sentence": "phrase",
                        },
                    },
                },
            }
        ]

    monkeypatch.setattr(
        run_judge_module,
        "JUDGES",
        {
            "etapes": {
                "call_fn": lambda *a, **k: "",
                "judge_pairs": _fake_judge_pairs,
                "default_model": "modele-etapes-fabrique",
            }
        },
    )
    _write_pairs(tmp_path, "dev", [PAIR])

    run_judge_module.main(["--campagne", "dev", "--juge", "etapes"])

    run = mlflow.search_runs(experiment_names=["dev"], output_format="list")[0]
    trace = mlflow.search_traces(run_id=run.info.run_id, return_type="list")[0]
    span_names = {s.name for s in trace.data.spans}
    assert span_names == {"judge-1:d1", "claim-1:d1", "document-1:d1", "verdict-1:d1"}

    document_span = next(s for s in trace.data.spans if s.name == "document-1:d1")
    assert document_span.inputs == {"doc_text": "y"}
    assert document_span.outputs == {"sentences": ["phrase"], "discarded": 0}
