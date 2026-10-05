"""Tests de la boucle commune aux trois juges (EXE-131, critères 1, 2, 3 et 6)."""

from __future__ import annotations

from rag_eval_scifact.judge_errors import JudgeCallError
from rag_eval_scifact.judge_runner import run_judge_pairs


def _pair(pair_id: str) -> dict:
    return {"pair_id": pair_id}


def _fake_judge_pair(pair: dict) -> dict:
    return {"pair_id": pair["pair_id"], "verdict": "SUPPORTS", "duration_seconds": 1.0}


# ---------------------------------------------------------------------------
# Critère 3 — résumé des paires déjà jugées en une seule ligne
# ---------------------------------------------------------------------------


def test_skip_summary_is_a_single_line_for_several_skipped_pairs(capsys):
    pairs = [_pair(f"1:d{i}") for i in range(1, 26)]
    already_judged_ids = {f"1:d{i}" for i in range(1, 26)}

    judgments = run_judge_pairs(pairs, _fake_judge_pair, already_judged_ids)

    assert judgments == []
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "déjà jugée" in line]
    assert lines == ["25 paires déjà jugées, non rejugées"]


def test_no_skip_summary_when_nothing_was_already_judged(capsys):
    run_judge_pairs([_pair("1:d1")], _fake_judge_pair, set())

    out = capsys.readouterr().out
    assert "déjà jugée" not in out


# ---------------------------------------------------------------------------
# Critère 1 — progression toutes les 10 paires
# ---------------------------------------------------------------------------


def test_progress_line_every_ten_pairs(capsys):
    pairs = [_pair(f"1:d{i}") for i in range(1, 26)]

    run_judge_pairs(pairs, _fake_judge_pair, set())

    out = capsys.readouterr().out
    progress_lines = [line for line in out.splitlines() if "traité" in line]
    assert progress_lines == [
        "10/25 traité, durée moyenne par paire : 1.00 s",
        "20/25 traité, durée moyenne par paire : 1.00 s",
    ]


def test_no_progress_line_under_ten_pairs(capsys):
    run_judge_pairs([_pair("1:d1")], _fake_judge_pair, set())

    out = capsys.readouterr().out
    assert "traité" not in out


# ---------------------------------------------------------------------------
# Critère 2 — chaque jugement est livré dès qu'il est produit
# ---------------------------------------------------------------------------


def test_on_judgment_is_called_immediately_for_each_judgment():
    pairs = [_pair("1:d1"), _pair("1:d2"), _pair("1:d3")]
    delivered = []

    run_judge_pairs(pairs, _fake_judge_pair, set(), on_judgment=delivered.append)

    assert [j["pair_id"] for j in delivered] == ["1:d1", "1:d2", "1:d3"]


# ---------------------------------------------------------------------------
# Critère 6 — arrêt propre sur JudgeCallError, aucune trace Python
# ---------------------------------------------------------------------------


def test_stops_cleanly_on_judge_call_error_keeping_prior_judgments(capsys):
    pairs = [_pair("1:d1"), _pair("1:d2"), _pair("1:d3")]

    def judge_pair_fn(pair):
        if pair["pair_id"] == "1:d2":
            raise JudgeCallError(
                "code de sortie 1", stdout="sortie du modèle", stderr="erreur"
            )
        return _fake_judge_pair(pair)

    judgments = run_judge_pairs(pairs, judge_pair_fn, set())

    assert [j["pair_id"] for j in judgments] == ["1:d1"]
    out = capsys.readouterr().out
    assert "1:d2 : code de sortie 1" in out
    assert "sortie du modèle" in out
    assert "erreur" in out
    assert "1 jugement(s) gardé(s)" in out


def test_judge_call_error_does_not_propagate_past_the_runner():
    pairs = [_pair("1:d1")]

    def judge_pair_fn(pair):
        raise JudgeCallError("Ollama ne répond pas")

    # Ne lève pas : la fonction rend la liste des jugements déjà produits.
    judgments = run_judge_pairs(pairs, judge_pair_fn, set())
    assert judgments == []


# ---------------------------------------------------------------------------
# Critère 11 (porté ici au niveau de la boucle) — respect de `limit`
# ---------------------------------------------------------------------------


def test_limit_bounds_the_number_of_pairs_judged():
    pairs = [_pair("1:d1"), _pair("1:d2"), _pair("1:d3")]

    judgments = run_judge_pairs(pairs, _fake_judge_pair, set(), limit=2)

    assert [j["pair_id"] for j in judgments] == ["1:d1", "1:d2"]
