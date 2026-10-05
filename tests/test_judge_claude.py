"""Tests du juge Claude (EXE-120, critères 3, 4 et 6). Aucun appel à Claude Code :
`call_fn` est toujours une fonction fabriquée.
"""

from __future__ import annotations

import json

import pytest

from rag_eval_scifact.judge_claude import (
    call_claude_code,
    citation_found,
    judge_pair,
    judge_pairs,
)
from rag_eval_scifact.judge_errors import ClaudeRefusal, JudgeCallError
from rag_eval_scifact.judge_prompt import SYSTEM_PROMPT, build_user_prompt


def _pair(**overrides):
    base = {
        "pair_id": "1:d1",
        "claim_id": "1",
        "claim_text": "0-dimensional biomaterials show inductive properties.",
        "doc_id": "d1",
        "doc_title": "Titre.",
        "doc_text": "Les biomatériaux 0D ont des propriétés inductives démontrées.",
        "rank": 42,
        "document_attendu": True,
        "famille": ["attendu"],
        "etiquette_origine": "SUPPORT",
    }
    base.update(overrides)
    return base


def _fake_call(
    response: str, model_reported: str = "claude-sonnet-5", input_tokens: int = 7
):
    calls = []

    def call_fn(model, system, user):
        calls.append({"model": model, "system": system, "user": user})
        return response, model_reported, input_tokens

    call_fn.calls = calls
    return call_fn


# ---------------------------------------------------------------------------
# Critère 3 — prompt système et utilisateur octet pour octet ceux du juge local
# ---------------------------------------------------------------------------


def test_sends_the_same_system_and_user_prompt_as_the_local_judge():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "ok"}
        )
    )

    judge_pair(pair, call_fn)

    sent = call_fn.calls[0]
    assert sent["system"] == SYSTEM_PROMPT
    assert sent["user"] == build_user_prompt(
        pair["claim_text"], pair["doc_title"], pair["doc_text"]
    )


# ---------------------------------------------------------------------------
# Critère 6 — règles communes : ni document_attendu, ni rang, transmis au modèle
# ---------------------------------------------------------------------------


def test_judge_only_sends_claim_and_document_text_to_the_model():
    pair = _pair(
        rank=42,
        document_attendu=True,
        famille=["attendu", "devant"],
        etiquette_origine="SUPPORT",
    )
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "ok"}
        )
    )

    judge_pair(pair, call_fn)

    sent_user_prompt = call_fn.calls[0]["user"]
    assert "42" not in sent_user_prompt
    assert "document_attendu" not in sent_user_prompt
    assert "devant" not in sent_user_prompt


# ---------------------------------------------------------------------------
# Critère 6 — citation introuvable
# ---------------------------------------------------------------------------


def test_citation_found_true_for_empty_evidence():
    assert citation_found("", "peu importe le texte") is True


def test_verdict_marked_citation_introuvable_when_evidence_not_in_doc_text():
    # EXE-131, critère 7 : le verdict rendu par le modèle est gardé.
    pair = _pair(doc_text="Un texte qui ne contient pas la citation inventée.")
    call_fn = _fake_call(
        json.dumps(
            {
                "verdict": "SUPPORTS",
                "level": "DIRECT",
                "evidence": "phrase qui n'existe pas",
                "reason": "x",
            }
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "SUPPORTS"
    assert judgment["citation_introuvable"] is True


def test_verdict_kept_when_evidence_is_verbatim_in_doc_text():
    pair = _pair(doc_text="Cette phrase précise est bien dans le document.")
    call_fn = _fake_call(
        json.dumps(
            {
                "verdict": "SUPPORTS",
                "level": "DIRECT",
                "evidence": "Cette phrase précise est bien dans le document.",
                "reason": "x",
            }
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "SUPPORTS"
    assert judgment["citation_introuvable"] is False


# ---------------------------------------------------------------------------
# Critère 6 — illisible après 2 nouvelles tentatives
# ---------------------------------------------------------------------------


def test_verdict_illisible_after_two_retries_of_unparseable_response():
    pair = _pair()
    call_fn = _fake_call("ceci n'est pas du JSON")

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"
    assert len(call_fn.calls) == 3  # 1 essai + 2 nouvelles tentatives


def test_verdict_recovers_if_a_retry_eventually_parses():
    pair = _pair(doc_text="Preuve décisive citée mot pour mot.")
    responses = iter(
        [
            ("pas du json", "claude-sonnet-5", 1),
            ("toujours pas", "claude-sonnet-5", 1),
            (
                json.dumps(
                    {
                        "verdict": "REFUTES",
                        "level": "DIRECT",
                        "evidence": "Preuve décisive citée mot pour mot.",
                        "reason": "x",
                    }
                ),
                "claude-sonnet-5",
                1,
            ),
        ]
    )
    calls = []

    def call_fn(model, system, user):
        calls.append(1)
        return next(responses)

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "REFUTES"
    assert len(calls) == 3


def test_verdict_illisible_when_verdict_field_outside_enum():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps({"verdict": "MAYBE", "level": "NONE", "evidence": "", "reason": "x"})
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"


# ---------------------------------------------------------------------------
# EXE-125, critère 6 — niveau de lecture incohérent avec le verdict
# ---------------------------------------------------------------------------


def test_verdict_illisible_when_supports_has_level_none():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "NONE", "evidence": "", "reason": "x"}
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"


def test_verdict_illisible_when_not_enough_info_has_a_non_none_level():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps(
            {
                "verdict": "NOT_ENOUGH_INFO",
                "level": "VOCABULARY",
                "evidence": "",
                "reason": "x",
            }
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"


# ---------------------------------------------------------------------------
# Critère 4 — le jugement enregistre le modèle rapporté et les tokens d'entrée
# tels que la sortie de la commande les rapporte, pas le modèle demandé
# ---------------------------------------------------------------------------


def test_judgment_records_reported_model_and_input_tokens_from_call_fn():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "ok"}
        ),
        model_reported="claude-sonnet-5",
        input_tokens=856,
    )

    judgment = judge_pair(pair, call_fn, model="sonnet")

    assert judgment["model"] == "claude-sonnet-5"
    assert judgment["input_tokens"] == 856
    assert call_fn.calls[0]["model"] == "sonnet"


# ---------------------------------------------------------------------------
# Critère 6 — reprise sans rejuger, dite sur la console
# ---------------------------------------------------------------------------


def test_judge_pairs_skips_already_judged_pairs_and_says_so(capsys):
    # EXE-131, critère 3 : une seule ligne de résumé, pas une par paire sautée.
    pairs = [_pair(pair_id="1:d1", doc_id="d1"), _pair(pair_id="1:d2", doc_id="d2")]
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "x"}
        )
    )

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"})

    assert [j["pair_id"] for j in judgments] == ["1:d2"]
    assert len(call_fn.calls) == 1
    out = capsys.readouterr().out
    assert "1 paire déjà jugée, non rejugée" in out
    assert "1:d1" not in out


def test_judge_pairs_respects_limit_on_new_judgments_only():
    pairs = [
        _pair(pair_id="1:d1", doc_id="d1"),
        _pair(pair_id="1:d2", doc_id="d2"),
        _pair(pair_id="1:d3", doc_id="d3"),
    ]
    call_fn = _fake_call(
        json.dumps(
            {"verdict": "SUPPORTS", "level": "DIRECT", "evidence": "", "reason": "x"}
        )
    )

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"}, limit=1)

    assert [j["pair_id"] for j in judgments] == ["1:d2"]


# ---------------------------------------------------------------------------
# EXE-131, critères 4 et 5 — refus du juge : enregistré tel quel, pas rejoué
# ---------------------------------------------------------------------------


def test_refusal_is_recorded_as_refus_du_juge_without_retry(capsys):
    pair = _pair()
    calls = []

    def call_fn(model, system, user):
        calls.append(1)
        raise ClaudeRefusal(
            "API Error: Sonnet 5 can't help with this. Start a new session to continue."
        )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "refus du juge"
    assert "Sonnet 5 can't help with this" in judgment["reason"]
    assert len(calls) == 1  # aucune reformulation, aucun second essai
    out = capsys.readouterr().out
    assert pair["pair_id"] in out
    assert "refus du juge" in out


def test_refused_pair_is_not_judged_again(capsys):
    pairs = [_pair(pair_id="1:d1", doc_id="d1")]

    def call_fn(model, system, user):
        raise AssertionError("une paire déjà refusée ne doit pas être rejugée")

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"})

    assert judgments == []


# ---------------------------------------------------------------------------
# EXE-131, critère 6 — l'appel Claude Code qui échoue devient une erreur
# propre, jamais une trace Python brute
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_call_claude_code_raises_claude_refusal_on_refusal_stop_reason(monkeypatch):
    import rag_eval_scifact.judge_claude as judge_claude_module

    payload = json.dumps(
        {
            "stop_reason": "refusal",
            "is_error": True,
            "terminal_reason": "api_error",
            "result": "API Error: Sonnet 5 can't help with this.",
            "usage": {},
            "modelUsage": {},
        }
    )
    monkeypatch.setattr(
        judge_claude_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(payload, returncode=1),
    )

    with pytest.raises(ClaudeRefusal):
        call_claude_code("sonnet", "system", "user")


def test_call_claude_code_raises_judge_call_error_on_non_zero_exit_without_refusal(
    monkeypatch,
):
    import rag_eval_scifact.judge_claude as judge_claude_module

    monkeypatch.setattr(
        judge_claude_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(
            "", stderr="session limit reached", returncode=1
        ),
    )

    with pytest.raises(JudgeCallError):
        call_claude_code("sonnet", "system", "user")


def test_call_claude_code_raises_judge_call_error_on_unreadable_stdout(monkeypatch):
    import rag_eval_scifact.judge_claude as judge_claude_module

    monkeypatch.setattr(
        judge_claude_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess("ceci n'est pas du JSON", returncode=0),
    )

    with pytest.raises(JudgeCallError):
        call_claude_code("sonnet", "system", "user")
