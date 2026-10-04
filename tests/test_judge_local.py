"""Tests du juge local (EXE-119, critères 5, 7, 8, 9). Aucun appel réseau : `call_fn`
est toujours une fonction fabriquée.
"""

from __future__ import annotations

import json

from rag_eval_scifact.judge_local import citation_found, judge_pair, judge_pairs
from rag_eval_scifact.judge_prompt import build_user_prompt


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
    }
    base.update(overrides)
    return base


def _fake_call(response: str):
    calls = []

    def call_fn(model, system, user, seed):
        calls.append({"model": model, "system": system, "user": user, "seed": seed})
        return response

    call_fn.calls = calls
    return call_fn


# ---------------------------------------------------------------------------
# Critère 5 — le juge ne reçoit ni document_attendu, ni rang, ni score
# ---------------------------------------------------------------------------


def test_judge_only_sends_claim_and_document_text_to_the_model():
    pair = _pair(rank=42, document_attendu=True)
    call_fn = _fake_call(
        json.dumps({"verdict": "SUPPORTS", "evidence": "", "reason": "ok"})
    )

    judge_pair(pair, call_fn)

    sent_user_prompt = call_fn.calls[0]["user"]
    assert sent_user_prompt == build_user_prompt(
        pair["claim_text"], pair["doc_title"], pair["doc_text"]
    )
    assert "42" not in sent_user_prompt
    assert "document_attendu" not in sent_user_prompt


# ---------------------------------------------------------------------------
# Critère 7 — citation introuvable
# ---------------------------------------------------------------------------


def test_citation_found_true_for_empty_evidence():
    assert citation_found("", "peu importe le texte") is True


def test_citation_found_false_when_sentence_not_verbatim_in_document():
    assert (
        citation_found("une phrase absente", "un texte totalement différent") is False
    )


def test_verdict_marked_citation_introuvable_when_evidence_not_in_doc_text():
    pair = _pair(doc_text="Un texte qui ne contient pas la citation inventée.")
    call_fn = _fake_call(
        json.dumps(
            {
                "verdict": "SUPPORTS",
                "evidence": "phrase qui n'existe pas",
                "reason": "x",
            }
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "citation introuvable"


def test_verdict_kept_when_evidence_is_verbatim_in_doc_text():
    pair = _pair(doc_text="Cette phrase précise est bien dans le document.")
    call_fn = _fake_call(
        json.dumps(
            {
                "verdict": "SUPPORTS",
                "evidence": "Cette phrase précise est bien dans le document.",
                "reason": "x",
            }
        )
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "SUPPORTS"


# ---------------------------------------------------------------------------
# Critère 8 — illisible après 2 nouvelles tentatives
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
            "pas du json",
            "toujours pas",
            json.dumps(
                {
                    "verdict": "REFUTES",
                    "evidence": "Preuve décisive citée mot pour mot.",
                    "reason": "x",
                }
            ),
        ]
    )
    calls = []

    def call_fn(model, system, user, seed):
        calls.append(1)
        return next(responses)

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "REFUTES"
    assert len(calls) == 3


def test_verdict_illisible_when_verdict_field_outside_enum():
    pair = _pair()
    call_fn = _fake_call(
        json.dumps({"verdict": "MAYBE", "evidence": "", "reason": "x"})
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"


# ---------------------------------------------------------------------------
# Critère 9 — reprise sans rejuger, dite sur la console
# ---------------------------------------------------------------------------


def test_judge_pairs_skips_already_judged_pairs_and_says_so(capsys):
    pairs = [_pair(pair_id="1:d1", doc_id="d1"), _pair(pair_id="1:d2", doc_id="d2")]
    call_fn = _fake_call(
        json.dumps({"verdict": "SUPPORTS", "evidence": "", "reason": "x"})
    )

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"})

    assert [j["pair_id"] for j in judgments] == ["1:d2"]
    assert len(call_fn.calls) == 1
    out = capsys.readouterr().out
    assert "1:d1" in out
    assert "déjà jugée" in out


def test_judge_pairs_respects_limit_on_new_judgments_only():
    pairs = [
        _pair(pair_id="1:d1", doc_id="d1"),
        _pair(pair_id="1:d2", doc_id="d2"),
        _pair(pair_id="1:d3", doc_id="d3"),
    ]
    call_fn = _fake_call(
        json.dumps({"verdict": "SUPPORTS", "evidence": "", "reason": "x"})
    )

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"}, limit=1)

    assert [j["pair_id"] for j in judgments] == ["1:d2"]
