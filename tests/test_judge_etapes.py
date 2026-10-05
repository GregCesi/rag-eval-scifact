"""Tests du juge par étapes (EXE-121, critères 3, 4, 6 et 11). Aucun appel réseau :
`call_fn` est toujours une fonction fabriquée, une réponse par appel dans l'ordre
des étapes (claim, document, verdict, cause).
"""

from __future__ import annotations

import json

from rag_eval_scifact.judge_etapes import judge_pair, judge_pairs


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


def _claim_response(
    topic="biomateriaux 0D", effect="proprietes inductives", direction="presence"
):
    return json.dumps({"topic": topic, "effect": effect, "direction": direction})


def _document_response(sentences):
    return json.dumps({"sentences": sentences})


def _verdict_response(verdict, decisive_sentence):
    return json.dumps({"verdict": verdict, "decisive_sentence": decisive_sentence})


def _cause_response(cause):
    return json.dumps({"cause": cause})


def _sequenced_call(responses):
    """Un `call_fn(model, system, user, seed)` qui rend les réponses dans l'ordre."""
    calls = []
    remaining = iter(responses)

    def call_fn(model, system, user, seed):
        calls.append({"model": model, "system": system, "user": user, "seed": seed})
        return next(remaining)

    call_fn.calls = calls
    return call_fn


DOC_SENTENCE = "Les biomatériaux 0D ont des propriétés inductives démontrées."


# ---------------------------------------------------------------------------
# Critère 2 — étape 1 sur le claim seul
# ---------------------------------------------------------------------------


def test_claim_step_receives_only_the_claim_text():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judge_pair(pair, call_fn)

    first_call_user_prompt = call_fn.calls[0]["user"]
    assert pair["claim_text"] in first_call_user_prompt
    assert pair["doc_text"] not in first_call_user_prompt
    assert "42" not in first_call_user_prompt
    assert "document_attendu" not in first_call_user_prompt


# ---------------------------------------------------------------------------
# Critère 3 — étape 2, citations vérifiées mot pour mot, phrases écartées comptées
# ---------------------------------------------------------------------------


def test_document_step_discards_sentences_not_verbatim_in_the_document_and_counts_them():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE, "une phrase inventée par le modèle"]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["discarded_sentences"] == 1
    assert judgment["steps"]["document"]["output"]["sentences"] == [DOC_SENTENCE]


def test_verdict_step_only_receives_retained_sentences_not_the_whole_document():
    pair = _pair(
        doc_text=(
            "Les biomatériaux 0D ont des propriétés inductives démontrées. "
            "Ceci est une autre phrase du document, non retenue par l'étape 2."
        )
    )
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judge_pair(pair, call_fn)

    verdict_user_prompt = call_fn.calls[2]["user"]
    assert "non retenue par l'étape 2" not in verdict_user_prompt
    assert DOC_SENTENCE in verdict_user_prompt


# ---------------------------------------------------------------------------
# Critère 4 — aucune phrase retenue : NOT_ENOUGH_INFO / hors sujet, pas d'étapes 3-4
# ---------------------------------------------------------------------------


def test_no_retained_sentence_gives_not_enough_info_hors_sujet_without_calling_steps_3_and_4():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([]),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "NOT_ENOUGH_INFO"
    assert judgment["cause"] == "hors sujet"
    assert judgment["evidence"] == ""
    assert len(call_fn.calls) == 2
    assert "verdict" not in judgment["steps"]
    assert "cause" not in judgment["steps"]


def test_all_candidate_sentences_discarded_also_skips_steps_3_and_4():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response(["une phrase qui n'est pas dans le document"]),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "NOT_ENOUGH_INFO"
    assert judgment["cause"] == "hors sujet"
    assert judgment["discarded_sentences"] == 1
    assert len(call_fn.calls) == 2


# ---------------------------------------------------------------------------
# Critère 5 — étape 3 : soutient, réfute, ou ni l'un ni l'autre
# ---------------------------------------------------------------------------


def test_verdict_step_supports_is_recorded_as_final_verdict_with_decisive_sentence():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "SUPPORTS"
    assert judgment["evidence"] == DOC_SENTENCE
    assert judgment["cause"] == ""
    assert len(call_fn.calls) == 3  # pas d'étape 4


def test_verdict_step_refutes_is_recorded_as_final_verdict():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("REFUTES", DOC_SENTENCE),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "REFUTES"
    assert len(call_fn.calls) == 3


# ---------------------------------------------------------------------------
# Critère 6 — étape 4 appelée seulement quand l'étape 3 répond « ni l'un ni l'autre »
# ---------------------------------------------------------------------------


def test_neither_verdict_calls_step_4_and_records_its_cause():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("NEITHER", DOC_SENTENCE),
            _cause_response("inférence"),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert len(call_fn.calls) == 4
    assert judgment["verdict"] == "NOT_ENOUGH_INFO"
    assert judgment["cause"] == "inférence"
    assert judgment["evidence"] == DOC_SENTENCE


def test_cause_must_be_one_of_the_three_declared_causes():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("NEITHER", DOC_SENTENCE),
            _cause_response("vocabulaire"),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["cause"] in {"vocabulaire", "inférence", "sujet voisin"}


# ---------------------------------------------------------------------------
# Critère 7 — le jugement porte la réponse de chaque étape appelée
# ---------------------------------------------------------------------------


def test_judgment_records_the_response_of_every_step_actually_called():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("NEITHER", DOC_SENTENCE),
            _cause_response("sujet voisin"),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert set(judgment["steps"]) == {"claim", "document", "verdict", "cause"}
    for step in judgment["steps"].values():
        assert "input" in step
        assert "output" in step


# ---------------------------------------------------------------------------
# Critère 8 — quatre nœuds, aucune boucle (le graphe ne rappelle jamais une étape
# déjà passée) : une réponse par étape, dans l'ordre, jamais rejouée.
# ---------------------------------------------------------------------------


def test_graph_calls_each_step_at_most_once_with_a_clean_run():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judge_pair(pair, call_fn)

    assert len(call_fn.calls) == 3


# ---------------------------------------------------------------------------
# Critère 11 — illisible après 2 nouvelles tentatives à une étape
# ---------------------------------------------------------------------------


def test_unparseable_claim_step_response_gives_illisible_after_two_retries():
    pair = _pair()
    call_fn = _sequenced_call(["pas du json", "toujours pas", "encore pas du json"])

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"
    assert len(call_fn.calls) == 3  # 1 essai + 2 nouvelles tentatives, étape 1 seule


def test_unparseable_verdict_step_response_gives_illisible_without_calling_cause():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            "pas du json",
            "toujours pas",
            "encore pas du json",
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"
    assert (
        len(call_fn.calls) == 5
    )  # étapes 1 et 2 (1 appel chacune) + étape 3 (3 essais)


def test_step_recovers_if_a_retry_eventually_parses():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            "pas du json",
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "SUPPORTS"
    assert len(call_fn.calls) == 4


def test_decisive_sentence_outside_the_retained_list_is_rejected_as_unparseable():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", "une phrase qui n'a pas été retenue"),
            "toujours pas",
            "encore pas",
        ]
    )

    judgment = judge_pair(pair, call_fn)

    assert judgment["verdict"] == "illisible"


# ---------------------------------------------------------------------------
# Règles communes (critère 11) — reprise sans rejuger, dite sur la console
# ---------------------------------------------------------------------------


def test_judge_pairs_skips_already_judged_pairs_and_says_so(capsys):
    # EXE-131, critère 3 : une seule ligne de résumé, pas une par paire sautée.
    pairs = [_pair(pair_id="1:d1", doc_id="d1"), _pair(pair_id="1:d2", doc_id="d2")]
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judgments = judge_pairs(pairs, call_fn, already_judged_ids={"1:d1"})

    assert [j["pair_id"] for j in judgments] == ["1:d2"]
    out = capsys.readouterr().out
    assert "1 paire déjà jugée, non rejugée" in out
    assert "1:d1" not in out


def test_judge_pairs_respects_limit_on_new_judgments_only():
    pairs = [
        _pair(pair_id="1:d1", doc_id="d1"),
        _pair(pair_id="1:d2", doc_id="d2"),
        _pair(pair_id="1:d3", doc_id="d3"),
    ]

    def _make_call_fn():
        return _sequenced_call(
            [
                _claim_response(),
                _document_response([DOC_SENTENCE]),
                _verdict_response("SUPPORTS", DOC_SENTENCE),
            ]
            * 3
        )

    judgments = judge_pairs(
        pairs, _make_call_fn(), already_judged_ids={"1:d1"}, limit=1
    )

    assert [j["pair_id"] for j in judgments] == ["1:d2"]


# ---------------------------------------------------------------------------
# Critère 9 — un modèle configurable, température 0, graine fixe, un appel
# Ollama par étape (la graine est transmise telle quelle à chaque appel)
# ---------------------------------------------------------------------------


def test_judge_pairs_stops_cleanly_on_judge_call_error_and_keeps_prior_judgments(
    capsys,
):
    # EXE-131, critère 6 : le juge par étapes reçoit cette règle comme les
    # deux autres. La paire en échec n'est pas enregistrée.
    from rag_eval_scifact.judge_errors import JudgeCallError

    pairs = [
        _pair(pair_id="1:d1", doc_id="d1"),
        _pair(pair_id="1:d2", doc_id="d2"),
    ]
    responses = iter(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    def call_fn(model, system, user, seed):
        try:
            return next(responses)
        except StopIteration:
            raise JudgeCallError("Ollama ne répond pas") from None

    judgments = judge_pairs(pairs, call_fn, already_judged_ids=set())

    assert [j["pair_id"] for j in judgments] == ["1:d1"]
    out = capsys.readouterr().out
    assert "1 jugement(s) gardé(s)" in out


def test_each_step_call_receives_the_configured_model_and_seed():
    pair = _pair()
    call_fn = _sequenced_call(
        [
            _claim_response(),
            _document_response([DOC_SENTENCE]),
            _verdict_response("SUPPORTS", DOC_SENTENCE),
        ]
    )

    judge_pair(pair, call_fn, model="llama3.1:8b", seed=7)

    for call in call_fn.calls:
        assert call["model"] == "llama3.1:8b"
        assert call["seed"] == 7
