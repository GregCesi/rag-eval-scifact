"""Tests de la rédaction HyDE (EXE-141, critères 2, 3, 6, 7, 9, 10, 11). Aucun
appel réseau : `call_fn` est toujours une fonction fabriquée.
"""

from __future__ import annotations

import json

import pytest
import requests

from rag_eval_scifact.hyde import (
    HydeCallError,
    build_prompt,
    call_ollama,
    load_hyde_texts,
    write_claim_hyde,
    write_hyde_texts,
)


def _claim(
    claim_id: str = "1",
    text: str = "0-dimensional biomaterials show inductive properties.",
):
    return {"_id": claim_id, "text": text}


def _fake_call(responses):
    """`responses` : une chaîne (même réponse à chaque appel) ou un itérable."""
    if isinstance(responses, str):
        fixed = responses
        responses = iter(lambda: fixed, object())
    else:
        responses = iter(responses)
    calls = []

    def call_fn(model, prompt, seed):
        calls.append({"model": model, "prompt": prompt, "seed": seed})
        return next(responses)

    call_fn.calls = calls
    return call_fn


# ---------------------------------------------------------------------------
# Critère 2 — le prompt H1 est envoyé mot pour mot
# ---------------------------------------------------------------------------


def test_build_prompt_matches_h1_word_for_word():
    prompt = build_prompt("0-dimensional biomaterials show inductive properties.")

    assert prompt == (
        "Write a short passage from the abstract of a scientific paper that "
        "reports evidence for the following claim. Write 3 to 4 sentences, in "
        "the style of a biomedical abstract. Do not repeat the claim word for "
        "word and do not add any commentary.\n\n"
        "Claim: 0-dimensional biomaterials show inductive properties."
    )


def test_write_claim_hyde_sends_the_h1_prompt():
    call_fn = _fake_call("Un texte rédigé.")

    write_claim_hyde(_claim(text="Une affirmation."), call_fn)

    assert call_fn.calls[0]["prompt"] == build_prompt("Une affirmation.")


# ---------------------------------------------------------------------------
# Critère 1 — champs du résultat ; le texte n'est ni corrigé ni complété
# ---------------------------------------------------------------------------


def test_write_claim_hyde_keeps_the_written_text_as_is_only_stripped():
    call_fn = _fake_call("  Un texte rédigé avec des espaces.  ")

    entry = write_claim_hyde(_claim(claim_id="3", text="Une affirmation."), call_fn)

    assert entry["claim_id"] == "3"
    assert entry["claim_text"] == "Une affirmation."
    assert entry["hyde_text"] == "Un texte rédigé avec des espaces."
    assert entry["model"]
    assert entry["duration_seconds"] >= 0.0


def test_write_claim_hyde_never_retries_on_its_own():
    # « Ce qui ne doit pas arriver » (EXE-141) : un seul essai, jamais un
    # second essai automatique après une erreur — à la différence des juges.
    call_fn = _fake_call("Un texte rédigé.")

    write_claim_hyde(_claim(), call_fn)

    assert len(call_fn.calls) == 1


# ---------------------------------------------------------------------------
# Critère 9 — reprise sans rédiger à nouveau, dite sur la console
# ---------------------------------------------------------------------------


def test_write_hyde_texts_skips_already_written_claims_and_says_so(capsys):
    claims = [_claim("1"), _claim("2")]
    call_fn = _fake_call("Un texte rédigé.")

    entries = write_hyde_texts(claims, call_fn, already_written_ids={"1"})

    assert [e["claim_id"] for e in entries] == ["2"]
    assert len(call_fn.calls) == 1
    out = capsys.readouterr().out
    assert "1 affirmation déjà rédigée, non rédigée à nouveau" in out


def test_write_hyde_texts_no_skip_message_when_nothing_already_written(capsys):
    write_hyde_texts(
        [_claim()], _fake_call("Un texte rédigé."), already_written_ids=set()
    )

    out = capsys.readouterr().out
    assert "déjà rédigée" not in out


# ---------------------------------------------------------------------------
# Critère 3 — progression toutes les 10 affirmations
# ---------------------------------------------------------------------------


def test_progress_line_every_ten_claims(capsys):
    claims = [_claim(str(i)) for i in range(1, 26)]

    write_hyde_texts(claims, _fake_call("Un texte rédigé."), already_written_ids=set())

    out = capsys.readouterr().out
    progress_lines = [line for line in out.splitlines() if "traité" in line]
    assert len(progress_lines) == 2
    assert progress_lines[0].startswith("10/25 traité, durée moyenne : ")
    assert progress_lines[1].startswith("20/25 traité, durée moyenne : ")


def test_no_progress_line_under_ten_claims(capsys):
    write_hyde_texts(
        [_claim()], _fake_call("Un texte rédigé."), already_written_ids=set()
    )

    out = capsys.readouterr().out
    assert "traité" not in out


# ---------------------------------------------------------------------------
# Critère 4 (porté ici) — chaque texte est livré dès qu'il est produit
# ---------------------------------------------------------------------------


def test_on_text_is_called_immediately_for_each_entry():
    claims = [_claim("1"), _claim("2"), _claim("3")]
    delivered = []

    write_hyde_texts(
        claims,
        _fake_call("Un texte rédigé."),
        already_written_ids=set(),
        on_text=delivered.append,
    )

    assert [e["claim_id"] for e in delivered] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Critère 6 — arrêt propre sur HydeCallError, aucune trace Python, rien
# d'enregistré pour l'affirmation en cours
# ---------------------------------------------------------------------------


def test_stops_cleanly_on_hyde_call_error_keeping_prior_entries(capsys):
    claims = [
        _claim("1", text="Affirmation 1."),
        _claim("2", text="Affirmation 2."),
        _claim("3", text="Affirmation 3."),
    ]

    def call_fn(model, prompt, seed):
        if "Affirmation 2." in prompt:
            raise HydeCallError("Ollama ne répond pas : erreur réseau")
        return "Un texte rédigé."

    entries = write_hyde_texts(claims, call_fn, already_written_ids=set())

    assert [e["claim_id"] for e in entries] == ["1"]
    out = capsys.readouterr().out
    assert "2 : Ollama ne répond pas : erreur réseau" in out
    assert "1 texte(s) gardé(s)" in out


def test_hyde_call_error_does_not_propagate_past_the_loop():
    def call_fn(model, prompt, seed):
        raise HydeCallError("Ollama ne répond pas")

    entries = write_hyde_texts([_claim()], call_fn, already_written_ids=set())
    assert entries == []


# ---------------------------------------------------------------------------
# Critère 7 — option bornant le nombre d'affirmations traitées
# ---------------------------------------------------------------------------


def test_limit_bounds_the_number_of_claims_written():
    claims = [_claim("1"), _claim("2"), _claim("3")]

    entries = write_hyde_texts(
        claims, _fake_call("Un texte rédigé."), already_written_ids=set(), limit=2
    )

    assert [e["claim_id"] for e in entries] == ["1", "2"]


def test_limit_applies_only_to_remaining_claims_not_already_written():
    claims = [_claim("1"), _claim("2"), _claim("3")]

    entries = write_hyde_texts(
        claims,
        _fake_call("Un texte rédigé."),
        already_written_ids={"1"},
        limit=1,
    )

    assert [e["claim_id"] for e in entries] == ["2"]


# ---------------------------------------------------------------------------
# call_ollama — réseau et texte vide deviennent une HydeCallError propre
# ---------------------------------------------------------------------------


def test_call_ollama_wraps_connection_failure_as_hyde_call_error(monkeypatch):
    import rag_eval_scifact.hyde as hyde_module

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Ollama ne répond pas")

    monkeypatch.setattr(hyde_module.requests, "post", _raise)

    with pytest.raises(HydeCallError):
        call_ollama("llama3.1:8b", "prompt", 0)


def test_call_ollama_sends_temperature_zero_and_fixed_seed(monkeypatch):
    import rag_eval_scifact.hyde as hyde_module

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "Un texte rédigé."}}

    def _post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(hyde_module.requests, "post", _post)

    text = call_ollama("llama3.1:8b", "un prompt", 0)

    assert text == "Un texte rédigé."
    assert captured["json"]["model"] == "llama3.1:8b"
    assert captured["json"]["options"] == {"temperature": 0, "seed": 0}
    assert captured["json"]["messages"] == [{"role": "user", "content": "un prompt"}]


def test_write_claim_hyde_raises_on_empty_text_from_any_call_fn():
    call_fn = _fake_call("   ")

    with pytest.raises(HydeCallError):
        write_claim_hyde(_claim(), call_fn)


# ---------------------------------------------------------------------------
# load_hyde_texts — critère 10 (compte des textes présents)
# ---------------------------------------------------------------------------


def test_load_hyde_texts_returns_empty_dict_when_file_missing(tmp_path):
    assert load_hyde_texts(tmp_path / "absent.json") == {}


def test_load_hyde_texts_maps_claim_id_to_hyde_text(tmp_path):
    path = tmp_path / "hyde.json"
    path.write_text(
        json.dumps(
            [
                {
                    "claim_id": "1",
                    "claim_text": "affirmation",
                    "hyde_text": "texte rédigé",
                    "model": "llama3.1:8b",
                    "duration_seconds": 1.2,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_hyde_texts(path) == {"1": "texte rédigé"}
