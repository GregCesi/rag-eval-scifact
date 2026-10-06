"""Tests du CLI de rédaction HyDE (EXE-141, critères 1, 4, 5, 6, 7). Aucun
modèle n'est appelé : `call_ollama` est remplacé par une fonction fabriquée.
"""

from __future__ import annotations

import json

import pytest

import rag_eval_scifact.run_hyde as run_hyde_module
from rag_eval_scifact.hyde import HydeCallError

CLAIMS = [{"_id": str(i), "text": f"Affirmation {i}."} for i in range(1, 31)]


@pytest.fixture
def _isolated_hyde(tmp_path, monkeypatch):
    hyde_path = tmp_path / "hyde.json"
    monkeypatch.setattr(run_hyde_module, "HYDE_PATH", hyde_path)
    monkeypatch.setattr(
        run_hyde_module, "load_test_queries", lambda *a, **k: (CLAIMS, {})
    )
    return hyde_path


def _fake_call_ollama(calls):
    def call_fn(model, prompt, seed):
        calls.append(prompt)
        return "Un texte rédigé."

    return call_fn


# ---------------------------------------------------------------------------
# Critère 1 — champs écrits dans hyde.json
# ---------------------------------------------------------------------------


def test_writes_one_entry_per_claim_with_required_fields(_isolated_hyde, monkeypatch):
    monkeypatch.setattr(run_hyde_module, "call_ollama", _fake_call_ollama([]))

    run_hyde_module.main(["--limit", "3"])

    entries = json.loads(_isolated_hyde.read_text(encoding="utf-8"))
    assert len(entries) == 3
    for entry, claim in zip(entries, CLAIMS[:3]):
        assert entry["claim_id"] == claim["_id"]
        assert entry["claim_text"] == claim["text"]
        assert entry["hyde_text"] == "Un texte rédigé."
        assert entry["model"]
        assert entry["duration_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Critère 7 — option bornant le nombre d'affirmations traitées
# ---------------------------------------------------------------------------


def test_limit_option_bounds_the_number_of_claims_written(_isolated_hyde, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(run_hyde_module, "call_ollama", _fake_call_ollama(calls))

    run_hyde_module.main(["--limit", "5"])

    assert len(calls) == 5
    entries = json.loads(_isolated_hyde.read_text(encoding="utf-8"))
    assert len(entries) == 5


# ---------------------------------------------------------------------------
# Critère 4 — une interruption laisse le fichier intact et lisible
# ---------------------------------------------------------------------------


def test_interruption_after_twenty_five_claims_keeps_the_file_readable(
    _isolated_hyde, monkeypatch
):
    def call_fn(model, prompt, seed):
        if "Affirmation 26." in prompt:
            raise KeyboardInterrupt
        return "Un texte rédigé."

    monkeypatch.setattr(run_hyde_module, "call_ollama", call_fn)

    with pytest.raises(KeyboardInterrupt):
        run_hyde_module.main([])

    entries = json.loads(_isolated_hyde.read_text(encoding="utf-8"))
    assert len(entries) == 25
    assert {e["claim_id"] for e in entries} == {str(i) for i in range(1, 26)}


# ---------------------------------------------------------------------------
# Critère 5 — relance : les affirmations déjà rédigées ne le sont pas à
# nouveau, la console le dit
# ---------------------------------------------------------------------------


def test_relaunch_does_not_rewrite_already_written_claims(
    _isolated_hyde, monkeypatch, capsys
):
    existing = [
        {
            "claim_id": str(i),
            "claim_text": f"Affirmation {i}.",
            "hyde_text": "déjà rédigé",
            "model": "llama3.1:8b",
            "duration_seconds": 1.0,
        }
        for i in range(1, 26)
    ]
    _isolated_hyde.write_text(json.dumps(existing), encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(run_hyde_module, "call_ollama", _fake_call_ollama(calls))

    run_hyde_module.main([])

    assert len(calls) == 5  # 30 affirmations - 25 déjà rédigées
    out = capsys.readouterr().out
    assert "25 affirmations déjà rédigées, non rédigées à nouveau" in out

    entries = json.loads(_isolated_hyde.read_text(encoding="utf-8"))
    assert len(entries) == 30
    for entry in entries[:25]:
        assert entry["hyde_text"] == "déjà rédigé"


# ---------------------------------------------------------------------------
# Critère 6 — arrêt propre sur erreur de modèle, aucune trace Python,
# l'affirmation en cours n'est pas enregistrée
# ---------------------------------------------------------------------------


def test_model_call_error_stops_cleanly_without_saving_the_current_claim(
    _isolated_hyde, monkeypatch, capsys
):
    def call_fn(model, prompt, seed):
        if "Affirmation 3." in prompt:
            raise HydeCallError("Ollama ne répond pas")
        return "Un texte rédigé."

    monkeypatch.setattr(run_hyde_module, "call_ollama", call_fn)

    run_hyde_module.main([])  # ne lève pas : l'erreur est gérée proprement

    entries = json.loads(_isolated_hyde.read_text(encoding="utf-8"))
    assert {e["claim_id"] for e in entries} == {"1", "2"}
    out = capsys.readouterr().out
    assert "3 : Ollama ne répond pas" in out
    assert "2 texte(s) gardé(s)" in out
