"""Rédaction HyDE : un faux résumé d'article par affirmation, via Ollama (EXE-141).

Le problème ciblé (fiche EXE-141) : les goldens manquants dans le top 5 de la
meilleure config de v2-grid sont ceux où l'affirmation et le document ne
parlent pas avec les mêmes mots. HyDE fait rédiger par un modèle local un
texte qui parle comme un article avant de chercher avec ce texte, à la place
de l'affirmation.

`call_fn` est injectable (tests sans aucun modèle réel) ; en production c'est
`call_ollama`, le seul point du module qui parle réseau — uniquement vers
Ollama en local (`.claude/rules/methodologie.md`).
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"
SEED = 0
PROGRESS_EVERY = 10
HYDE_PATH = Path("results/v4-leviers/hyde.json")
N_CLAIMS = 300

# Prompt H1, décidé par le chat le 6 octobre 2026 (critère 2) : envoyé mot pour
# mot, sans système ajouté — c'est un texte de document à produire, pas une
# question à juger.
PROMPT_TEMPLATE = (
    "Write a short passage from the abstract of a scientific paper that "
    "reports evidence for the following claim. Write 3 to 4 sentences, in "
    "the style of a biomedical abstract. Do not repeat the claim word for "
    "word and do not add any commentary.\n\nClaim: {claim}"
)

CallFn = Callable[[str, str, int], str]


class HydeCallError(Exception):
    """L'appel au modèle a échoué (réseau) ou a rendu un texte vide (critère 6).
    L'affirmation en cours n'est pas enregistrée ; elle sera rédigée à la relance."""


def build_prompt(claim_text: str) -> str:
    """Prompt H1, mot pour mot (critère 2)."""
    return PROMPT_TEMPLATE.format(claim=claim_text)


def call_ollama(model: str, prompt: str, seed: int) -> str:
    """Appel réel à Ollama (`/api/chat`, un tour, température 0, graine fixe).

    Une erreur réseau ou une sortie illisible comme JSON devient une
    `HydeCallError` propre (critère 6), jamais une trace Python brute. Rend
    le texte tel que rendu par le modèle (`write_claim_hyde` le nettoie).
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "seed": seed},
            },
            timeout=600,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HydeCallError(f"Ollama ne répond pas : {exc}") from exc

    try:
        return response.json()["message"]["content"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise HydeCallError("sortie Ollama illisible comme JSON") from exc


def write_claim_hyde(
    claim: dict,
    call_fn: CallFn,
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
) -> dict:
    """Rédige le texte HyDE d'une affirmation : un seul appel, jamais de
    second essai automatique après une erreur (« ce qui ne doit pas arriver »,
    EXE-141 — à la différence des juges, qui retentent sur une sortie illisible).

    Seuls les espaces de début et de fin du texte rendu sont retirés : il
    n'est ni corrigé, ni tronqué, ni complété (« ce qui ne doit pas
    arriver »). Un texte vide après ce nettoyage devient une `HydeCallError`
    (critère 6), que `call_fn` soit réel ou fabriqué.
    """
    prompt = build_prompt(claim["text"])
    start = time.monotonic()
    text = call_fn(model, prompt, seed).strip()
    duration = time.monotonic() - start
    if not text:
        raise HydeCallError("le modèle a rendu un texte vide")
    return {
        "claim_id": claim["_id"],
        "claim_text": claim["text"],
        "hyde_text": text,
        "model": model,
        "duration_seconds": duration,
    }


def _skip_message(n: int) -> str:
    suffix = "" if n == 1 else "s"
    return (
        f"{n} affirmation{suffix} déjà rédigée{suffix}, non rédigée{suffix} à nouveau"
    )


def _report_call_error(claim_id: str, error: HydeCallError, n_kept: int) -> None:
    print(f"{claim_id} : {error}")
    print(f"{n_kept} texte(s) gardé(s)")


OnText = Callable[[dict], None]


def write_hyde_texts(
    claims: list[dict],
    call_fn: CallFn,
    already_written_ids: set[str],
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    limit: int | None = None,
    on_text: OnText = lambda entry: None,
) -> list[dict]:
    """Rédige les affirmations non encore rédigées, dans l'ordre, jusqu'à `limit`.

    Livre chaque texte à `on_text` dès qu'il est produit (écriture au fil de
    l'eau, pour qu'une interruption garde tout ce qui est déjà rédigé —
    critère 4), affiche la progression toutes les `PROGRESS_EVERY`
    affirmations (critère 3), et s'arrête sans trace Python si l'appel au
    modèle échoue (critère 6) — l'affirmation en cours n'est alors pas
    enregistrée.
    """
    n_skipped = sum(1 for c in claims if c["_id"] in already_written_ids)
    if n_skipped:
        print(_skip_message(n_skipped))

    remaining = [c for c in claims if c["_id"] not in already_written_ids]
    if limit is not None:
        remaining = remaining[:limit]
    total = len(remaining)

    entries: list[dict] = []
    for i, claim in enumerate(remaining, start=1):
        try:
            entry = write_claim_hyde(claim, call_fn, model=model, seed=seed)
        except HydeCallError as error:
            _report_call_error(claim["_id"], error, len(entries))
            break
        entries.append(entry)
        on_text(entry)
        if i % PROGRESS_EVERY == 0:
            avg = statistics.mean(e["duration_seconds"] for e in entries)
            print(f"{i}/{total} traité, durée moyenne : {avg:.2f} s")
    return entries


def load_hyde_texts(path: Path = HYDE_PATH) -> dict[str, str]:
    """Charge `hyde.json` : `claim_id` -> texte rédigé. Vide si le fichier est absent."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["claim_id"]: entry["hyde_text"] for entry in data}
