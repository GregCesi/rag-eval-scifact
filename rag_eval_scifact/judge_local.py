"""Juge local Ollama : un appel par paire, prompt H1 envoyé tel quel (EXE-119).

`call_fn` est injectable (tests sans aucun modèle réel) ; en production c'est
`call_ollama`, le seul point du module qui parle réseau — uniquement vers
Ollama en local (`.claude/rules/methodologie.md`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import requests

from rag_eval_scifact.judge_prompt import SYSTEM_PROMPT, build_user_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"
SEED = 0
MAX_RETRIES = 2
VALID_VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}

CallFn = Callable[[str, str, str, int], str]


def call_ollama(model: str, system_prompt: str, user_prompt: str, seed: int) -> str:
    """Appel réel à Ollama (`/api/chat`, un tour, température 0, graine fixe)."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": seed},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_verdict(raw: str) -> dict | None:
    """Parse la réponse JSON du modèle ; `None` si illisible ou hors énumération."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    evidence = data.get("evidence", "")
    reason = data.get("reason", "")
    if (
        verdict not in VALID_VERDICTS
        or not isinstance(evidence, str)
        or not isinstance(reason, str)
    ):
        return None
    return {"verdict": verdict, "evidence": evidence, "reason": reason}


def citation_found(evidence: str, doc_text: str) -> bool:
    """Phrase citée verbatim (mot pour mot) dans le texte du document (critère 7).

    Une citation vide (NOT_ENOUGH_INFO, H1) n'est jamais marquée introuvable.
    """
    if evidence == "":
        return True
    return evidence in doc_text


def judge_pair(
    pair: dict,
    call_fn: CallFn,
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Juge une paire. Ne transmet au modèle que le claim et le document (critère 5) :
    `pair["rank"]` et `pair["document_attendu"]` ne sont jamais lus ici pour le prompt.
    """
    user_prompt = build_user_prompt(
        pair["claim_text"], pair["doc_title"], pair["doc_text"]
    )

    start = time.monotonic()
    parsed = None
    for _ in range(max_retries + 1):
        raw = call_fn(model, SYSTEM_PROMPT, user_prompt, seed)
        parsed = parse_verdict(raw)
        if parsed is not None:
            break
    duration = time.monotonic() - start

    judgment = {
        "pair_id": pair["pair_id"],
        "claim_id": pair["claim_id"],
        "doc_id": pair["doc_id"],
        "model": model,
        "duration_seconds": duration,
    }

    if parsed is None:
        return {**judgment, "verdict": "illisible", "evidence": "", "reason": ""}

    if not citation_found(parsed["evidence"], pair["doc_text"]):
        return {
            **judgment,
            "verdict": "citation introuvable",
            "evidence": parsed["evidence"],
            "reason": parsed["reason"],
        }

    return {**judgment, **parsed}


def judge_pairs(
    pairs: list[dict],
    call_fn: CallFn,
    already_judged_ids: set[str],
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    limit: int | None = None,
) -> list[dict]:
    """Juge les paires non encore jugées, dans l'ordre, jusqu'à `limit` (critère 9 et 11).

    Une paire dont le `pair_id` est dans `already_judged_ids` n'est jamais
    rejugée : la reprise après interruption le dit sur la console.
    """
    judgments: list[dict] = []
    for pair in pairs:
        if pair["pair_id"] in already_judged_ids:
            print(f"{pair['pair_id']} : déjà jugée, non rejugée")
            continue
        if limit is not None and len(judgments) >= limit:
            break
        judgments.append(judge_pair(pair, call_fn, model=model, seed=seed))
    return judgments
