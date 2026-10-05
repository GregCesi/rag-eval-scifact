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

from rag_eval_scifact.judge_errors import JudgeCallError
from rag_eval_scifact.judge_prompt import SYSTEM_PROMPT, build_user_prompt
from rag_eval_scifact.judge_runner import OnJudgment, run_judge_pairs

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"
SEED = 0
MAX_RETRIES = 2
VALID_VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
VALID_LEVELS = {"DIRECT", "VOCABULARY", "REASONING", "NONE"}
RESPONDING_VERDICTS = {"SUPPORTS", "REFUTES"}

CallFn = Callable[[str, str, str, int], str]


def call_ollama(model: str, system_prompt: str, user_prompt: str, seed: int) -> str:
    """Appel réel à Ollama (`/api/chat`, un tour, température 0, graine fixe).

    Une erreur réseau (Ollama qui ne répond pas) ou une sortie illisible
    devient une `JudgeCallError` propre (EXE-131, critère 6), jamais une trace
    Python brute.
    """
    try:
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
    except requests.RequestException as exc:
        raise JudgeCallError(f"Ollama ne répond pas : {exc}") from exc

    try:
        return response.json()["message"]["content"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise JudgeCallError(
            "sortie Ollama illisible comme JSON", stdout=response.text
        ) from exc


def _level_consistent_with_verdict(verdict: str, level: str) -> bool:
    """Critère 6 (EXE-125) : NONE seulement pour NOT_ENOUGH_INFO, jamais pour
    SUPPORTS/REFUTES, et réciproquement."""
    if verdict in RESPONDING_VERDICTS:
        return level != "NONE"
    return level == "NONE"


def parse_verdict(raw: str) -> dict | None:
    """Parse la réponse JSON du modèle ; `None` si illisible, hors énumération,
    ou si le niveau de lecture ne va pas avec le verdict (critère 6, EXE-125)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    level = data.get("level")
    evidence = data.get("evidence", "")
    reason = data.get("reason", "")
    if (
        verdict not in VALID_VERDICTS
        or level not in VALID_LEVELS
        or not isinstance(evidence, str)
        or not isinstance(reason, str)
        or not _level_consistent_with_verdict(verdict, level)
    ):
        return None
    return {"verdict": verdict, "level": level, "evidence": evidence, "reason": reason}


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
        return {
            **judgment,
            "verdict": "illisible",
            "level": "",
            "evidence": "",
            "reason": "",
            "citation_introuvable": False,
        }

    citation_ok = citation_found(parsed["evidence"], pair["doc_text"])
    return {**judgment, **parsed, "citation_introuvable": not citation_ok}


def judge_pairs(
    pairs: list[dict],
    call_fn: CallFn,
    already_judged_ids: set[str],
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    limit: int | None = None,
    on_judgment: OnJudgment = lambda judgment: None,
) -> list[dict]:
    """Juge les paires non encore jugées, dans l'ordre, jusqu'à `limit`, par la
    boucle commune aux trois juges (`judge_runner`, EXE-131)."""
    return run_judge_pairs(
        pairs,
        lambda pair: judge_pair(pair, call_fn, model=model, seed=seed),
        already_judged_ids,
        on_judgment=on_judgment,
        limit=limit,
    )
