"""Juge de référence Claude : un appel Claude Code non interactif par paire (EXE-120).

Même prompt octet pour octet que le juge local (`judge_prompt.py`, critère 3) et
mêmes règles de lecture de la réponse (`parse_verdict`, `citation_found`,
réessais, importés depuis `judge_local.py` sans le modifier — le juge local ne
change pas). `call_fn` est injectable (tests sans appel réel) ; en production
c'est `call_claude_code`, seul point du module qui lance le CLI Claude Code —
l'exception documentée dans `.claude/rules/methodologie.md` (juge de référence
v3-juge, abonnement, sans clé d'API).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from rag_eval_scifact.judge_local import MAX_RETRIES, citation_found, parse_verdict
from rag_eval_scifact.judge_prompt import SYSTEM_PROMPT, build_user_prompt

DEFAULT_MODEL = "sonnet"
CLAUDE_TIMEOUT_SECONDS = 600
EMPTY_DIR = Path(tempfile.gettempdir()) / "rag-eval-scifact-juge-claude"

# call_fn(model, system_prompt, user_prompt) -> (contenu brut, modèle rapporté, tokens d'entrée)
CallFn = Callable[[str, str, str], tuple[str, str, int]]


def call_claude_code(
    model: str, system_prompt: str, user_prompt: str
) -> tuple[str, str, int]:
    """Un appel non interactif, sans outils, un tour, depuis un répertoire vide (H2, H3)."""
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--system-prompt",
            system_prompt,
            "--tools",
            "",
            "--output-format",
            "json",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            user_prompt,
        ],
        cwd=EMPTY_DIR,
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT_SECONDS,
        check=True,
    )
    payload = json.loads(result.stdout)
    model_reported = next(iter(payload["modelUsage"]))
    usage = payload["usage"]
    # Claude Code met le prompt entier en cache dès le premier appel : sans compter
    # cache_creation_input_tokens et cache_read_input_tokens, usage["input_tokens"]
    # reste proche de 0 à chaque appel et ne dit rien de la taille réelle du contexte.
    input_tokens = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    return payload["result"], model_reported, input_tokens


def judge_pair(
    pair: dict,
    call_fn: CallFn,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Juge une paire par Claude. Ne transmet que le claim et le document (critère 6,
    comme le juge local) : `pair["rank"]` et `pair["document_attendu"]` ne sont jamais
    lus ici pour le prompt.
    """
    user_prompt = build_user_prompt(
        pair["claim_text"], pair["doc_title"], pair["doc_text"]
    )

    start = time.monotonic()
    parsed = None
    model_reported, input_tokens = model, 0
    for _ in range(max_retries + 1):
        raw, model_reported, input_tokens = call_fn(model, SYSTEM_PROMPT, user_prompt)
        parsed = parse_verdict(raw)
        if parsed is not None:
            break
    duration = time.monotonic() - start

    judgment = {
        "pair_id": pair["pair_id"],
        "claim_id": pair["claim_id"],
        "doc_id": pair["doc_id"],
        "model": model_reported,
        "input_tokens": input_tokens,
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
    limit: int | None = None,
) -> list[dict]:
    """Juge les paires non encore jugées, dans l'ordre, jusqu'à `limit` (critère 6,
    reprise sans rejuger — comme le juge local).
    """
    judgments: list[dict] = []
    for pair in pairs:
        if pair["pair_id"] in already_judged_ids:
            print(f"{pair['pair_id']} : déjà jugée, non rejugée")
            continue
        if limit is not None and len(judgments) >= limit:
            break
        judgments.append(judge_pair(pair, call_fn, model=model))
    return judgments
