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

from rag_eval_scifact.judge_errors import ClaudeRefusal, JudgeCallError
from rag_eval_scifact.judge_local import MAX_RETRIES, citation_found, parse_verdict
from rag_eval_scifact.judge_prompt import SYSTEM_PROMPT, build_user_prompt
from rag_eval_scifact.judge_runner import OnJudgment, run_judge_pairs

DEFAULT_MODEL = "sonnet"
CLAUDE_TIMEOUT_SECONDS = 600
EMPTY_DIR = Path(tempfile.gettempdir()) / "rag-eval-scifact-juge-claude"

# call_fn(model, system_prompt, user_prompt) -> (contenu brut, modèle rapporté, tokens d'entrée)
CallFn = Callable[[str, str, str], tuple[str, str, int]]


def call_claude_code(
    model: str, system_prompt: str, user_prompt: str
) -> tuple[str, str, int]:
    """Un appel non interactif, sans outils, un tour, depuis un répertoire vide (H2, H3).

    Un refus de Claude (`stop_reason == "refusal"`) lève `ClaudeRefusal` ; tout
    autre échec de l'appel (code de sortie, délai, sortie illisible comme
    JSON) lève `JudgeCallError` (EXE-131, critères 4 et 6). Le refus se
    reconnaît à `stop_reason`, jamais au texte du message (H2).
    """
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)
    try:
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
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JudgeCallError(
            f"délai dépassé ({CLAUDE_TIMEOUT_SECONDS} s)",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc

    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise TypeError("la sortie n'est pas un objet JSON")
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeCallError(
            "sortie illisible comme JSON", stdout=result.stdout, stderr=result.stderr
        ) from exc

    if payload.get("stop_reason") == "refusal":
        raise ClaudeRefusal(payload.get("result", ""))

    if result.returncode != 0:
        raise JudgeCallError(
            f"code de sortie {result.returncode}",
            stdout=result.stdout,
            stderr=result.stderr,
        )

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
    try:
        parsed = None
        model_reported, input_tokens = model, 0
        for _ in range(max_retries + 1):
            raw, model_reported, input_tokens = call_fn(
                model, SYSTEM_PROMPT, user_prompt
            )
            parsed = parse_verdict(raw)
            if parsed is not None:
                break
    except ClaudeRefusal as refusal:
        # Aucune reformulation, aucun second essai (ce qui ne doit pas
        # arriver) : le refus est enregistré tel quel, sans retry.
        duration = time.monotonic() - start
        print(f"{pair['pair_id']} : refus du juge — {refusal}")
        return {
            "pair_id": pair["pair_id"],
            "claim_id": pair["claim_id"],
            "doc_id": pair["doc_id"],
            "model": model,
            "input_tokens": 0,
            "duration_seconds": duration,
            "verdict": "refus du juge",
            "level": "",
            "evidence": "",
            "reason": str(refusal),
            "citation_introuvable": False,
        }
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
    limit: int | None = None,
    on_judgment: OnJudgment = lambda judgment: None,
) -> list[dict]:
    """Juge les paires non encore jugées, dans l'ordre, jusqu'à `limit`, par la
    boucle commune aux trois juges (`judge_runner`, EXE-131)."""
    return run_judge_pairs(
        pairs,
        lambda pair: judge_pair(pair, call_fn, model=model),
        already_judged_ids,
        on_judgment=on_judgment,
        limit=limit,
    )
