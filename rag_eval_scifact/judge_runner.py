"""Boucle commune aux trois juges (EXE-131, critères 1, 2, 3, 6 et 11).

Chaque juge (`judge_local`, `judge_claude`, `judge_etapes`) ne fournit plus
que sa fonction `judge_pair` ; cette boucle affiche le résumé des paires déjà
jugées, juge les suivantes dans l'ordre jusqu'à `limit`, livre chaque
jugement à `on_judgment` dès qu'il est produit (pour une écriture au fil de
l'eau, critère 2), affiche la progression toutes les `PROGRESS_EVERY`
paires, et s'arrête sans trace Python si l'appel au modèle échoue
(`JudgeCallError`, critère 6) — la paire en cours n'est alors pas enregistrée.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from rag_eval_scifact.judge_errors import JudgeCallError

PROGRESS_EVERY = 10

JudgePairFn = Callable[[dict], dict]
OnJudgment = Callable[[dict], None]


def _skip_message(n: int) -> str:
    suffix = "" if n == 1 else "s"
    return f"{n} paire{suffix} déjà jugée{suffix}, non rejugée{suffix}"


def _report_call_error(pair_id: str, error: JudgeCallError, n_kept: int) -> None:
    print(f"{pair_id} : {error}")
    if error.stdout:
        print(f"sortie standard : {error.stdout}")
    if error.stderr:
        print(f"sortie d'erreur : {error.stderr}")
    print(f"{n_kept} jugement(s) gardé(s)")


def run_judge_pairs(
    pairs: list[dict],
    judge_pair_fn: JudgePairFn,
    already_judged_ids: set[str],
    on_judgment: OnJudgment = lambda judgment: None,
    limit: int | None = None,
) -> list[dict]:
    n_skipped = sum(1 for p in pairs if p["pair_id"] in already_judged_ids)
    if n_skipped:
        print(_skip_message(n_skipped))

    remaining = [p for p in pairs if p["pair_id"] not in already_judged_ids]
    if limit is not None:
        remaining = remaining[:limit]
    total = len(remaining)

    judgments: list[dict] = []
    for i, pair in enumerate(remaining, start=1):
        try:
            judgment = judge_pair_fn(pair)
        except JudgeCallError as error:
            _report_call_error(pair["pair_id"], error, len(judgments))
            break
        judgments.append(judgment)
        on_judgment(judgment)
        if i % PROGRESS_EVERY == 0:
            avg = statistics.mean(j["duration_seconds"] for j in judgments)
            print(f"{i}/{total} traité, durée moyenne par paire : {avg:.2f} s")
    return judgments
