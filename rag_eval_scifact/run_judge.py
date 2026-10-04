"""CLI : juge les paires d'une campagne avec un juge déclaré (EXE-119).

Seul le juge « local » (Ollama) existe à ce tour ; `--juge` est déjà pluriel
pour que les juges suivants (Claude, par étapes) s'ajoutent sans renommer rien.

Usage : python -m rag_eval_scifact.run_judge --campagne v3-juge
        python -m rag_eval_scifact.run_judge --campagne dev --limit 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.processor.base_mlflow import flush_all_batch_processors

import rag_eval_scifact.mlflow_tracking  # noqa: F401  — URI de tracking + astuce agent désactivée
from rag_eval_scifact.judge_local import DEFAULT_MODEL, call_ollama, judge_pairs
from rag_eval_scifact.judge_pairs import load_pairs

GATED_CAMPAGNE = "v3-juge"
PREDICTION_PATH = "results/v3-juge/PREDICTION.md"
RESULTS_DIR = Path("results")

JUDGES = {"local": call_ollama}


def _prediction_committed() -> bool:
    """`PREDICTION_PATH` est présent dans le dernier commit (`git show HEAD:...`)."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{PREDICTION_PATH}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _pairs_path(campagne: str) -> Path:
    return RESULTS_DIR / campagne / "paires.json"


def _judgments_path(campagne: str, juge: str) -> Path:
    return RESULTS_DIR / campagne / f"jugements-{juge}.json"


def _load_existing_judgments(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {j["pair_id"]: j for j in json.loads(path.read_text(encoding="utf-8"))}


def _write_judgments(path: Path, judgments_by_id: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(judgments_by_id.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_judgment_traces(run_id: str, judgments: list[dict]) -> None:
    for j in judgments:
        with mlflow.start_span(
            name=f"judge-{j['pair_id']}", span_type=SpanType.LLM, run_id=run_id
        ) as span:
            span.set_inputs({"pair_id": j["pair_id"]})
            span.set_outputs(
                {
                    "verdict": j["verdict"],
                    "evidence": j["evidence"],
                    "reason": j["reason"],
                }
            )
    flush_all_batch_processors()


def _print_summary(judgments: list[dict]) -> None:
    if not judgments:
        print("Aucun nouveau jugement.")
        return
    counts = Counter(j["verdict"] for j in judgments)
    print(f"{len(judgments)} nouveau(x) jugement(s) :")
    for verdict, n in counts.items():
        print(f"  {verdict} : {n}")
    duree_moyenne = statistics.mean(j["duration_seconds"] for j in judgments)
    print(f"  durée moyenne par jugement : {duree_moyenne:.2f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campagne",
        default=GATED_CAMPAGNE,
        help=f"Campagne jugée (défaut : {GATED_CAMPAGNE}).",
    )
    parser.add_argument("--juge", default="local", choices=sorted(JUDGES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit", type=int, default=None, help="Nombre de paires à juger au plus."
    )
    args = parser.parse_args(argv)

    if args.campagne == GATED_CAMPAGNE and not _prediction_committed():
        print(
            f"Jugement refusé : {PREDICTION_PATH} n'est pas dans le dernier "
            "commit (.claude/rules/methodologie.md)."
        )
        raise SystemExit(1)

    pairs = load_pairs(_pairs_path(args.campagne))
    judgments_path = _judgments_path(args.campagne, args.juge)
    existing = _load_existing_judgments(judgments_path)

    call_fn = JUDGES[args.juge]

    mlflow.set_experiment(args.campagne)
    with mlflow.start_run(run_name=f"juge-{args.juge}") as run:
        new_judgments = judge_pairs(
            pairs, call_fn, set(existing), model=args.model, limit=args.limit
        )
        mlflow.log_param("model", args.model)
        mlflow.log_param("juge", args.juge)
        mlflow.log_metric("n_judgments", len(existing) + len(new_judgments))
        _log_judgment_traces(run.info.run_id, new_judgments)

    existing.update({j["pair_id"]: j for j in new_judgments})
    _write_judgments(judgments_path, existing)

    _print_summary(new_judgments)


if __name__ == "__main__":
    main()
