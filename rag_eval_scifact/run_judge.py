"""CLI : juge les paires d'une campagne avec un juge déclaré (EXE-119, EXE-120, EXE-121).

`--juge` est pluriel depuis EXE-119 pour que les juges suivants s'ajoutent sans
renommer rien ; « claude » (EXE-120) et « etapes » (EXE-121) s'ajoutent à « local ».

Usage : python -m rag_eval_scifact.run_judge --campagne v3-juge
        python -m rag_eval_scifact.run_judge --campagne dev --juge claude --limit 10
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
from rag_eval_scifact.judge_claude import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
from rag_eval_scifact.judge_claude import call_claude_code
from rag_eval_scifact.judge_claude import judge_pairs as judge_pairs_claude
from rag_eval_scifact.judge_etapes import DEFAULT_MODEL as ETAPES_DEFAULT_MODEL
from rag_eval_scifact.judge_etapes import judge_pairs as judge_pairs_etapes
from rag_eval_scifact.judge_local import DEFAULT_MODEL as LOCAL_DEFAULT_MODEL
from rag_eval_scifact.judge_local import call_ollama
from rag_eval_scifact.judge_local import judge_pairs as judge_pairs_local
from rag_eval_scifact.judge_pairs import load_pairs

GATED_CAMPAGNE = "v3-juge"
PREDICTION_PATH = "results/v3-juge/PREDICTION.md"
RESULTS_DIR = Path("results")
OLD_CITATION_INTROUVABLE_VERDICT = "citation introuvable"

JUDGES = {
    "local": {
        "call_fn": call_ollama,
        "judge_pairs": judge_pairs_local,
        "default_model": LOCAL_DEFAULT_MODEL,
    },
    "claude": {
        "call_fn": call_claude_code,
        "judge_pairs": judge_pairs_claude,
        "default_model": CLAUDE_DEFAULT_MODEL,
    },
    "etapes": {
        "call_fn": call_ollama,
        "judge_pairs": judge_pairs_etapes,
        "default_model": ETAPES_DEFAULT_MODEL,
    },
}


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
    """Écriture atomique (fichier temporaire puis renommage) : une interruption
    pendant l'écriture laisse l'ancien fichier intact et lisible (EXE-131, H4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(list(judgments_by_id.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _already_judged_ids(existing: dict[str, dict]) -> set[str]:
    """Exclut les jugements « citation introuvable » (ancienne forme) : ils
    sont rejugés à la relance, le verdict qu'ils portaient ayant été perdu
    (EXE-131, critère 10)."""
    return {
        pair_id
        for pair_id, judgment in existing.items()
        if judgment.get("verdict") != OLD_CITATION_INTROUVABLE_VERDICT
    }


def _log_judgment_traces(run_id: str, judgments: list[dict]) -> None:
    """Une trace par jugement. Un juge à étapes (EXE-121 critère 10) ajoute une
    étape enfant par nœud de son graphe réellement appelé, avec son entrée et sa
    sortie ; les juges à un seul appel (local, claude) n'ajoutent aucune étape."""
    for j in judgments:
        with mlflow.start_span(
            name=f"judge-{j['pair_id']}", span_type=SpanType.LLM, run_id=run_id
        ) as span:
            span.set_inputs({"pair_id": j["pair_id"]})
            outputs = {"verdict": j["verdict"], "evidence": j["evidence"]}
            if "reason" in j:
                outputs["reason"] = j["reason"]
            if "cause" in j:
                outputs["cause"] = j["cause"]
            span.set_outputs(outputs)

            for step_name, step in j.get("steps", {}).items():
                with mlflow.start_span(
                    name=f"{step_name}-{j['pair_id']}", span_type=SpanType.LLM
                ) as step_span:
                    step_span.set_inputs(step["input"])
                    step_span.set_outputs(step["output"] or {})
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
    parser.add_argument(
        "--model",
        default=None,
        help="Modèle demandé (défaut : celui du juge choisi).",
    )
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
    already_judged_ids = _already_judged_ids(existing)

    judge_config = JUDGES[args.juge]
    model = args.model or judge_config["default_model"]
    call_fn = judge_config["call_fn"]
    judge_pairs_fn = judge_config["judge_pairs"]

    def _on_judgment(judgment: dict) -> None:
        # Écrit chaque jugement dès qu'il est produit (EXE-131, critère 2) :
        # une interruption garde tout ce qui a déjà été jugé.
        existing[judgment["pair_id"]] = judgment
        _write_judgments(judgments_path, existing)

    mlflow.set_experiment(args.campagne)
    with mlflow.start_run(run_name=f"juge-{args.juge}") as run:
        mlflow.log_param("model", model)
        mlflow.log_param("juge", args.juge)
        try:
            new_judgments = judge_pairs_fn(
                pairs,
                call_fn,
                already_judged_ids,
                model=model,
                limit=args.limit,
                on_judgment=_on_judgment,
            )
        finally:
            # Le nombre réellement gardé, même si l'appel ci-dessus a été
            # interrompu ou a échoué (EXE-131, critère 11).
            mlflow.log_metric("n_judgments", len(existing))
        _log_judgment_traces(run.info.run_id, new_judgments)

    _print_summary(new_judgments)


if __name__ == "__main__":
    main()
