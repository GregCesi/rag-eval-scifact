"""CLI : écrit le fichier des paires à juger d'une campagne (EXE-119).

Usage : python -m rag_eval_scifact.run_judge_pairs
        python -m rag_eval_scifact.run_judge_pairs --campagne dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval_scifact.compare import load_run
from rag_eval_scifact.ingest import CORPUS_PATH, load_corpus
from rag_eval_scifact.judge_pairs import build_pairs, write_pairs

V2_GRID_DIR = Path("results/v2-grid")
REFERENCE_RUN_NAME = "dense-qwen3-passages-sans-reranker"
DEFAULT_CAMPAGNE = "v3-juge"


def _load_grid_runs() -> list[dict]:
    return [load_run(p) for p in sorted(V2_GRID_DIR.glob("*.json.gz"))]


def _load_reference_run() -> dict:
    matches = sorted(V2_GRID_DIR.glob(f"{REFERENCE_RUN_NAME}-*.json.gz"))
    if not matches:
        raise FileNotFoundError(
            f"run de référence introuvable : {V2_GRID_DIR}/{REFERENCE_RUN_NAME}-*.json.gz"
        )
    return load_run(matches[-1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campagne",
        default=DEFAULT_CAMPAGNE,
        help=f"Campagne destinataire du fichier de paires (défaut : {DEFAULT_CAMPAGNE}).",
    )
    args = parser.parse_args(argv)

    grid_runs = _load_grid_runs()
    reference_run = _load_reference_run()
    corpus = load_corpus(CORPUS_PATH)
    corpus_by_id = {doc["_id"]: doc for doc in corpus}

    pairs = build_pairs(grid_runs, reference_run, corpus_by_id)
    path = write_pairs(pairs, args.campagne)

    n_with_expected = sum(1 for p in pairs if p["document_attendu"])
    n_claims = len({p["claim_id"] for p in pairs})
    print(
        f"{path} : {n_claims} claims, {len(pairs)} paires "
        f"({n_with_expected} avec document attendu, {len(pairs) - n_with_expected} sans)"
    )


if __name__ == "__main__":
    main()
