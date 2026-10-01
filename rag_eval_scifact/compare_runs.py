"""Point d'entrée CLI : compare deux runs de campagne par un test apparié.

Usage : python -m rag_eval_scifact.compare_runs <run_a.json[.gz]> <run_b.json[.gz]>
        python -m rag_eval_scifact.compare_runs a.json.gz b.json.gz --metric mrr --seed 0
"""

from __future__ import annotations

import argparse

from rag_eval_scifact.compare import SUPPORTED_METRICS, compare_runs, load_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="Fichier du premier run (.json ou .json.gz)")
    parser.add_argument("run_b", help="Fichier du second run (.json ou .json.gz)")
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        action="append",
        dest="metrics",
        help="Métrique comparée (répétable). Par défaut : toutes.",
    )
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    metrics = args.metrics or list(SUPPORTED_METRICS)

    name_a = run_a.get("run_name", args.run_a)
    name_b = run_b.get("run_name", args.run_b)
    print(f"{name_a} vs {name_b}")
    for metric in metrics:
        result = compare_runs(
            run_a, run_b, metric, n_permutations=args.n_permutations, seed=args.seed
        )
        print(
            f"  {metric:10s} diff moyenne = {result['mean_diff']:+.4f}   "
            f"p-value = {result['p_value']:.4f}   (N={result['n_permutations']})"
        )


if __name__ == "__main__":
    main()
