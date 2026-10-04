"""CLI : écrit le rapport comparant les juges d'une campagne (EXE-122).

Usage : python -m rag_eval_scifact.run_judge_report
        python -m rag_eval_scifact.run_judge_report --campagne dev
"""

from __future__ import annotations

import argparse

from rag_eval_scifact.judge_report import write_report

DEFAULT_CAMPAGNE = "v3-juge"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campagne",
        default=DEFAULT_CAMPAGNE,
        help=f"Campagne de juges comparée (défaut : {DEFAULT_CAMPAGNE}).",
    )
    args = parser.parse_args(argv)

    path = write_report(args.campagne)
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
