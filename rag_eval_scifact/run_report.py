"""Point d'entrée CLI : écrit le rapport d'une campagne.

Usage : python -m rag_eval_scifact.run_report <campagne>
        python -m rag_eval_scifact.run_report v2-grid
"""

from __future__ import annotations

import argparse

from rag_eval_scifact.report import write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campagne", help="Nom de la campagne (results/<campagne>/).")
    args = parser.parse_args(argv)

    path = write_report(args.campagne)
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
