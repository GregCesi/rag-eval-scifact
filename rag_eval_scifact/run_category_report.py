"""CLI : écrit le rapport par catégorie d'étiquette d'origine (EXE-124),
généralisé à une campagne nommée et à un run de référence (EXE-142).

Usage : python -m rag_eval_scifact.run_category_report
        python -m rag_eval_scifact.run_category_report --campagne v4-leviers
        python -m rag_eval_scifact.run_category_report --campagne v4-leviers --reference qwen3-passages-reference
"""

from __future__ import annotations

import argparse

from rag_eval_scifact.category_report import (
    NoRunsFound,
    write_report,
    write_report_for_campaign,
)
from rag_eval_scifact.report import InvalidRunFile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campagne",
        default=None,
        help="Campagne dont le rapport est écrit (défaut : v2-grid, à l'identique).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Run de référence : ajoute, pour chaque autre run, l'écart de "
            "nDCG@10 et sa p-value face à ce run, sur les 188 claims avec preuve."
        ),
    )
    args = parser.parse_args(argv)

    if args.campagne is None and args.reference is None:
        try:
            path = write_report()
        except InvalidRunFile as exc:
            print(str(exc))
            raise SystemExit(1)
        print(f"Rapport écrit : {path}")
        return

    campagne = args.campagne or "v2-grid"
    try:
        path = write_report_for_campaign(campagne, reference_run=args.reference)
    except NoRunsFound:
        print(
            f"Campagne « {campagne} » : aucun run sous results/{campagne}/. "
            "Aucun rapport écrit."
        )
        raise SystemExit(1)
    except InvalidRunFile as exc:
        print(str(exc))
        raise SystemExit(1)
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
