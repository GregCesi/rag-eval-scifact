"""CLI : écrit le rapport de v2-grid par catégorie d'étiquette d'origine (EXE-124).

Usage : python -m rag_eval_scifact.run_category_report
"""

from __future__ import annotations

from rag_eval_scifact.category_report import write_report


def main(argv: list[str] | None = None) -> None:
    path = write_report()
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
