"""CLI : rédige un faux résumé d'article (HyDE) pour chacune des 300 affirmations
du jeu de test (EXE-141).

Écrit `results/v4-leviers/hyde.json`. Reprise sur une relance interrompue : les
affirmations déjà rédigées ne sont pas rédigées à nouveau.

Usage : python -m rag_eval_scifact.run_hyde
        python -m rag_eval_scifact.run_hyde --limit 25
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rag_eval_scifact.hyde import (
    DEFAULT_MODEL,
    HYDE_PATH,
    call_ollama,
    write_hyde_texts,
)
from rag_eval_scifact.retrieve import QRELS_PATH, QUERIES_PATH, load_test_queries


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        entry["claim_id"]: entry
        for entry in json.loads(path.read_text(encoding="utf-8"))
    }


def _write_entries(path: Path, entries_by_id: dict[str, dict]) -> None:
    """Écriture atomique (fichier temporaire puis renommage) : une interruption
    (Ctrl+C) laisse le fichier déjà écrit intact et lisible (critère 4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(list(entries_by_id.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _print_summary(entries: list[dict]) -> None:
    if not entries:
        print("Aucun nouveau texte.")
        return
    avg = statistics.mean(e["duration_seconds"] for e in entries)
    print(f"{len(entries)} nouveau(x) texte(s), durée moyenne : {avg:.2f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modèle Ollama (défaut : {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre d'affirmations à rédiger au plus.",
    )
    args = parser.parse_args(argv)

    claims, _ = load_test_queries(QUERIES_PATH, QRELS_PATH)
    existing = _load_existing(HYDE_PATH)
    already_written_ids = set(existing)

    def _on_text(entry: dict) -> None:
        # Écrit chaque texte dès qu'il est produit (critère 4) : une
        # interruption garde tout ce qui est déjà rédigé.
        existing[entry["claim_id"]] = entry
        _write_entries(HYDE_PATH, existing)

    new_entries = write_hyde_texts(
        claims,
        call_ollama,
        already_written_ids,
        model=args.model,
        limit=args.limit,
        on_text=_on_text,
    )
    _print_summary(new_entries)


if __name__ == "__main__":
    main()
