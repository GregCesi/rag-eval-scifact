"""CLI : écrit le fichier des étiquettes d'origine par paire (claim, document) (EXE-124).

Usage : python -m rag_eval_scifact.run_origin_labels
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag_eval_scifact.origin_labels import (
    build_origin_labels_document,
    load_claims,
    load_origin_corpus,
    load_qrels_pairs,
)

CLAIMS_PATH = Path("data/scifact/origine/data/claims_dev.jsonl")
ORIGIN_CORPUS_PATH = Path("data/scifact/origine/data/corpus.jsonl")
QRELS_PATH = Path("data/scifact/qrels/test.tsv")
OUTPUT_PATH = Path("results/etiquettes-origine.json")


def main(argv: list[str] | None = None) -> None:
    claims = load_claims(CLAIMS_PATH)
    corpus = load_origin_corpus(ORIGIN_CORPUS_PATH)
    qrels_pairs = load_qrels_pairs(QRELS_PATH)

    document = build_origin_labels_document(claims, corpus, qrels_pairs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    label_counts = Counter(p["label"] for p in document["pairs"])
    categorie_counts = Counter(c["categorie"] for c in document["claims"])
    print(
        f"{OUTPUT_PATH} : {len(document['pairs'])} paires {dict(label_counts)}, "
        f"{len(document['claims'])} claims {dict(categorie_counts)}"
    )


if __name__ == "__main__":
    main()
