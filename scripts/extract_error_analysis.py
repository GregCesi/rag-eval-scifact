"""Bucketise les queries d'un run v1 par profondeur d'echec.

Sorties :
  results/v1-buckets.json   — mapping query_id -> bucket + rang + score
  results/v1-error-analysis-miss100.md — fiche d'annotation par query miss_100

Usage : python scripts/extract_error_analysis.py [results/v1-dense-*.json]
        (defaut : dernier fichier JSON dans results/)
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

RESULTS_DIR = Path("results")
CORPUS_PATH = Path("data/scifact/corpus.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


def load_corpus() -> dict[str, dict]:
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc["title"], "text": doc["text"]}
    return corpus


def classify_query(q: dict) -> tuple[str, int | None, float | None]:
    """Retourne (bucket, best_rank, score_at_best_rank)."""
    best_rank = q["per_query_metrics"]["best_rank"]
    if best_rank is None:
        return "miss_100", None, None
    # Trouver le score au rang du doc pertinent
    score = None
    for doc in q["retrieved_top100"]:
        if doc["rank"] == best_rank:
            score = doc["score"]
            break
    if best_rank == 1:
        return "perfect", best_rank, score
    if best_rank <= 10:
        return "near_miss", best_rank, score
    return "deep_miss", best_rank, score


def insert_truncation_marker(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> str:
    """Insere le marqueur de coupe a l'endroit exact de troncature."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text
    seen = tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
    lost = tokenizer.decode(encoded[max_tokens:], skip_special_tokens=True)
    return f"{seen}\n--- ✂ coupe {max_tokens} tokens ---\n{lost}"


def build_buckets(run_data: dict) -> dict:
    """Construit le mapping query_id -> bucket info."""
    buckets_map = {}
    counts = {"perfect": 0, "near_miss": 0, "deep_miss": 0, "miss_100": 0}

    for q in run_data["queries"]:
        bucket, rank, score = classify_query(q)
        counts[bucket] += 1
        buckets_map[q["query_id"]] = {
            "bucket": bucket,
            "best_rank": rank,
            "score": score,
        }

    return {"counts": counts, "queries": buckets_map}


def build_miss100_md(run_data: dict, corpus: dict, tokenizer) -> str:
    """Genere le fichier .md d'annotation pour les queries miss_100."""
    miss_queries = []
    for q in run_data["queries"]:
        bucket, _, _ = classify_query(q)
        if bucket == "miss_100":
            miss_queries.append(q)

    lines = [
        f"# Error analysis — miss_100 ({len(miss_queries)} queries)",
        "",
        "Queries dont aucun doc pertinent n'apparait dans le top 100.",
        "Annotation en deux passes : open coding puis categorisation.",
        "",
    ]

    for q in miss_queries:
        qid = q["query_id"]
        lines.append(f"## query_id: {qid}")
        lines.append("")
        lines.append(f"**Query** : {q['query_text']}")
        lines.append("")

        # Docs attendus (qrels)
        lines.append("### Docs attendus")
        lines.append("")
        for ed in q["expected_docs"]:
            doc_data = corpus.get(ed["doc_id"])
            if not doc_data:
                lines.append(f"- `{ed['doc_id']}` — *non trouve dans le corpus*")
                continue
            lines.append(f"**{doc_data['title']}** (`{ed['doc_id']}`, {ed['token_count']} tokens)")
            lines.append("")
            abstract_with_marker = insert_truncation_marker(doc_data["text"], tokenizer)
            lines.append(abstract_with_marker)
            lines.append("")

        # Top-5 retrouve
        lines.append("### Top-5 retrouve")
        lines.append("")
        for doc in q["retrieved_top100"][:5]:
            doc_data = corpus.get(doc["doc_id"])
            title = doc_data["title"] if doc_data else "?"
            abstract_preview = doc_data["text"][:200] if doc_data else ""
            lines.append(
                f"- **Rang {doc['rank']}** (score {doc['score']:.4f}) — "
                f"**{title}** (`{doc['doc_id']}`)"
            )
            lines.append(f"  {abstract_preview}...")
            lines.append("")

        # Champs annotation
        lines.append("categorie:")
        lines.append("note:")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    # Determiner le fichier de run
    if len(sys.argv) > 1:
        run_path = Path(sys.argv[1])
    else:
        jsons = sorted(RESULTS_DIR.glob("v1-dense-*.json"), reverse=True)
        if not jsons:
            print("Aucun fichier v1-dense-*.json trouve dans results/")
            sys.exit(1)
        run_path = jsons[0]

    print(f"Run : {run_path}")

    with open(run_path) as f:
        run_data = json.load(f)

    corpus = load_corpus()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Buckets JSON
    buckets = build_buckets(run_data)
    out_buckets = RESULTS_DIR / "v1-buckets.json"
    with open(out_buckets, "w") as f:
        json.dump(buckets, f, indent=2, ensure_ascii=False)
    print(f"Buckets : {out_buckets}")
    for bucket, count in buckets["counts"].items():
        print(f"  {bucket}: {count}")

    # Miss100 markdown
    md_content = build_miss100_md(run_data, corpus, tokenizer)
    out_md = RESULTS_DIR / "v1-error-analysis-miss100.md"
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"Annotation : {out_md} ({buckets['counts']['miss_100']} queries)")


if __name__ == "__main__":
    main()
