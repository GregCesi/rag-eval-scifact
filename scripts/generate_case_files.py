"""Genere les fiches markdown de cas pour l'annotation d'erreurs de retrieval.

Produit un fichier par query dans results/{bucket}_cases/query_{id}.md,
au format attendu par le skill /annotate (sous-agents annotateurs).

Usage : python scripts/generate_case_files.py <bucket> [--query-ids 13,70,128]
        bucket : near_miss | deep_miss | miss_100
"""

import argparse
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


def split_at_truncation(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> tuple[str, str]:
    """Decoupe le texte a la frontiere de troncature du tokenizer.

    Retourne (partie_vue, partie_ignoree).
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text, ""
    seen = tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
    lost = tokenizer.decode(encoded[max_tokens:], skip_special_tokens=True)
    return seen, lost


def text_with_cut(text: str, tokenizer) -> str:
    """Retourne le texte avec marqueur de coupe en texte brut."""
    seen, lost = split_at_truncation(text, tokenizer)
    if not lost:
        return text
    return f"{seen}\n\n--- ✂ coupe {MAX_TOKENS} tokens ---\n\n{lost}"


def generate_case_md(q: dict, corpus: dict, tokenizer) -> str:
    """Genere le markdown d'une query pour annotation par sous-agent."""
    lines = []
    rank = q["per_query_metrics"]["best_rank"]
    expected_ids = {d["doc_id"] for d in q["expected_docs"]}

    lines.append(f"## Query {q['query_id']}")
    lines.append(f"**Question** : {q['query_text']}")
    lines.append(f"**Meilleur rang** : {rank if rank else 'NON TROUVE dans le top 100'}")
    lines.append("")

    lines.append("### Document(s) attendu(s)")
    lines.append("")
    for ed in q["expected_docs"]:
        doc_data = corpus.get(ed["doc_id"], {})
        tc = ed["token_count"]
        status = "TRONQUE" if tc > MAX_TOKENS else "COMPLET"
        lines.append(f"**{doc_data.get('title', '?')}** (`{ed['doc_id']}`, {tc} tokens, {status})")
        lines.append("")
        lines.append(text_with_cut(doc_data.get("text", ""), tokenizer))
        lines.append("")

    lines.append("### Top-5 retrouve")
    lines.append("")
    for d in q["retrieved_top100"][:5]:
        doc_data = corpus.get(d["doc_id"], {})
        text = doc_data.get("text", "")
        seen, lost = split_at_truncation(text, tokenizer)
        status = "TRONQUE" if lost else "COMPLET"
        hit = " ✅ PERTINENT" if d["doc_id"] in expected_ids else ""
        lines.append(
            f"**Rang {d['rank']}** (score {d['score']:.4f}, {status}) "
            f"— **{doc_data.get('title', '?')}** (`{d['doc_id']}`){hit}"
        )
        lines.append("")
        lines.append(text_with_cut(text, tokenizer))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Genere les fiches de cas pour annotation")
    parser.add_argument("bucket", choices=["near_miss", "deep_miss", "miss_100"],
                        help="Bucket de queries a traiter")
    parser.add_argument("--query-ids", type=str, default=None,
                        help="IDs de queries specifiques (separes par des virgules)")
    args = parser.parse_args()

    # Charger les buckets
    buckets_path = RESULTS_DIR / "v1-buckets.json"
    if not buckets_path.exists():
        print(f"Erreur : {buckets_path} introuvable. Lancer extract_error_analysis.py d'abord.")
        sys.exit(1)
    with open(buckets_path) as f:
        buckets = json.load(f)

    # Trouver les queries du bucket
    query_ids = []
    for qid, info in buckets["queries"].items():
        if info["bucket"] == args.bucket:
            query_ids.append(qid)
    query_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    # Filtrer si --query-ids
    if args.query_ids:
        requested = set(args.query_ids.split(","))
        query_ids = [qid for qid in query_ids if qid in requested]

    if not query_ids:
        print(f"Aucune query dans le bucket '{args.bucket}'.")
        sys.exit(0)

    # Charger le run le plus recent
    run_files = sorted(RESULTS_DIR.glob("v1-dense-*.json"), reverse=True)
    if not run_files:
        print("Erreur : aucun fichier v1-dense-*.json dans results/")
        sys.exit(1)
    run_path = run_files[0]
    print(f"Run : {run_path}")
    with open(run_path) as f:
        run_data = json.load(f)

    queries_by_id = {q["query_id"]: q for q in run_data["queries"]}

    # Charger corpus et tokenizer
    print("Chargement corpus...")
    corpus = load_corpus()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generer les fiches
    out_dir = RESULTS_DIR / f"{args.bucket}_cases"
    out_dir.mkdir(exist_ok=True)

    count = 0
    for qid in query_ids:
        q = queries_by_id.get(qid)
        if not q:
            print(f"  Warning : query {qid} absente du run, skipped")
            continue
        md = generate_case_md(q, corpus, tokenizer)
        out_file = out_dir / f"query_{qid}.md"
        with open(out_file, "w") as f:
            f.write(md)
        count += 1

    print(f"\n{count} fiches generees dans {out_dir}/")


if __name__ == "__main__":
    main()
