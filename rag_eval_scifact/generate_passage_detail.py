"""CLI : fichier dérivé des passages d'un run en unité passages (EXE-112).

Pour un run dense ou BM25 en unité passages, sans reranker, recalcule le
score de chaque passage de chaque document attendu et de son top 10, en
rejouant le découpage et le regroupement exacts du run (critère « ce qui ne
doit pas arriver » : aucun découpage approché).

Les embeddings de documents du run restent ceux mis en cache par
`retrieve_campaign` lors de son lancement : ce module les lit, il ne les
recalcule jamais (H2 — un cache absent arrête sans réindexation). Les 300
requêtes du split sont ré-encodées (H3), ce qui charge le modèle du run.
BM25 n'a pas de cache (jamais mis en cache côté `retrieve_campaign` non
plus) : les passages sont réindexés, sans modèle d'embedding.

Usage : python -m rag_eval_scifact.generate_passage_detail results/v2-grid/<run>.json.gz
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_eval_scifact.bm25 import build_bm25_index, score_queries
from rag_eval_scifact.cache import document_embeddings_cache_path
from rag_eval_scifact.chunking import Passage, chunk_text, default_offsets_fn
from rag_eval_scifact.compare import load_run
from rag_eval_scifact.ingest import CORPUS_PATH, compute_dataset_hash, load_corpus
from rag_eval_scifact.passage_detail import (
    build_run_passage_detail,
    eligible_for_passage_detail,
    verify_max_grouping_scores,
    write_passage_detail,
)
from rag_eval_scifact.retrieve import QRELS_PATH, QUERIES_PATH, load_test_queries

DEFAULT_CACHE_DIR = Path("~/.cache/rag-eval-scifact")


def _default_query_embedder(
    model_name: str, max_seq_length: int, batch_size: int
) -> Callable[[list[str]], np.ndarray]:
    def embed(texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
        vectors = model.encode(texts, show_progress_bar=False, batch_size=batch_size)
        return np.asarray(vectors, dtype=np.float32)

    return embed


def _bm25_scorer(
    passages: list[Passage],
    query_text_by_id: dict[str, str],
    run_queries: list[dict],
    k1: float,
    b: float,
) -> Callable[[dict, list[Passage]], list[float]]:
    passage_texts = [p.text for p in passages]
    passage_index = {p.passage_id: i for i, p in enumerate(passages)}
    print(f"  [bm25] indexation de {len(passages)} passages (jamais mis en cache)...")
    index = build_bm25_index(passage_texts, k1, b)

    query_ids = [q["query_id"] for q in run_queries]
    query_texts = [query_text_by_id[qid] for qid in query_ids]
    sim_matrix = score_queries(index, query_texts)
    scores_by_qid = {qid: sim_matrix[i] for i, qid in enumerate(query_ids)}

    def score(query: dict, doc_passages: list[Passage]) -> list[float]:
        row = scores_by_qid[query["query_id"]]
        return [float(row[passage_index[p.passage_id]]) for p in doc_passages]

    return score


def _dense_scorer(
    passage_ids: list[str],
    embeddings: np.ndarray,
    query_text_by_id: dict[str, str],
    run_queries: list[dict],
    query_instruction: str,
    embed: Callable[[list[str]], np.ndarray],
) -> Callable[[dict, list[Passage]], list[float]]:
    """Même matmul que `retrieve._rank_top_k_grouped` (requêtes x tous les
    passages, en une fois) — pas une matmul par document : deux formes de
    matmul mathématiquement égales divergent dans leurs derniers bits en
    float32 (routine BLAS différente), ce qui suffit à casser l'égalité à la
    4e décimale (H4) sur quelques documents."""
    passage_index = {pid: i for i, pid in enumerate(passage_ids)}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    query_ids = [q["query_id"] for q in run_queries]
    query_texts = [query_text_by_id[qid] for qid in query_ids]
    if query_instruction:
        query_texts = [query_instruction + t for t in query_texts]
    print(f"  [dense] ré-encodage de {len(query_texts)} requêtes...")
    query_embeddings = np.asarray(embed(query_texts), dtype=np.float32)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / query_norms

    sim_matrix = query_embeddings_norm @ embeddings_norm.T
    scores_by_qid = {qid: sim_matrix[i] for i, qid in enumerate(query_ids)}

    def score(query: dict, doc_passages: list[Passage]) -> list[float]:
        row = scores_by_qid[query["query_id"]]
        return [float(row[passage_index[p.passage_id]]) for p in doc_passages]

    return score


def generate_passage_detail(
    run: dict,
    run_path: Path,
    cache_dir: Path,
    corpus_path: Path = CORPUS_PATH,
    queries_path: Path = QUERIES_PATH,
    qrels_path: Path = QRELS_PATH,
    offsets_fn: Callable[[str], list[tuple[int, int]]] | None = None,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
) -> Path | None:
    """Produit le fichier dérivé d'un run, ou None s'il n'est pas produit.

    `offsets_fn` et `embedder` sont injectables pour les tests (aucun
    tokenizer ni modèle réel n'est chargé tant qu'ils ne sont pas fournis) ;
    en production, le tokenizer et le modèle du run (`retriever.model`).
    """
    if not eligible_for_passage_detail(run):
        print(
            f"[{run['run_name']}] pas éligible au détail des passages "
            "(unité document, retriever hybride, ou reranker actif) — rien produit."
        )
        return None

    retriever = run["config"]["retriever"]

    dataset_hash = compute_dataset_hash(corpus_path)
    if dataset_hash != run["dataset_hash"]:
        print(
            "ARRÊT — porte de sortie : le dataset_hash du corpus actuel "
            f"({dataset_hash}) diffère de celui du run ({run['dataset_hash']})."
        )
        return None

    docs = load_corpus(corpus_path)
    doc_texts = {doc["_id"]: doc["title"] + " " + doc["text"] for doc in docs}
    offsets = offsets_fn or default_offsets_fn(retriever["model"])
    passages = [
        passage
        for doc_id, text in doc_texts.items()
        for passage in chunk_text(
            doc_id, text, retriever["chunk_size"], retriever["chunk_overlap"], offsets
        )
    ]

    queries, _ = load_test_queries(queries_path, qrels_path)
    query_text_by_id = {q["_id"]: q["text"] for q in queries}

    if retriever["name"] == "bm25":
        score_doc_passages_for_query = _bm25_scorer(
            passages,
            query_text_by_id,
            run["queries"],
            retriever["bm25_k1"],
            retriever["bm25_b"],
        )
    else:
        chunking_unit = (
            f"passages-{retriever['chunk_size']}-{retriever['chunk_overlap']}"
        )
        cache_path = document_embeddings_cache_path(
            cache_dir,
            dataset_hash,
            retriever["model"],
            retriever["max_seq_length"],
            chunking_unit,
        )
        if not cache_path.exists():
            print(
                f"ARRÊT — porte de sortie : cache d'embeddings absent ({cache_path}) "
                "— aucune réindexation n'est lancée."
            )
            return None
        print(f"  [cache embeddings] trouvé ({cache_path}) — aucun passage ré-embeddé.")
        data = np.load(cache_path, allow_pickle=False)
        passage_ids = [str(d) for d in data["doc_ids"]]
        embeddings = data["embeddings"].astype(np.float32)

        embed = embedder or _default_query_embedder(
            retriever["model"], retriever["max_seq_length"], retriever["batch_size"]
        )
        score_doc_passages_for_query = _dense_scorer(
            passage_ids,
            embeddings,
            query_text_by_id,
            run["queries"],
            retriever["query_instruction"],
            embed,
        )

    detail = build_run_passage_detail(run, passages, score_doc_passages_for_query)

    mismatches = verify_max_grouping_scores(run, detail)
    if mismatches:
        worst = max(mismatches, key=lambda m: m[2])
        print(
            f"ARRÊT — prémisse fausse : {len(mismatches)} document(s) où le "
            "meilleur score de passage ne correspond pas au score du run à la "
            f"4e décimale (écart le plus grand {worst[2]:.6f}, requête "
            f"{worst[0]}, document {worst[1]})."
        )
        return None

    return write_passage_detail(run_path, detail)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path, help="Fichier du run (.json.gz)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Cache des embeddings (défaut : ~/.cache/rag-eval-scifact)",
    )
    args = parser.parse_args(argv)

    run = load_run(args.run_path)
    result = generate_passage_detail(run, args.run_path, args.cache_dir.expanduser())
    if result is not None:
        print(f"Fichier dérivé écrit : {result}")


if __name__ == "__main__":
    main()
