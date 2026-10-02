"""Reranker cross-encoder : reclasse les N premiers documents d'un classement existant.

Un cross-encoder relit chaque paire (claim, document) ensemble — plus coûteux
qu'un bi-encoder, donc borné aux `top_n` premiers documents d'un classement de
premier étage déjà construit par `rag_eval_scifact.retrieve.retrieve_campaign`.
Appliqué après coup, jamais à l'intérieur de ce premier étage : son cache
(embeddings, classement) reste inchangé, reranker actif ou non (EXE-96
critère 4). `score_fn` est injectable (tests) : aucun modèle cross-encoder
n'est chargé tant qu'il n'est pas fourni par l'appelant.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rag_eval_scifact.retrieve import RetrievalResult

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_N = 100


def _default_cross_encoder_scorer(
    model_name: str,
) -> Callable[[list[tuple[str, str]]], list[float]]:
    """Encode les paires (claim, document) avec un `CrossEncoder` chargé paresseusement."""
    model_box: dict[str, object] = {}

    def _score(pairs: list[tuple[str, str]]) -> list[float]:
        if "model" not in model_box:
            from sentence_transformers import CrossEncoder

            model_box["model"] = CrossEncoder(model_name)
        scores = model_box["model"].predict(list(pairs))
        return [float(s) for s in scores]

    return _score


def rerank_ranking(
    query_text: str,
    retrieved: list[dict],
    doc_texts: dict[str, str],
    top_n: int,
    score_fn: Callable[[list[tuple[str, str]]], list[float]],
) -> list[dict]:
    """Reclasse les `top_n` premiers documents de `retrieved` par score cross-encoder.

    `retrieved` : liste triée {"doc_id", "rank", "score"} (contrat
    `RetrievalResult`). Chaque paire reclassée est (`query_text`, `doc_texts`
    du document) — titre + texte, au niveau document, quelle que soit l'unité
    du premier étage (EXE-96 critère 2). Les documents au-delà de `top_n`
    gardent leur rang relatif, inchangé, à la suite des documents reclassés :
    le résultat contient exactement les mêmes documents que `retrieved`
    (EXE-96 critère 3).
    """
    head = retrieved[:top_n]
    tail = retrieved[top_n:]

    pairs = [(query_text, doc_texts.get(d["doc_id"], "")) for d in head]
    scores = score_fn(pairs) if pairs else []
    reranked_head = sorted(zip(head, scores), key=lambda item: item[1], reverse=True)

    ordered_docs = [d for d, _ in reranked_head] + tail
    ordered_scores = [score for _, score in reranked_head] + [d["score"] for d in tail]

    return [
        {"doc_id": doc["doc_id"], "rank": rank, "score": score}
        for rank, (doc, score) in enumerate(zip(ordered_docs, ordered_scores), start=1)
    ]


def rerank_campaign_results(
    results: list[RetrievalResult],
    doc_texts: dict[str, str],
    top_n: int = DEFAULT_RERANK_TOP_N,
    model_name: str = DEFAULT_RERANK_MODEL,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> tuple[list[RetrievalResult], float]:
    """Reclasse chaque requête d'un run et mesure la durée totale du reclassement.

    `score_fn` injectable (tests, EXE-96 critère 8) : aucun modèle cross-encoder
    n'est chargé tant qu'il n'est pas fourni par l'appelant.
    """
    score = score_fn or _default_cross_encoder_scorer(model_name)

    start = time.perf_counter()
    reranked = [
        RetrievalResult(
            query_id=r.query_id,
            query_text=r.query_text,
            retrieved=rerank_ranking(
                r.query_text, r.retrieved, doc_texts, top_n, score
            ),
        )
        for r in results
    ]
    duration = time.perf_counter() - start
    return reranked, duration
