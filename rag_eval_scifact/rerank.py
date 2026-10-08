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
    instruction: str = "",
    half_precision: bool = False,
) -> Callable[[list[tuple[str, str]]], list[float]]:
    """Encode les paires (claim, document) avec un `CrossEncoder` chargé paresseusement.

    `instruction`, quand non vide, remplace l'instruction par défaut du modèle
    (prompt du `CrossEncoder` — EXE-157 critère 5) ; vide, aucun prompt n'est
    passé. `half_precision` charge le modèle en demi-précision (EXE-157
    critère 6).
    """
    model_box: dict[str, object] = {}

    def _score(pairs: list[tuple[str, str]]) -> list[float]:
        if "model" not in model_box:
            from sentence_transformers import CrossEncoder

            init_kwargs = {}
            if half_precision:
                init_kwargs["model_kwargs"] = {"torch_dtype": "float16"}
            model_box["model"] = CrossEncoder(model_name, **init_kwargs)
        predict_kwargs = {"prompt": instruction} if instruction else {}
        scores = model_box["model"].predict(list(pairs), **predict_kwargs)
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
    instruction: str = "",
    half_precision: bool = False,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> tuple[list[RetrievalResult], float]:
    """Reclasse chaque requête d'un run et mesure la durée totale du reclassement.

    `score_fn` injectable (tests, EXE-96 critère 8) : aucun modèle cross-encoder
    n'est chargé tant qu'il n'est pas fourni par l'appelant. `instruction` et
    `half_precision` (EXE-157 critères 5, 6) n'affectent que le scorer par
    défaut — ignorés quand `score_fn` est fourni.
    """
    score = score_fn or _default_cross_encoder_scorer(
        model_name, instruction, half_precision
    )

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


def essai_rerank_campaign(
    results: list[RetrievalResult],
    doc_texts: dict[str, str],
    top_n: int,
    model_name: str = DEFAULT_RERANK_MODEL,
    instruction: str = "",
    half_precision: bool = False,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    sample_size: int = 10,
    n_claims: int = 300,
) -> dict[str, float]:
    """Essai rapide (EXE-157 critère 7) : reclasse les `top_n` premiers documents
    des `sample_size` premières requêtes de `results`, chronomètre et extrapole
    à `n_claims`.

    `results` : classement de premier étage déjà construit (cache), inchangé
    par cet essai. N'écrit jamais de fichier de résultats ni de run MLflow.
    """
    sample = results[:sample_size]
    _, duration_seconds = rerank_campaign_results(
        sample,
        doc_texts,
        top_n=top_n,
        model_name=model_name,
        instruction=instruction,
        half_precision=half_precision,
        score_fn=score_fn,
    )

    n_sample = len(sample) or 1
    extrapolated_minutes = duration_seconds * (n_claims / n_sample) / 60

    return {
        "duration_seconds": duration_seconds,
        "extrapolated_minutes": extrapolated_minutes,
    }
