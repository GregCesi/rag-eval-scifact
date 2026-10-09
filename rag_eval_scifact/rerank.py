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

import re
import time
from collections.abc import Callable

from rag_eval_scifact.retrieve import RetrievalResult

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_N = 100
# Défaut de `CrossEncoder.predict` (sentence-transformers) : explicité ici pour
# que la nouvelle clé `rerank.batch_size` (EXE-159 critère 3) reproduise à
# l'identique le comportement des runs déjà faits qui ne la posent pas.
DEFAULT_RERANK_BATCH_SIZE = 32

_REQUESTED_MEMORY_RE = re.compile(r"[Tt]ried to allocate ([\d.]+\s*\w+)")
_AVAILABLE_MEMORY_RE = re.compile(r"([\d.]+\s*\w+) free")
_MAX_ALLOWED_MEMORY_RE = re.compile(r"max allowed: ([\d.]+\s*\w+)")


def _load_cross_encoder(model_name: str, half_precision: bool = False):
    """Charge le `CrossEncoder` — isolé pour être appelé directement par les
    tests (EXE-159 critère 1) sur le modèle réellement construit, pas sur les
    arguments passés."""
    from sentence_transformers import CrossEncoder

    init_kwargs = {}
    if half_precision:
        init_kwargs["model_kwargs"] = {"torch_dtype": "float16"}
    return CrossEncoder(model_name, **init_kwargs)


def _accelerator_memory_gb() -> float:
    """Mémoire occupée sur l'accélérateur actif, en Go (EXE-159 critère 2)."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def is_rerank_out_of_memory(exc: BaseException) -> bool:
    """`True` si `exc` est le `RuntimeError` PyTorch de dépassement mémoire
    (EXE-159 critère 5), quel que soit l'accélérateur (MPS, CUDA)."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def format_rerank_oom_message(run_name: str, exc: RuntimeError) -> str:
    """Phrase de remplacement de la trace Python brute (EXE-159 critère 5) :
    nomme le run, reprend la mémoire demandée et disponible rapportées par
    `exc`, et rappelle la clé qui permet de réduire la charge."""
    message = str(exc)
    requested = _REQUESTED_MEMORY_RE.search(message)
    available = _AVAILABLE_MEMORY_RE.search(message) or _MAX_ALLOWED_MEMORY_RE.search(
        message
    )
    requested_txt = requested.group(1) if requested else "quantité non rapportée"
    available_txt = available.group(1) if available else "limite non rapportée"
    return (
        f"{run_name} : mémoire insuffisante pour le reranker "
        f"(demandé {requested_txt}, disponible {available_txt}). "
        "Réduire la clé rerank.batch_size."
    )


def _default_cross_encoder_scorer(
    model_name: str,
    instruction: str = "",
    half_precision: bool = False,
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
) -> Callable[[list[tuple[str, str]]], list[float]]:
    """Encode les paires (claim, document) avec un `CrossEncoder` chargé paresseusement.

    `instruction`, quand non vide, remplace l'instruction par défaut du modèle
    (prompt du `CrossEncoder` — EXE-157 critère 5) ; vide, aucun prompt n'est
    passé. `half_precision` charge le modèle en demi-précision (EXE-157
    critère 6). `batch_size` (EXE-159 critère 3) borne la mémoire d'activation
    consommée par `CrossEncoder.predict`.
    """
    model_box: dict[str, object] = {}

    def _score(pairs: list[tuple[str, str]]) -> list[float]:
        if "model" not in model_box:
            model_box["model"] = _load_cross_encoder(model_name, half_precision)
        predict_kwargs: dict[str, object] = {"batch_size": batch_size}
        if instruction:
            predict_kwargs["prompt"] = instruction
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
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> tuple[list[RetrievalResult], float]:
    """Reclasse chaque requête d'un run et mesure la durée totale du reclassement.

    `score_fn` injectable (tests, EXE-96 critère 8) : aucun modèle cross-encoder
    n'est chargé tant qu'il n'est pas fourni par l'appelant. `instruction`,
    `half_precision` et `batch_size` (EXE-157 critères 5, 6 ; EXE-159 critère 3)
    n'affectent que le scorer par défaut — ignorés quand `score_fn` est fourni.
    """
    score = score_fn or _default_cross_encoder_scorer(
        model_name, instruction, half_precision, batch_size
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
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    sample_size: int = 10,
    n_claims: int = 300,
) -> dict[str, float]:
    """Essai rapide (EXE-157 critère 7) : reclasse les `top_n` premiers documents
    des `sample_size` premières requêtes de `results`, chronomètre et extrapole
    à `n_claims`.

    `results` : classement de premier étage déjà construit (cache), inchangé
    par cet essai. N'écrit jamais de fichier de résultats ni de run MLflow.
    Rapporte aussi la mémoire occupée sur l'accélérateur à la fin de l'essai,
    en Go (EXE-159 critère 2).
    """
    sample = results[:sample_size]
    _, duration_seconds = rerank_campaign_results(
        sample,
        doc_texts,
        top_n=top_n,
        model_name=model_name,
        instruction=instruction,
        half_precision=half_precision,
        batch_size=batch_size,
        score_fn=score_fn,
    )

    n_sample = len(sample) or 1
    extrapolated_minutes = duration_seconds * (n_claims / n_sample) / 60

    return {
        "duration_seconds": duration_seconds,
        "extrapolated_minutes": extrapolated_minutes,
        "memory_gb": _accelerator_memory_gb(),
    }
