"""Métriques d'évaluation fait-main pour le retrieval dense.

Recall@k, nDCG@k, MRR — implémentées from scratch (interdiction de libs d'éval).
Chaque fonction opère au niveau d'une requête unique.
L'agrégation macro-average est faite par l'appelant (run_eval).
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k pour une requête.

    Recall@k = |{docs pertinents ∩ top-k}| / |{docs pertinents}|
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return found / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG@k pour une requête (relevance binaire).

    DCG@k  = Σᵢ₌₁ᵏ relᵢ / log₂(i+1)   — rang 1 → log₂(2) = 1
    IDCG@k = DCG du classement parfait (docs pertinents en premier)
    nDCG@k = DCG / IDCG  (0.0 si IDCG = 0)
    """
    if not relevant_ids:
        return 0.0

    # DCG sur le ranking réel
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG : classement parfait — les min(|relevant|, k) docs pertinents aux premiers rangs
    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(1, n_relevant_in_k + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR (Mean Reciprocal Rank) pour une requête.

    1/rang du premier doc pertinent trouvé dans la liste.
    Retourne 0.0 si aucun doc pertinent n'est trouvé.

    Note : l'appelant fournit le top-100. Un doc pertinent au-delà du rang 100
    contribue 0 au MRR — cohérent avec la profondeur de log top-100.
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
