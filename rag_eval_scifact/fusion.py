"""Fusion de deux classements de retrieval, dense et BM25 (EXE-95).

Codée à la main — aucune lib de fusion (`ranx` et consorts) :
`.claude/rules/methodologie.md`. Les deux fonctions combinent, pour une seule
requête, deux listes déjà classées (`retrieved`, triées par rang croissant,
rang démarrant à 1 — contrat `RetrievalResult`) en une liste fusionnée triée,
rang recalculé à partir de 1.
"""

from __future__ import annotations

from itertools import zip_longest

# Nombre de candidats de chaque sous-retriever que la fusion regarde (EXE-95
# critères 2 et 3). Fixe, contrairement à `k` (RRF) qui est une valeur de
# configuration : le ticket ne rend que `k` configurable.
UNION_SLICE = 50
CANDIDATE_POOL = 100


def union_fuse(dense: list[dict], bm25: list[dict], top_k: int = 100) -> list[dict]:
    """Union simple par alternance (critère 2) : 1er dense, 1er BM25, 2e dense, ...

    Ne regarde que les `UNION_SLICE` premiers de chaque liste. Un document
    déjà placé (par l'autre retriever ou un rang antérieur) est sauté.
    `top_k` documents au plus.
    """
    seen: set[str] = set()
    fused: list[dict] = []
    for dense_entry, bm25_entry in zip_longest(dense[:UNION_SLICE], bm25[:UNION_SLICE]):
        for entry in (dense_entry, bm25_entry):
            if len(fused) >= top_k:
                break
            if entry is None or entry["doc_id"] in seen:
                continue
            seen.add(entry["doc_id"])
            fused.append({"doc_id": entry["doc_id"], "score": entry["score"]})
        if len(fused) >= top_k:
            break

    for rank, entry in enumerate(fused, start=1):
        entry["rank"] = rank
    return fused


def rrf_fuse(
    dense: list[dict], bm25: list[dict], k: int = 60, top_k: int = 100
) -> list[dict]:
    """Fusion RRF (critère 3) : score = somme de 1/(k + rang) sur les
    `CANDIDATE_POOL` premiers de chaque retriever."""
    scores: dict[str, float] = {}
    for ranked in (dense[:CANDIDATE_POOL], bm25[:CANDIDATE_POOL]):
        for entry in ranked:
            scores[entry["doc_id"]] = scores.get(entry["doc_id"], 0.0) + 1.0 / (
                k + entry["rank"]
            )

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {"doc_id": doc_id, "rank": rank, "score": score}
        for rank, (doc_id, score) in enumerate(ordered, start=1)
    ]
