"""Sélection des paires (claim, document) à juger pour une campagne de juge LLM (EXE-119).

Un claim entre dans le jeu de paires quand les runs de v2-grid ne s'accordent
pas sur lui : au moins un ne l'a pas trouvé dans son top 10 (qu'un autre l'ait
trouvé ou non). Un claim où les 34 runs le trouvent tous dans leur top 10
n'apporte rien à juger : le retrieval n'y est pas en cause.

Pour un claim retenu, les documents à juger sont lus dans un seul run de
référence (`dense-qwen3-passages-sans-reranker`) : chaque document attendu,
plus les documents non attendus classés avant le premier document attendu
trouvé dans le top 100, 5 au plus ; à défaut de document attendu dans le top
100, les 5 premiers documents retournés.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")
MAX_NON_EXPECTED_PER_CLAIM = 5


def select_claims_to_judge(grid_runs: list[dict]) -> set[str]:
    """Claims pour lesquels found@10 n'est pas vrai sur tous les runs de la grille."""
    found_by_claim: dict[str, list[bool]] = {}
    for run in grid_runs:
        for query in run["queries"]:
            found_by_claim.setdefault(query["query_id"], []).append(
                query["per_query_metrics"]["found@10"]
            )
    return {
        claim_id
        for claim_id, found_flags in found_by_claim.items()
        if not all(found_flags)
    }


def _build_pair(
    claim_id: str,
    claim_text: str,
    doc_id: str,
    rank: int | None,
    document_attendu: bool,
    corpus_by_id: dict[str, dict],
) -> dict:
    doc = corpus_by_id[doc_id]
    return {
        "pair_id": f"{claim_id}:{doc_id}",
        "claim_id": claim_id,
        "claim_text": claim_text,
        "doc_id": doc_id,
        "doc_title": doc["title"],
        "doc_text": doc["text"],
        "rank": rank,
        "document_attendu": document_attendu,
    }


def _pairs_for_claim(query: dict, corpus_by_id: dict[str, dict]) -> list[dict]:
    """Paires d'un claim : ses documents attendus, puis les intrus classés avant."""
    claim_id = query["query_id"]
    claim_text = query["query_text"]
    expected_ids = [doc["doc_id"] for doc in query["expected_docs"]]
    rank_by_doc = {d["doc_id"]: d["rank"] for d in query["retrieved_top100"]}
    retrieved_sorted = sorted(query["retrieved_top100"], key=lambda d: d["rank"])
    best_rank = query["per_query_metrics"]["best_rank"]

    if best_rank is not None:
        non_expected = [
            d
            for d in retrieved_sorted
            if d["rank"] < best_rank and d["doc_id"] not in expected_ids
        ][:MAX_NON_EXPECTED_PER_CLAIM]
    else:
        non_expected = retrieved_sorted[:MAX_NON_EXPECTED_PER_CLAIM]

    pairs = [
        _build_pair(
            claim_id, claim_text, doc_id, rank_by_doc.get(doc_id), True, corpus_by_id
        )
        for doc_id in expected_ids
    ]
    pairs += [
        _build_pair(claim_id, claim_text, d["doc_id"], d["rank"], False, corpus_by_id)
        for d in non_expected
    ]
    return pairs


def build_pairs(
    grid_runs: list[dict],
    reference_run: dict,
    corpus_by_id: dict[str, dict],
) -> list[dict]:
    """Construit la liste ordonnée des paires à juger (critères 1 à 3)."""
    retained = select_claims_to_judge(grid_runs)
    queries_by_id = {q["query_id"]: q for q in reference_run["queries"]}
    pairs: list[dict] = []
    for claim_id in sorted(retained, key=int):
        pairs.extend(_pairs_for_claim(queries_by_id[claim_id], corpus_by_id))
    return pairs


def write_pairs(pairs: list[dict], campagne: str) -> Path:
    """Écrit `results/<campagne>/paires.json`."""
    path = RESULTS_DIR / campagne / "paires.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_pairs(path: Path) -> list[dict]:
    """Charge un fichier de paires déjà écrit par `write_pairs`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
