"""Détail des passages d'un run en unité passages (EXE-112).

Pour chaque requête d'un run, liste les passages de chaque document attendu
et de son top 10, avec le score de chacun — ce que le fichier de run ne garde
pas (seul le score du document après regroupement y est écrit).

Logique pure : le découpage (`rag_eval_scifact.chunking`) et le score de
chaque passage pour une requête sont fournis par l'appelant
(`generate_passage_detail.py`, qui seul charge un modèle ou indexe BM25).
Jamais pour un run hybride (reclassement entre deux sous-retrievers) ni avec
reranker (ce tour ne couvre que les runs sans reranker, EXE-112 — ce qui ne
doit pas arriver).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path

from rag_eval_scifact.chunking import Passage

TOP_N_FOR_DETAIL = 10


def eligible_for_passage_detail(run: dict) -> bool:
    """Un run reçoit un fichier dérivé seulement s'il est en unité passages,
    pas hybride, et sans reranker (EXE-112, « ce qui ne doit pas arriver »)."""
    retriever = run["config"]["retriever"]
    rerank = run["config"].get("rerank")
    rerank_enabled = rerank is not None and rerank.get("name") == "cross-encoder"
    return (
        retriever["unit"] == "passages"
        and retriever["name"] != "hybrid"
        and not rerank_enabled
    )


def group_passages_by_doc(passages: list[Passage]) -> dict[str, list[Passage]]:
    """Passages d'un corpus, regroupés par document, dans leur ordre d'origine."""
    by_doc: dict[str, list[Passage]] = {}
    for p in passages:
        by_doc.setdefault(p.doc_id, []).append(p)
    return by_doc


def needed_doc_ids(query: dict) -> list[str]:
    """Docs attendus puis top 10, dans cet ordre, sans doublon.

    Un doc attendu absent du top 100 reste présent (critère 7) : il n'est
    jamais filtré ici, seulement classé en premier.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for d in query["expected_docs"]:
        if d["doc_id"] not in seen_set:
            seen.append(d["doc_id"])
            seen_set.add(d["doc_id"])
    for d in query["retrieved_top100"][:TOP_N_FOR_DETAIL]:
        if d["doc_id"] not in seen_set:
            seen.append(d["doc_id"])
            seen_set.add(d["doc_id"])
    return seen


def build_query_passage_detail(
    query: dict,
    passages_by_doc: dict[str, list[Passage]],
    score_doc_passages: Callable[[list[Passage]], list[float]],
) -> dict:
    """Détail d'une requête : pour chaque doc nécessaire, ses passages et leurs scores."""
    docs: dict[str, list[dict]] = {}
    for doc_id in needed_doc_ids(query):
        doc_passages = passages_by_doc.get(doc_id, [])
        scores = score_doc_passages(doc_passages)
        docs[doc_id] = [
            {
                "passage_id": p.passage_id,
                "char_start": p.char_start,
                "char_end": p.char_end,
                "token_count": p.token_count,
                "score": float(score),
            }
            for p, score in zip(doc_passages, scores)
        ]
    return {"query_id": query["query_id"], "docs": docs}


def build_run_passage_detail(
    run: dict,
    passages: list[Passage],
    score_doc_passages_for_query: Callable[[dict, list[Passage]], list[float]],
) -> dict:
    """Détail complet d'un run : une entrée par requête.

    `score_doc_passages_for_query(query, doc_passages)` calcule le score de
    chaque passage d'un doc pour une requête — dense (cosinus) ou BM25,
    jamais recalculé ici (voir `generate_passage_detail.py`).
    """
    passages_by_doc = group_passages_by_doc(passages)
    queries = [
        build_query_passage_detail(
            q,
            passages_by_doc,
            lambda doc_passages, q=q: score_doc_passages_for_query(q, doc_passages),
        )
        for q in run["queries"]
    ]
    return {
        "campagne": run["campagne"],
        "run_name": run["run_name"],
        "queries": queries,
    }


def passage_detail_path(run_path: Path) -> Path:
    """Chemin du fichier dérivé, dans le dossier de la campagne (critère 1).

    Sous un sous-dossier `passages/`, jamais à plat à côté du fichier de run :
    `list_campaign_run_files` (dashboard) et les tests de EXE-109 listent le
    dossier d'une campagne par `*.json.gz` sans récursion — un dérivé à plat
    s'y ferait passer pour un second run du même nom.
    """
    name = run_path.name
    if not name.endswith(".json.gz"):
        raise ValueError(f"fichier de run attendu en .json.gz : {run_path}")
    return (
        run_path.parent / "passages" / (name[: -len(".json.gz")] + ".passages.json.gz")
    )


def write_passage_detail(run_path: Path, detail: dict) -> Path:
    """Écrit le fichier dérivé, compressé gzip comme un fichier de run (versioning.md)."""
    path = passage_detail_path(run_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(detail, indent=2, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return path


def load_passage_detail(run_path: Path) -> dict | None:
    """Charge le fichier dérivé d'un run, ou None s'il est absent (critère 8)."""
    path = passage_detail_path(run_path)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_query_passage_detail(detail: dict | None, qid: str) -> dict | None:
    """Détail d'une requête dans un fichier dérivé chargé, ou None (détail
    absent, ou requête absente — ne devrait pas arriver, mais jamais levé)."""
    if detail is None:
        return None
    return next((q for q in detail["queries"] if q["query_id"] == qid), None)


def verify_max_grouping_scores(run: dict, detail: dict) -> list[tuple[str, str, float]]:
    """Écarts entre le meilleur score de passage et le score du run (critère 3, H4).

    Ne s'applique qu'au regroupement `max` (seul regroupement couvert par le
    critère) et seulement aux docs présents dans `retrieved_top100` du run —
    un doc attendu absent du top 100 n'a pas de score de run à comparer.
    Retourne les écarts qui dépassent la 4e décimale : (query_id, doc_id, écart).
    """
    if run["config"]["retriever"]["grouping"] != "max":
        return []

    run_scores_by_qid = {
        q["query_id"]: {d["doc_id"]: d["score"] for d in q["retrieved_top100"]}
        for q in run["queries"]
    }

    mismatches: list[tuple[str, str, float]] = []
    for q in detail["queries"]:
        qid = q["query_id"]
        run_scores = run_scores_by_qid.get(qid, {})
        for doc_id, passages in q["docs"].items():
            if doc_id not in run_scores or not passages:
                continue
            best = max(p["score"] for p in passages)
            run_score = run_scores[doc_id]
            if round(best, 4) != round(run_score, 4):
                mismatches.append((qid, doc_id, abs(best - run_score)))
    return mismatches
