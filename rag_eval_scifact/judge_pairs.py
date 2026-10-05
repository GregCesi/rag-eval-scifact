"""Sélection des paires (claim, document) à juger pour la campagne v3-juge
(EXE-119, refondue par EXE-125).

Deux familles, lues dans un seul run de référence
(`dense-qwen3-passages-sans-reranker`) et dans le fichier des étiquettes
d'origine (`results/etiquettes-origine.json`, EXE-124) :

(a) « attendu » — chaque paire (claim, document attendu) du jeu, quelle que
    soit l'étiquette d'origine du document (SUPPORT, CONTRADICT, SANS_PREUVE).
(b) « devant » — pour chaque claim dont au moins un document a une étiquette
    d'origine SUPPORT ou CONTRADICT (un claim « avec preuve »), les documents
    classés avant le premier document SUPPORT ou CONTRADICT dans le run de
    référence, 5 au plus, dans l'ordre des rangs. À défaut d'un tel document
    dans le top 100, les 5 premiers documents retournés.

Une paire peut appartenir aux deux familles (un document cité sans preuve,
classé devant un document avec preuve pour le même claim) : elle n'apparaît
alors qu'une fois, avec les deux familles dans son champ `famille`.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")
EVIDENCE_LABELS = {"SUPPORT", "CONTRADICT"}
EVIDENCE_CATEGORIES = {"confirme", "contredit"}
MAX_DEVANT_PER_CLAIM = 5


def claims_with_evidence(origin_labels: dict) -> list[str]:
    """Claims « avec preuve » (critère 1b) : catégorie confirme ou contredit."""
    return [
        c["query_id"]
        for c in origin_labels["claims"]
        if c["categorie"] in EVIDENCE_CATEGORIES
    ]


def _first_evidence_rank(
    retrieved_sorted: list[dict],
    claim_id: str,
    label_by_pair: dict[tuple[str, str], str],
) -> int | None:
    for doc in retrieved_sorted:
        if label_by_pair.get((claim_id, doc["doc_id"])) in EVIDENCE_LABELS:
            return doc["rank"]
    return None


def _devant_docs_for_claim(
    query: dict, label_by_pair: dict[tuple[str, str], str]
) -> list[dict]:
    """Documents « devant » d'un claim avec preuve (critère 1b)."""
    retrieved_sorted = sorted(query["retrieved_top100"], key=lambda d: d["rank"])
    evidence_rank = _first_evidence_rank(
        retrieved_sorted, query["query_id"], label_by_pair
    )
    if evidence_rank is None:
        return retrieved_sorted[:MAX_DEVANT_PER_CLAIM]
    return [d for d in retrieved_sorted if d["rank"] < evidence_rank][
        :MAX_DEVANT_PER_CLAIM
    ]


def _build_pair(
    claim_id: str,
    claim_text: str,
    doc_id: str,
    rank: int | None,
    famille: set[str],
    etiquette_origine: str | None,
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
        "document_attendu": "attendu" in famille,
        "famille": sorted(famille),
        "etiquette_origine": etiquette_origine,
    }


def build_pairs(
    reference_run: dict,
    origin_labels: dict,
    corpus_by_id: dict[str, dict],
) -> list[dict]:
    """Construit la liste ordonnée des paires à juger (critères 1 à 3, EXE-125)."""
    label_by_pair = {
        (p["query_id"], p["doc_id"]): p["label"] for p in origin_labels["pairs"]
    }
    attendu_by_claim: dict[str, list[dict]] = {}
    for p in origin_labels["pairs"]:
        attendu_by_claim.setdefault(p["query_id"], []).append(p)
    evidence_claims = set(claims_with_evidence(origin_labels))
    queries_by_id = {q["query_id"]: q for q in reference_run["queries"]}

    pairs: list[dict] = []
    for claim_id in sorted(attendu_by_claim, key=int):
        query = queries_by_id[claim_id]
        claim_text = query["query_text"]
        rank_by_doc = {d["doc_id"]: d["rank"] for d in query["retrieved_top100"]}

        devant_docs = (
            _devant_docs_for_claim(query, label_by_pair)
            if claim_id in evidence_claims
            else []
        )
        devant_ids = {d["doc_id"] for d in devant_docs}

        attendu_sorted = sorted(attendu_by_claim[claim_id], key=lambda r: r["doc_id"])
        attendu_ids = {r["doc_id"] for r in attendu_sorted}

        for record in attendu_sorted:
            doc_id = record["doc_id"]
            famille = {"attendu"} | ({"devant"} if doc_id in devant_ids else set())
            pairs.append(
                _build_pair(
                    claim_id,
                    claim_text,
                    doc_id,
                    rank_by_doc.get(doc_id),
                    famille,
                    record["label"],
                    corpus_by_id,
                )
            )

        for d in devant_docs:
            if d["doc_id"] in attendu_ids:
                continue
            pairs.append(
                _build_pair(
                    claim_id,
                    claim_text,
                    d["doc_id"],
                    d["rank"],
                    {"devant"},
                    None,
                    corpus_by_id,
                )
            )

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
