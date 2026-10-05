"""Étiquettes des annotateurs d'origine par paire (claim, document) du split test (EXE-124).

BEIR SciFact réduit chaque paire qrel à « document attendu », sans dire si le
document confirme le claim, le contredit, ou ne le prouve pas. Ce module relit
la source d'origine (`claims_dev.jsonl` + `corpus.jsonl` sentence-splitté) pour
reconstruire cette distinction, et la vérifie contre les qrels BEIR
(`data/scifact/qrels/test.tsv`) : les deux fichiers décrivent le même jeu de
paires par deux chemins indépendants.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

CATEGORY_NAMES = ("confirme", "contredit", "sans_preuve")

LABEL_IN_CLEAR = {
    "SUPPORT": "confirme",
    "CONTRADICT": "contredit",
    "SANS_PREUVE": "sans preuve",
}


def label_in_clear(label: str) -> str:
    """Étiquette d'origine d'un document, en clair (critère 10, EXE-124)."""
    return LABEL_IN_CLEAR[label]


def load_claims(path: Path) -> dict[str, dict]:
    """Charge `claims_dev.jsonl`, indexé par id de claim (str)."""
    claims: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            claims[str(record["id"])] = record
    return claims


def load_origin_corpus(path: Path) -> dict[str, dict]:
    """Charge le corpus d'origine (sentence-splitté), indexé par doc_id (str)."""
    corpus: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            corpus[str(record["doc_id"])] = record
    return corpus


def load_qrels_pairs(path: Path) -> list[tuple[str, str]]:
    """Charge les paires (query-id, corpus-id) des qrels BEIR, dans l'ordre du fichier."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pairs.append((row["query-id"], row["corpus-id"]))
    return pairs


def claim_doc_pairs_from_claims(claims: dict[str, dict]) -> list[tuple[str, str]]:
    """Paires (claim, document cité), déduites de `cited_doc_ids`, sans doublon."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for claim_id, claim in claims.items():
        for doc_id in claim["cited_doc_ids"]:
            pair = (claim_id, str(doc_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def verify_pairs_match_qrels(
    claim_pairs: list[tuple[str, str]], qrels_pairs: list[tuple[str, str]]
) -> None:
    """Critère 4 : les paires côté claims et côté qrels doivent coïncider exactement."""
    claim_set = set(claim_pairs)
    qrels_set = set(qrels_pairs)
    missing_from_qrels = sorted(claim_set - qrels_set)
    missing_from_claims = sorted(qrels_set - claim_set)
    if missing_from_qrels or missing_from_claims:
        raise ValueError(
            "les paires (claim, document) de claims_dev.jsonl et de "
            "qrels/test.tsv ne correspondent pas : "
            f"{len(missing_from_qrels)} présente(s) seulement côté claims "
            f"{missing_from_qrels[:5]}, "
            f"{len(missing_from_claims)} présente(s) seulement côté qrels "
            f"{missing_from_claims[:5]}"
        )


def label_for_pair(claim: dict, doc_id: str) -> str:
    """Étiquette d'origine d'une paire (claim, document) : SUPPORT, CONTRADICT, SANS_PREUVE."""
    evidence = claim["evidence"].get(doc_id)
    if not evidence:
        return "SANS_PREUVE"
    labels = {entry["label"] for entry in evidence}
    if "SUPPORT" in labels:
        return "SUPPORT"
    if "CONTRADICT" in labels:
        return "CONTRADICT"
    return next(iter(labels))


def evidence_sentences_for_pair(
    claim: dict, doc_id: str, corpus: dict[str, dict]
) -> list[str]:
    """Texte des phrases-preuve d'une paire SUPPORT/CONTRADICT, dans l'ordre du document."""
    evidence = claim["evidence"].get(doc_id)
    if not evidence:
        return []
    indices: set[int] = set()
    for entry in evidence:
        indices.update(entry["sentences"])
    abstract = corpus[doc_id]["abstract"]
    return [abstract[i] for i in sorted(indices)]


def build_pair_record(claim_id: str, doc_id: str, claim: dict, corpus: dict) -> dict:
    """Enregistrement d'une paire : étiquette, et phrases-preuve pour SUPPORT/CONTRADICT."""
    label = label_for_pair(claim, doc_id)
    evidence_sentences = (
        evidence_sentences_for_pair(claim, doc_id, corpus)
        if label != "SANS_PREUVE"
        else []
    )
    return {
        "query_id": claim_id,
        "doc_id": doc_id,
        "label": label,
        "evidence_sentences": evidence_sentences,
    }


def categorie_for_claim(pair_records_for_claim: list[dict]) -> str:
    """Catégorie d'un claim (critère 3) : confirme si un SUPPORT, contredit si un
    CONTRADICT, sans_preuve sinon."""
    labels = {record["label"] for record in pair_records_for_claim}
    if "SUPPORT" in labels:
        return "confirme"
    if "CONTRADICT" in labels:
        return "contredit"
    return "sans_preuve"


def build_origin_labels_document(
    claims: dict[str, dict],
    corpus: dict[str, dict],
    qrels_pairs: list[tuple[str, str]],
) -> dict:
    """Construit le document complet : `pairs` (critères 1, 2) et `claims` (critère 3).

    Vérifie d'abord les paires contre les qrels (critère 4) : lève `ValueError`
    si une paire manque d'un côté ou de l'autre, avant d'écrire quoi que ce soit.
    """
    claim_pairs = claim_doc_pairs_from_claims(claims)
    verify_pairs_match_qrels(claim_pairs, qrels_pairs)

    pairs_by_claim: dict[str, list[dict]] = {}
    for claim_id, doc_id in claim_pairs:
        record = build_pair_record(claim_id, doc_id, claims[claim_id], corpus)
        pairs_by_claim.setdefault(claim_id, []).append(record)

    sorted_claim_ids = sorted(pairs_by_claim, key=int)
    pairs = [
        record
        for claim_id in sorted_claim_ids
        for record in sorted(pairs_by_claim[claim_id], key=lambda r: r["doc_id"])
    ]
    claims_out = [
        {
            "query_id": claim_id,
            "categorie": categorie_for_claim(pairs_by_claim[claim_id]),
        }
        for claim_id in sorted_claim_ids
    ]
    return {"pairs": pairs, "claims": claims_out}


def build_label_lookup(
    origin_labels: dict,
) -> tuple[
    dict[tuple[str, str], str], dict[tuple[str, str], list[str]], dict[str, str]
]:
    """Dérive du document écrit par `build_origin_labels_document` les trois
    structures que le dashboard consulte (critères 10 à 12, EXE-124) :
    étiquette et phrases-preuve par paire, catégorie par claim."""
    label_by_pair = {
        (p["query_id"], p["doc_id"]): p["label"] for p in origin_labels["pairs"]
    }
    evidence_by_pair = {
        (p["query_id"], p["doc_id"]): p["evidence_sentences"]
        for p in origin_labels["pairs"]
    }
    categorie_by_qid = {c["query_id"]: c["categorie"] for c in origin_labels["claims"]}
    return label_by_pair, evidence_by_pair, categorie_by_qid


def filter_rows_by_categorie(
    rows: list[dict], categories: set[str], categorie_by_qid: dict[str, str]
) -> list[dict]:
    """Critère 12 : limite `rows` (chacun avec une clé `query_id`) aux claims
    dont la catégorie d'origine est dans `categories`."""
    return [r for r in rows if categorie_by_qid.get(r["query_id"]) in categories]
