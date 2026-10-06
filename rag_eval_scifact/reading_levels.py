"""Niveaux de lecture du juge Claude sur les 339 goldens de v2-grid (EXE-137).

La campagne v3-juge a fait dire à Claude, pour chacun des 339 goldens (une
paire claim/document attendu de `results/etiquettes-origine.json`), s'il
répond à l'affirmation et à quel niveau de lecture (direct, vocabulaire,
raisonnement). Ce module croise ce jugement (`results/v3-juge/jugements-
claude.json`) avec les 34 runs déjà commités de `results/v2-grid/`, sans
jamais relancer un run ni un juge.

Sept groupes, disjoints et couvrant les 339 goldens :
- `direct`, `vocabulaire`, `raisonnement` — avec preuve (étiquette SUPPORT ou
  CONTRADICT), jugés répondre, par niveau.
- `avec_preuve_ne_repond_pas` — avec preuve, jugés ne pas répondre.
- `sans_preuve_repond` / `sans_preuve_ne_repond_pas` — sans preuve (étiquette
  SANS_PREUVE), selon que le juge les juge répondre ou non.
- `non_juge` — refus ou réponse illisible du juge.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_scifact.category_report import group_runs_by_base_strategy
from rag_eval_scifact.judge_pairs import load_pairs
from rag_eval_scifact.judge_report import expected_crosstab

RESULTS_DIR = Path("results")
CAMPAGNE = "v2-grid"
JUGE_CAMPAGNE = "v3-juge"
ORIGIN_LABELS_PATH = RESULTS_DIR / "etiquettes-origine.json"
PAIRES_PATH = RESULTS_DIR / JUGE_CAMPAGNE / "paires.json"
JUGEMENTS_PATH = RESULTS_DIR / JUGE_CAMPAGNE / "jugements-claude.json"

GROUP_NAMES = (
    "direct",
    "vocabulaire",
    "raisonnement",
    "avec_preuve_ne_repond_pas",
    "sans_preuve_repond",
    "sans_preuve_ne_repond_pas",
    "non_juge",
)

GROUP_LABELS = {
    "direct": "Direct",
    "vocabulaire": "Vocabulaire",
    "raisonnement": "Raisonnement",
    "avec_preuve_ne_repond_pas": "Avec preuve, ne répond pas",
    "sans_preuve_repond": "Sans preuve, répond",
    "sans_preuve_ne_repond_pas": "Sans preuve, ne répond pas",
    "non_juge": "Non jugé",
}

THRESHOLD_LABELS = ("rang 1", "5 premiers", "10 premiers")
THRESHOLD_RANKS = {"rang 1": 1, "5 premiers": 5, "10 premiers": 10}
DEFAULT_THRESHOLD_LABEL = "5 premiers"

DEFAULT_COMPARISON_RUN_NAMES = (
    "bm25-document-sans-reranker",
    "dense-minilm-256-sans-reranker",
    "dense-qwen3-passages-sans-reranker",
)

_RESPONDING_VERDICTS = ("SUPPORTS", "REFUTES")
_LEVEL_TO_GROUP = {
    "DIRECT": "direct",
    "VOCABULARY": "vocabulaire",
    "REASONING": "raisonnement",
}


def load_origin_labels(path: Path = ORIGIN_LABELS_PATH) -> dict | None:
    """`None` si `path` est absent (étiquettes d'origine manquantes)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_judgments(path: Path = JUGEMENTS_PATH) -> list[dict] | None:
    """`None` si `path` est absent (jugements du juge Claude manquants)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _group_for(label: str, verdict: str | None, level: str | None) -> str:
    responds = verdict in _RESPONDING_VERDICTS
    ne_repond_pas = verdict == "NOT_ENOUGH_INFO"
    avec_preuve = label in ("SUPPORT", "CONTRADICT")
    if avec_preuve and responds:
        return _LEVEL_TO_GROUP[level]
    if avec_preuve and ne_repond_pas:
        return "avec_preuve_ne_repond_pas"
    if not avec_preuve and responds:
        return "sans_preuve_repond"
    if not avec_preuve and ne_repond_pas:
        return "sans_preuve_ne_repond_pas"
    return "non_juge"


def build_goldens(origin_labels: dict, judgments: list[dict]) -> list[dict]:
    """Les 339 goldens, un par paire de `origin_labels["pairs"]`, enrichis du
    jugement de Claude et du groupe qu'ils forment (critère 2)."""
    judgment_by_pair_id = {j["pair_id"]: j for j in judgments}
    goldens = []
    for pair in origin_labels["pairs"]:
        pair_id = f"{pair['query_id']}:{pair['doc_id']}"
        judgment = judgment_by_pair_id.get(pair_id)
        verdict = judgment["verdict"] if judgment else None
        level = judgment.get("level") if judgment else None
        goldens.append(
            {
                "query_id": pair["query_id"],
                "doc_id": pair["doc_id"],
                "label": pair["label"],
                "verdict": verdict,
                "evidence": judgment.get("evidence", "") if judgment else "",
                "reason": judgment.get("reason", "") if judgment else "",
                "group": _group_for(pair["label"], verdict, level),
            }
        )
    return goldens


def load_goldens(
    origin_labels_path: Path = ORIGIN_LABELS_PATH,
    jugements_path: Path = JUGEMENTS_PATH,
) -> list[dict] | None:
    """`None` si le fichier des étiquettes ou celui des jugements manque."""
    origin_labels = load_origin_labels(origin_labels_path)
    judgments = load_judgments(jugements_path)
    if origin_labels is None or judgments is None:
        return None
    return build_goldens(origin_labels, judgments)


def group_counts(goldens: list[dict]) -> dict[str, int]:
    """Effectif de chaque groupe (critère 2)."""
    counts = {name: 0 for name in GROUP_NAMES}
    for golden in goldens:
        counts[golden["group"]] += 1
    return counts


def filter_goldens_by_categorie(goldens: list[dict], categorie: str) -> list[dict]:
    """Critère 8 : restreint aux goldens « confirme » (étiquette SUPPORT) ou
    « contredit » (étiquette CONTRADICT)."""
    target_label = {"confirme": "SUPPORT", "contredit": "CONTRADICT"}[categorie]
    return [g for g in goldens if g["label"] == target_label]


def doc_ranks_by_query(run_data: dict) -> dict[str, dict[str, int]]:
    """Rang de chaque document retrouvé, par requête, pour un run chargé."""
    return {
        q["query_id"]: {d["doc_id"]: d["rank"] for d in q["retrieved_top100"]}
        for q in run_data["queries"]
    }


def rank_in_run(
    ranks_by_query: dict[str, dict[str, int]], query_id: str, doc_id: str
) -> int | None:
    return ranks_by_query.get(query_id, {}).get(doc_id)


def found_counts(
    goldens: list[dict], run_data: dict, threshold: int
) -> dict[str, dict[str, int]]:
    """Critères 3, 5, 6 : effectif retrouvé sous `threshold` et effectif
    total, par groupe, pour un run."""
    ranks = doc_ranks_by_query(run_data)
    counts = {name: {"found": 0, "total": 0} for name in GROUP_NAMES}
    for golden in goldens:
        bucket = counts[golden["group"]]
        bucket["total"] += 1
        rank = rank_in_run(ranks, golden["query_id"], golden["doc_id"])
        if rank is not None and rank <= threshold:
            bucket["found"] += 1
    return counts


def build_table_rows(
    goldens: list[dict], runs: list[dict], threshold: int
) -> list[dict]:
    """Critère 3 : une ligne par run, avec son `found_counts`."""
    return [
        {
            "run_name": run_data["run_name"],
            "counts": found_counts(goldens, run_data, threshold),
        }
        for run_data in runs
    ]


def group_rate(row: dict, group: str) -> float:
    """Part des goldens de `group` retrouvés, pour une ligne de `build_table_rows`."""
    counts = row["counts"][group]
    return counts["found"] / counts["total"] if counts["total"] else 0.0


def sort_rows_by_group(rows: list[dict], group: str) -> list[dict]:
    """Critère 7 : les lignes, de la meilleure à la moins bonne sur `group`."""
    return sorted(rows, key=lambda row: group_rate(row, group), reverse=True)


def reranker_diff_rows(
    goldens: list[dict], runs: list[dict], threshold: int
) -> list[dict]:
    """Critère 10 : écart (avec − sans), en nombre de goldens retrouvés sous
    `threshold`, par groupe, pour chaque stratégie de base ayant ses deux
    variantes."""
    grouped = group_runs_by_base_strategy(runs)
    rows = []
    for base in sorted(grouped):
        variants = grouped[base]
        if "avec" not in variants or "sans" not in variants:
            continue
        sans_counts = found_counts(goldens, variants["sans"], threshold)
        avec_counts = found_counts(goldens, variants["avec"], threshold)
        diffs = {
            name: avec_counts[name]["found"] - sans_counts[name]["found"]
            for name in GROUP_NAMES
        }
        rows.append({"base": base, "diffs": diffs})
    return rows


def build_group_detail_rows(
    goldens: list[dict],
    run_data: dict,
    group: str,
    corpus: dict[str, dict] | None = None,
) -> list[dict]:
    """Critère 11 : une ligne par golden du groupe — identifiant et texte de
    l'affirmation (pris dans le run, qui porte `query_text`), titre du
    document (pris dans le corpus), rang dans ce run, verdict du juge, phrase
    citée et raison."""
    queries_by_id = {q["query_id"]: q for q in run_data["queries"]}
    ranks = doc_ranks_by_query(run_data)
    rows = []
    for golden in goldens:
        if golden["group"] != group:
            continue
        query = queries_by_id.get(golden["query_id"], {})
        doc = (corpus or {}).get(golden["doc_id"], {})
        rows.append(
            {
                "query_id": golden["query_id"],
                "claim_text": query.get("query_text", ""),
                "doc_id": golden["doc_id"],
                "doc_title": doc.get("title", ""),
                "rank": rank_in_run(ranks, golden["query_id"], golden["doc_id"]),
                "verdict": golden["verdict"],
                "evidence": golden["evidence"],
                "reason": golden["reason"],
            }
        )
    return rows


def sort_rows_beyond_threshold_first(rows: list[dict], threshold: int) -> list[dict]:
    """Critère 12 : les goldens que ce run ne retrouve pas sous `threshold`
    apparaissent en premier."""

    def _found(row: dict) -> bool:
        return row["rank"] is not None and row["rank"] <= threshold

    return sorted(rows, key=_found)


def judge_agreement_with_annotators(
    paires_path: Path = PAIRES_PATH, jugements_path: Path = JUGEMENTS_PATH
) -> float | None:
    """Taux d'accord du juge Claude avec les étiquettes d'origine, sur les
    documents attendus (H2) — relit `judge_report.expected_crosstab`, calcul
    déjà fait pour `results/v3-juge/RAPPORT.md`, jamais reproduit ici.
    `None` si `paires.json` ou les jugements manquent."""
    if not paires_path.exists():
        return None
    judgments = load_judgments(jugements_path)
    if judgments is None:
        return None
    pairs = load_pairs(paires_path)
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    return expected_crosstab(judgments, pairs_by_id)["agreement"]
