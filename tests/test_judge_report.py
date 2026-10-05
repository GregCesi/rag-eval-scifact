"""Tests du rapport comparant les juges de v3-juge (EXE-122, critères 1 à 8).

Campagne fabriquée, sans aucun modèle : paires et jugements écrits à la main en
JSON, comme `test_report.py` le fait pour les runs de retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval_scifact import judge_report as judge_report_module
from rag_eval_scifact.judge_report import (
    annotation_crosstab,
    cause_summary,
    dataset_summary,
    devant_summary,
    expected_crosstab,
    judge_summary,
    level_distribution,
    pairwise_comparison,
)
from rag_eval_scifact.stats import cohen_kappa

PAIRS = [
    {
        "pair_id": "1:d1",
        "claim_id": "1",
        "doc_id": "d1",
        "document_attendu": True,
        "famille": ["attendu"],
        "etiquette_origine": "SUPPORT",
    },
    {
        "pair_id": "1:d2",
        "claim_id": "1",
        "doc_id": "d2",
        "document_attendu": False,
        "famille": ["devant"],
        "etiquette_origine": None,
    },
    {
        "pair_id": "2:d3",
        "claim_id": "2",
        "doc_id": "d3",
        "document_attendu": True,
        "famille": ["attendu"],
        "etiquette_origine": "CONTRADICT",
    },
    {
        "pair_id": "2:d4",
        "claim_id": "2",
        "doc_id": "d4",
        "document_attendu": False,
        "famille": ["devant"],
        "etiquette_origine": None,
    },
]
PAIRS_BY_ID = {p["pair_id"]: p for p in PAIRS}

# Claim 1 : le document attendu répond (SUPPORTS), l'intrus ne répond pas.
# Claim 2 : le document attendu ne répond pas, l'intrus répond (REFUTES).
LOCAL_JUDGMENTS = [
    {
        "pair_id": "1:d1",
        "claim_id": "1",
        "doc_id": "d1",
        "verdict": "SUPPORTS",
        "duration_seconds": 1.0,
    },
    {
        "pair_id": "1:d2",
        "claim_id": "1",
        "doc_id": "d2",
        "verdict": "NOT_ENOUGH_INFO",
        "duration_seconds": 2.0,
    },
    {
        "pair_id": "2:d3",
        "claim_id": "2",
        "doc_id": "d3",
        "verdict": "NOT_ENOUGH_INFO",
        "duration_seconds": 3.0,
    },
    {
        "pair_id": "2:d4",
        "claim_id": "2",
        "doc_id": "d4",
        "verdict": "REFUTES",
        "duration_seconds": 4.0,
    },
]


# ---------------------------------------------------------------------------
# Critère 2 — effectifs de verdicts, citations introuvables, durée moyenne
# ---------------------------------------------------------------------------


def test_judge_summary_counts_verdicts_and_average_duration():
    judgments = LOCAL_JUDGMENTS + [
        {
            "pair_id": "3:d5",
            "claim_id": "3",
            "doc_id": "d5",
            "verdict": "illisible",
            "duration_seconds": 5.0,
        },
        {
            "pair_id": "3:d6",
            "claim_id": "3",
            "doc_id": "d6",
            "verdict": "citation introuvable",
            "duration_seconds": 6.0,
        },
    ]

    summary = judge_summary(judgments)

    assert summary["n_judged"] == 6
    assert summary["supports"] == 1
    assert summary["refutes"] == 1
    assert summary["not_enough_info"] == 2
    assert summary["illisible"] == 1
    assert summary["citation_introuvable"] == 1
    assert summary["avg_duration_seconds"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Critère 3 — ce que le juge dit du jeu de données
# ---------------------------------------------------------------------------


def test_dataset_summary_rates_and_claim_counts():
    summary = dataset_summary(LOCAL_JUDGMENTS, PAIRS_BY_ID)

    assert summary["expected_responds_rate"] == pytest.approx(0.5)  # 1 sur 2
    assert summary["non_expected_responds_rate"] == pytest.approx(0.5)  # 1 sur 2
    assert summary["n_claims_with_non_expected_responds"] == 1  # claim 2
    assert summary["n_claims_without_expected_responds"] == 1  # claim 2


# ---------------------------------------------------------------------------
# Critère 6 — jugements du juge par étapes, par cause
# ---------------------------------------------------------------------------


def test_cause_summary_counts_by_cause_including_zero():
    judgments = [
        {"verdict": "NOT_ENOUGH_INFO", "cause": "hors sujet"},
        {"verdict": "NOT_ENOUGH_INFO", "cause": "hors sujet"},
        {"verdict": "NOT_ENOUGH_INFO", "cause": "vocabulaire"},
        {"verdict": "SUPPORTS", "cause": ""},
    ]

    summary = cause_summary(judgments)

    assert summary == {
        "hors sujet": 2,
        "vocabulaire": 1,
        "inférence": 0,
        "sujet voisin": 0,
    }


# ---------------------------------------------------------------------------
# EXE-125, critère 8 — tableau croisé verdict / étiquette d'origine
# ---------------------------------------------------------------------------


def test_expected_crosstab_restricted_to_attendu_pairs_with_valid_verdict():
    # Seuls d1 et d3 sont attendus : d1 SUPPORTS/SUPPORT (accord), d3
    # NOT_ENOUGH_INFO/CONTRADICT (désaccord). d2 et d4 sont « devant », ignorés.
    crosstab = expected_crosstab(LOCAL_JUDGMENTS, PAIRS_BY_ID)

    assert crosstab["confusion"] == {
        ("SUPPORTS", "SUPPORT"): 1,
        ("NOT_ENOUGH_INFO", "CONTRADICT"): 1,
    }
    assert crosstab["n_valid"] == 2
    assert crosstab["agreement"] == pytest.approx(0.5)


def test_expected_crosstab_excludes_illisible_and_citation_introuvable():
    judgments = [
        {"pair_id": "1:d1", "verdict": "illisible"},
        {"pair_id": "2:d3", "verdict": "citation introuvable"},
    ]

    crosstab = expected_crosstab(judgments, PAIRS_BY_ID)

    assert crosstab["n_valid"] == 0
    assert crosstab["agreement"] == 0.0


# ---------------------------------------------------------------------------
# EXE-125, critère 9 — documents classés devant
# ---------------------------------------------------------------------------


def test_devant_summary_counts_verdicts_and_responding_claims():
    # Seuls d2 et d4 sont « devant » : d2 NOT_ENOUGH_INFO (claim 1), d4 REFUTES
    # (claim 2, répond). d1 et d3 sont « attendu » seul, ignorés.
    summary = devant_summary(LOCAL_JUDGMENTS, PAIRS_BY_ID)

    assert summary["supports"] == 0
    assert summary["refutes"] == 1
    assert summary["not_enough_info"] == 1
    assert summary["claims"] == ["2"]


# ---------------------------------------------------------------------------
# EXE-125, critère 10 — répartition des niveaux de lecture
# ---------------------------------------------------------------------------

LEVELED_JUDGMENTS = [
    {
        "pair_id": "1:d1",
        "claim_id": "1",
        "verdict": "SUPPORTS",
        "level": "DIRECT",
    },
    {
        "pair_id": "2:d3",
        "claim_id": "2",
        "verdict": "REFUTES",
        "level": "VOCABULARY",
    },
    {
        "pair_id": "1:d2",
        "claim_id": "1",
        "verdict": "SUPPORTS",
        "level": "REASONING",
    },
    {
        "pair_id": "2:d4",
        "claim_id": "2",
        "verdict": "NOT_ENOUGH_INFO",
        "level": "NONE",
    },
]


def test_level_distribution_on_attendu_restricted_to_support_or_contradict():
    # d1 (attendu, SUPPORT, répond, DIRECT) compte ; d3 (attendu, CONTRADICT,
    # répond, VOCABULARY) compte. Aucun autre document attendu dans ce jeu.
    distribution = level_distribution(LEVELED_JUDGMENTS, PAIRS_BY_ID, "attendu")

    assert distribution == {"DIRECT": 1, "VOCABULARY": 1, "REASONING": 0}


def test_level_distribution_on_devant_ignores_origin_label():
    # d2 (devant, répond, REASONING) et d4 (devant, NOT_ENOUGH_INFO -> ne
    # répond pas, ignoré) : seul d2 compte.
    distribution = level_distribution(LEVELED_JUDGMENTS, PAIRS_BY_ID, "devant")

    assert distribution == {"DIRECT": 0, "VOCABULARY": 0, "REASONING": 1}


def test_level_distribution_is_none_when_no_judgment_carries_a_level():
    # Le juge par étapes ne rend jamais de niveau (EXE-121, ne change pas).
    distribution = level_distribution(LOCAL_JUDGMENTS, PAIRS_BY_ID, "attendu")

    assert distribution is None


# ---------------------------------------------------------------------------
# Critère 4 et 5 — tableau croisé, accord et kappa entre deux juges
# ---------------------------------------------------------------------------


def test_pairwise_comparison_restricts_to_common_valid_verdicts():
    judgments_a = [
        {"pair_id": "p1", "verdict": "SUPPORTS"},
        {"pair_id": "p2", "verdict": "REFUTES"},
        {"pair_id": "p3", "verdict": "NOT_ENOUGH_INFO"},
        {"pair_id": "p4", "verdict": "illisible"},
    ]
    judgments_b = [
        {"pair_id": "p1", "verdict": "SUPPORTS"},
        {"pair_id": "p2", "verdict": "NOT_ENOUGH_INFO"},
        {"pair_id": "p3", "verdict": "NOT_ENOUGH_INFO"},
        {"pair_id": "p4", "verdict": "SUPPORTS"},
        {"pair_id": "p5", "verdict": "SUPPORTS"},
    ]

    result = pairwise_comparison(judgments_a, judgments_b)

    # Seules p1, p2, p3 sont communes avec un verdict valide des deux côtés :
    # p4 est illisible côté a, p5 n'existe pas côté a.
    assert result["n_common"] == 3
    expected_full = cohen_kappa(
        ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        ["SUPPORTS", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"],
    )
    assert result["full"]["agreement"] == pytest.approx(expected_full["agreement"])
    assert result["full"]["kappa"] == pytest.approx(expected_full["kappa"])

    expected_binary = cohen_kappa(
        ["répond", "répond", "ne répond pas"],
        ["répond", "ne répond pas", "ne répond pas"],
    )
    assert result["binary"]["agreement"] == pytest.approx(expected_binary["agreement"])
    assert result["binary"]["kappa"] == pytest.approx(expected_binary["kappa"])


def test_pairwise_comparison_is_none_without_any_common_valid_pair():
    judgments_a = [{"pair_id": "p1", "verdict": "illisible"}]
    judgments_b = [{"pair_id": "p1", "verdict": "SUPPORTS"}]

    assert pairwise_comparison(judgments_a, judgments_b) is None


# ---------------------------------------------------------------------------
# Critère 7 — croisement avec les étiquettes d'error analysis v1
# ---------------------------------------------------------------------------


@pytest.fixture
def _annotation_files(tmp_path, monkeypatch):
    v1_path = tmp_path / "v1-annotations.json"
    deep_miss_path = tmp_path / "v1-deep-miss-annotations.json"
    v1_path.write_text(
        json.dumps(
            {
                "1": {"categorie": "Q"},
                "2": {"categorie": "Q"},
                "3": {"categorie": "Q "},
                "99": {"categorie": "Q"},
            }
        ),
        encoding="utf-8",
    )
    deep_miss_path.write_text(json.dumps({"1": {"categorie": "T"}}), encoding="utf-8")
    monkeypatch.setattr(
        judge_report_module, "ANNOTATION_FILES", (v1_path, deep_miss_path)
    )
    return v1_path, deep_miss_path


def test_annotation_crosstab_rows(_annotation_files):
    pairs = PAIRS + [
        {
            "pair_id": "3:d5",
            "claim_id": "3",
            "doc_id": "d5",
            "document_attendu": True,
        },
    ]
    judgments_by_judge = {"local": LOCAL_JUDGMENTS}

    rows = annotation_crosstab(pairs, judgments_by_judge)

    by_key = {(r["fichier"], r["etiquette"]): r for r in rows}

    # "Q" (v1-annotations) regroupe les claims 1 et 2, pas 99 (absent des paires).
    q_row = by_key[("v1-annotations.json", "Q")]
    assert q_row["n_claims"] == 2
    assert q_row["par_juge"]["local"]["expected_responds"] == 1  # claim 1
    assert q_row["par_juge"]["local"]["non_expected_responds"] == 1  # claim 2

    # "Q " (avec espace) n'est jamais fusionnée avec "Q".
    assert ("v1-annotations.json", "Q ") in by_key
    assert by_key[("v1-annotations.json", "Q ")]["n_claims"] == 1

    # Le fichier d'origine de chaque étiquette est indiqué séparément.
    t_row = by_key[("v1-deep-miss-annotations.json", "T")]
    assert t_row["n_claims"] == 1


# ---------------------------------------------------------------------------
# Critères 1 et 8 — rapport complet : juges manquants, régénération identique
# ---------------------------------------------------------------------------


def _write_pairs(results_dir: Path, campagne: str, pairs: list[dict]) -> None:
    path = results_dir / campagne / "paires.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")


def _write_judgments(
    results_dir: Path, campagne: str, juge: str, judgments: list[dict]
) -> None:
    path = results_dir / campagne / f"jugements-{juge}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(judgments), encoding="utf-8")


@pytest.fixture
def _campaign(tmp_path, monkeypatch, _annotation_files):
    monkeypatch.setattr(judge_report_module, "RESULTS_DIR", tmp_path)
    _write_pairs(tmp_path, "campagne-test", PAIRS)
    _write_judgments(tmp_path, "campagne-test", "local", LOCAL_JUDGMENTS)
    return tmp_path


def test_report_says_which_judges_are_missing(_campaign):
    path = judge_report_module.write_report("campagne-test")
    content = path.read_text(encoding="utf-8")

    assert "### local" in content
    assert "### claude" in content
    assert "### etapes" in content
    # claude et etapes n'ont pas de fichier de jugements dans cette campagne.
    lines = content.splitlines()
    claude_idx = lines.index("### claude")
    etapes_idx = lines.index("### etapes")
    assert lines[claude_idx + 2] == "absent"
    assert lines[etapes_idx + 2] == "absent"


def test_report_regenerates_identically_when_judgments_are_unchanged(_campaign):
    first = judge_report_module.write_report("campagne-test").read_text(
        encoding="utf-8"
    )
    second = judge_report_module.write_report("campagne-test").read_text(
        encoding="utf-8"
    )

    assert first == second
