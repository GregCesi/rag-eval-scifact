"""Tests du rapport de v2-grid par catégorie d'étiquette d'origine (EXE-124,
critères 5, 6, 8, 9).

Campagne fabriquée de quatre runs (deux stratégies de base, avec et sans
reranker) et un fichier d'étiquettes d'origine fabriqué de trois claims (un
par catégorie) : aucun run réel n'est relu, tout est du JSON gzip écrit à la
main au format `versioning.md`. Les écarts et p-values attendus sont calculés
en appelant directement `stats.paired_permutation_test` (déjà testé par
`test_stats.py`), jamais recalculés à la main ici.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rag_eval_scifact import category_report
from rag_eval_scifact import report as report_module
from rag_eval_scifact.metrics import ndcg_at_k
from rag_eval_scifact.stats import paired_permutation_test

ORIGIN_LABELS = {
    "pairs": [
        {
            "query_id": "1",
            "doc_id": "d1",
            "label": "SUPPORT",
            "evidence_sentences": ["s"],
        },
        {
            "query_id": "2",
            "doc_id": "d2",
            "label": "CONTRADICT",
            "evidence_sentences": ["t"],
        },
        {
            "query_id": "3",
            "doc_id": "d3",
            "label": "SANS_PREUVE",
            "evidence_sentences": [],
        },
    ],
    "claims": [
        {"query_id": "1", "categorie": "confirme"},
        {"query_id": "2", "categorie": "contredit"},
        {"query_id": "3", "categorie": "sans_preuve"},
    ],
}


def _retrieved(doc_at_rank: dict[str, int], n: int = 10) -> list[dict]:
    """Top-n fabriqué : `doc_at_rank` place un doc précis à un rang donné, le
    reste est comblé par des distracteurs (jamais un document attendu)."""
    slot_to_doc = {rank: doc_id for doc_id, rank in doc_at_rank.items()}
    return [
        {
            "doc_id": slot_to_doc.get(rank, f"distracteur-{rank}"),
            "rank": rank,
            "score": 1.0 / rank,
        }
        for rank in range(1, n + 1)
    ]


def _run(run_name: str, retrieved_by_claim: dict[str, dict[str, int]]) -> dict:
    return {
        "campagne": "v2-grid",
        "run_name": run_name,
        "date": "2026-10-04T00:00:00",
        "dataset_hash": "sha256:fake",
        "config": {},
        "metrics": {},
        "extended_metrics": {},
        "queries": [
            {
                "query_id": qid,
                "query_text": f"claim {qid}",
                "expected_docs": [],
                "retrieved_top100": _retrieved(doc_at_rank),
                "per_query_metrics": {},
            }
            for qid, doc_at_rank in retrieved_by_claim.items()
        ],
    }


SANS_A = _run(
    "strat-a-sans-reranker", {"1": {"d1": 1}, "2": {"d2": 5}, "3": {"d3": 10}}
)
AVEC_A = _run("strat-a-avec-reranker", {"1": {"d1": 1}, "2": {"d2": 1}, "3": {"d3": 1}})
SANS_B = _run("strat-b-sans-reranker", {"1": {"d1": 2}, "2": {"d2": 2}, "3": {"d3": 2}})
AVEC_B = _run("strat-b-avec-reranker", {"1": {"d1": 1}, "2": {"d2": 3}, "3": {"d3": 5}})


def _write_gz(path: Path, run_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(run_data, f)


@pytest.fixture
def _campaign(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(report_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(category_report, "V2_GRID_DIR", results_dir / "v2-grid")
    monkeypatch.setattr(
        category_report, "ORIGIN_LABELS_PATH", results_dir / "etiquettes-origine.json"
    )

    (results_dir / "v2-grid").mkdir(parents=True)
    (results_dir / "etiquettes-origine.json").write_text(
        json.dumps(ORIGIN_LABELS), encoding="utf-8"
    )
    for run_data in (SANS_A, AVEC_A, SANS_B, AVEC_B):
        _write_gz(
            results_dir / "v2-grid" / f"{run_data['run_name']}-20261004.json.gz",
            run_data,
        )

    return results_dir


@pytest.fixture
def _fab_campaign(tmp_path, monkeypatch):
    """Campagne fabriquée nommée `fab-campagne`, distincte de v2-grid (EXE-142,
    H2 : cette fiche ne dépend pas des deux autres fiches de la campagne)."""
    results_dir = tmp_path / "results"
    monkeypatch.setattr(report_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        category_report, "ORIGIN_LABELS_PATH", results_dir / "etiquettes-origine.json"
    )

    (results_dir / "fab-campagne").mkdir(parents=True)
    (results_dir / "etiquettes-origine.json").write_text(
        json.dumps(ORIGIN_LABELS), encoding="utf-8"
    )
    fab_runs = [SANS_A, AVEC_A, SANS_B, AVEC_B]
    for run_data in fab_runs:
        _write_gz(
            results_dir / "fab-campagne" / f"{run_data['run_name']}-20261006.json.gz",
            run_data,
        )

    return results_dir, fab_runs


# ---------------------------------------------------------------------------
# Critère 3 (relu ici) — regroupement des claims par catégorie
# ---------------------------------------------------------------------------


def test_claims_by_categorie_groups_and_sorts_numerically():
    grouped = category_report.claims_by_categorie(ORIGIN_LABELS)

    assert grouped == {
        "confirme": ["1"],
        "contredit": ["2"],
        "sans_preuve": ["3"],
    }


# ---------------------------------------------------------------------------
# Critère 5 — documents attendus et taux par catégorie
# ---------------------------------------------------------------------------


def test_expected_docs_for_rank_metrics_restricts_to_the_right_label_per_category():
    expected = category_report.expected_docs_for_rank_metrics(ORIGIN_LABELS)

    assert expected["confirme"] == {"1": {"d1"}}
    assert expected["contredit"] == {"2": {"d2"}}
    assert expected["sans_preuve"] == {"3": {"d3"}}


def test_category_rank_rates_reads_rank_from_retrieved_top100():
    expected = category_report.expected_docs_for_rank_metrics(ORIGIN_LABELS)

    rates = category_report.category_rank_rates(SANS_A, expected)

    assert rates["confirme"] == {"n": 1, "rang1": 1.0, "top10": 1.0, "top100": 1.0}
    assert rates["contredit"] == {"n": 1, "rang1": 0.0, "top10": 1.0, "top100": 1.0}
    assert rates["sans_preuve"] == {"n": 1, "rang1": 0.0, "top10": 1.0, "top100": 1.0}


# ---------------------------------------------------------------------------
# Critère 6 — nDCG@10 par ensemble de claims
# ---------------------------------------------------------------------------


def test_expected_docs_for_ndcg_sets_matches_the_three_sets():
    expected = category_report.expected_docs_for_ndcg_sets(ORIGIN_LABELS)

    assert expected["300"] == {"1": {"d1"}, "2": {"d2"}, "3": {"d3"}}
    assert expected["188_avec_preuve"] == {"1": {"d1"}, "2": {"d2"}}
    assert expected["112_sans_preuve"] == {"3": {"d3"}}


def test_ndcg_set_values_matches_ndcg_at_k_called_directly_per_claim():
    expected_sets = category_report.expected_docs_for_ndcg_sets(ORIGIN_LABELS)
    queries_by_id = {q["query_id"]: q for q in SANS_A["queries"]}

    def _expected_mean(qids_to_docs: dict[str, set[str]]) -> float:
        values = [
            ndcg_at_k(
                [d["doc_id"] for d in queries_by_id[qid]["retrieved_top100"]], docs, 10
            )
            for qid, docs in qids_to_docs.items()
        ]
        return sum(values) / len(values)

    values = category_report.ndcg_set_values(SANS_A, expected_sets)

    assert values["300"] == pytest.approx(_expected_mean(expected_sets["300"]))
    assert values["188_avec_preuve"] == pytest.approx(
        _expected_mean(expected_sets["188_avec_preuve"])
    )
    assert values["112_sans_preuve"] == pytest.approx(
        _expected_mean(expected_sets["112_sans_preuve"])
    )


# ---------------------------------------------------------------------------
# Critère 8 — effet du reranker par stratégie de base
# ---------------------------------------------------------------------------


def test_base_strategy_name_splits_the_two_known_suffixes():
    assert category_report.base_strategy_name("bm25-document-avec-reranker") == (
        "bm25-document",
        "avec",
    )
    assert category_report.base_strategy_name("bm25-document-sans-reranker") == (
        "bm25-document",
        "sans",
    )
    assert category_report.base_strategy_name("bm25-document") is None


def test_reranker_effect_rows_matches_paired_permutation_test_called_directly():
    expected_sets = category_report.expected_docs_for_ndcg_sets(ORIGIN_LABELS)
    expected_188, expected_112 = (
        expected_sets["188_avec_preuve"],
        expected_sets["112_sans_preuve"],
    )

    rows = category_report.reranker_effect_rows(
        [SANS_A, AVEC_A, SANS_B, AVEC_B], expected_188, expected_112
    )
    by_base = {r["base"]: r for r in rows}
    assert set(by_base) == {"strat-a", "strat-b"}

    qids_188 = sorted(expected_188)
    values_sans = category_report.ndcg_per_query_values(SANS_A, expected_188)
    values_avec = category_report.ndcg_per_query_values(AVEC_A, expected_188)
    expected_cmp_188 = paired_permutation_test(
        [values_sans[q] for q in qids_188],
        [values_avec[q] for q in qids_188],
        n_permutations=category_report.N_PERMUTATIONS,
        seed=category_report.SEED,
    )
    assert by_base["strat-a"]["diff_188"] == pytest.approx(
        expected_cmp_188["mean_diff"]
    )
    assert by_base["strat-a"]["p_188"] == pytest.approx(expected_cmp_188["p_value"])


def test_reranker_effect_rows_skips_a_base_strategy_missing_a_variant():
    lone = _run(
        "strat-c-sans-reranker", {"1": {"d1": 1}, "2": {"d2": 1}, "3": {"d3": 1}}
    )
    expected_sets = category_report.expected_docs_for_ndcg_sets(ORIGIN_LABELS)

    rows = category_report.reranker_effect_rows(
        [lone], expected_sets["188_avec_preuve"], expected_sets["112_sans_preuve"]
    )

    assert rows == []


# ---------------------------------------------------------------------------
# Critère 9 — régénération à l'identique si les runs n'ont pas changé
# ---------------------------------------------------------------------------


def test_write_report_writes_under_v2_grid_with_run_names_and_category_titles(
    _campaign,
):
    path = category_report.write_report()

    assert path == _campaign / "v2-grid" / "RAPPORT-PAR-CATEGORIE.md"
    content = path.read_text(encoding="utf-8")
    for name in (
        "strat-a-sans-reranker",
        "strat-a-avec-reranker",
        "strat-b-sans-reranker",
        "strat-b-avec-reranker",
    ):
        assert name in content
    assert "confirme" in content
    assert "contredit" in content
    assert "sans preuve" in content


def test_write_report_regenerates_identically_when_runs_are_unchanged(_campaign):
    first = category_report.write_report().read_text(encoding="utf-8")
    second = category_report.write_report().read_text(encoding="utf-8")

    assert first == second


def test_write_report_matches_the_committed_v2_grid_report_byte_for_byte():
    """EXE-142, critère 2, sur les fichiers réels : la commande sans campagne
    nommée régénère `results/v2-grid/RAPPORT-PAR-CATEGORIE.md` à l'identique."""
    path = Path("results/v2-grid/RAPPORT-PAR-CATEGORIE.md")
    before = path.read_text(encoding="utf-8")

    written = category_report.write_report()
    after = written.read_text(encoding="utf-8")

    assert written == path
    assert after == before


# ---------------------------------------------------------------------------
# EXE-142, critère 1 — une campagne nommée s'écrit dans son propre dossier
# ---------------------------------------------------------------------------


def test_write_report_for_campaign_writes_under_the_named_campaign_dir(_fab_campaign):
    results_dir, _ = _fab_campaign

    path = category_report.write_report_for_campaign("fab-campagne")

    assert path == results_dir / "fab-campagne" / "RAPPORT-PAR-CATEGORIE.md"
    content = path.read_text(encoding="utf-8")
    assert "Campagne : fab-campagne" in content
    for name in ("strat-a-sans-reranker", "strat-a-avec-reranker"):
        assert name in content


# ---------------------------------------------------------------------------
# EXE-142, critère 3 — écart de nDCG@10 face à un run de référence
# ---------------------------------------------------------------------------


def test_reference_diff_rows_matches_paired_permutation_test_called_directly():
    expected_sets = category_report.expected_docs_for_ndcg_sets(ORIGIN_LABELS)
    expected_188 = expected_sets["188_avec_preuve"]
    runs = [SANS_A, AVEC_A, SANS_B]

    rows = category_report.reference_diff_rows(
        runs, "strat-a-sans-reranker", expected_188
    )

    by_name = {r["run_name"]: r for r in rows}
    assert set(by_name) == {"strat-a-avec-reranker", "strat-b-sans-reranker"}

    qids = sorted(expected_188)
    reference_values = category_report.ndcg_per_query_values(SANS_A, expected_188)
    values_avec_a = category_report.ndcg_per_query_values(AVEC_A, expected_188)
    expected_cmp = paired_permutation_test(
        [reference_values[q] for q in qids],
        [values_avec_a[q] for q in qids],
        n_permutations=category_report.N_PERMUTATIONS,
        seed=category_report.SEED,
    )
    assert by_name["strat-a-avec-reranker"]["diff_188"] == pytest.approx(
        expected_cmp["mean_diff"]
    )
    assert by_name["strat-a-avec-reranker"]["p_188"] == pytest.approx(
        expected_cmp["p_value"]
    )


def test_write_report_for_campaign_adds_reference_section_only_when_named(
    _fab_campaign,
):
    without_reference = category_report.write_report_for_campaign(
        "fab-campagne"
    ).read_text(encoding="utf-8")
    assert "référence" not in without_reference.lower()

    with_reference = category_report.write_report_for_campaign(
        "fab-campagne", reference_run="strat-a-sans-reranker"
    ).read_text(encoding="utf-8")
    assert "strat-a-sans-reranker" in with_reference
    assert "strat-a-avec-reranker" in with_reference
    assert "référence" in with_reference.lower()


# ---------------------------------------------------------------------------
# EXE-142, critère 4 — niveaux de lecture par groupe, campagne nommée
# ---------------------------------------------------------------------------


def test_group_rank_rows_matches_reading_levels_found_counts_called_directly(
    _fab_campaign,
):
    _, fab_runs = _fab_campaign
    from rag_eval_scifact import reading_levels

    goldens = reading_levels.load_goldens(
        reading_levels.ORIGIN_LABELS_PATH, reading_levels.JUGEMENTS_PATH
    )

    rows = category_report.group_rank_rows(fab_runs)

    by_name = {r["run_name"]: r for r in rows}
    sample_run = fab_runs[0]
    expected_5 = reading_levels.found_counts(goldens, sample_run, 5)
    expected_10 = reading_levels.found_counts(goldens, sample_run, 10)
    assert by_name[sample_run["run_name"]]["counts_5"] == expected_5
    assert by_name[sample_run["run_name"]]["counts_10"] == expected_10


def test_write_report_for_campaign_includes_a_group_section_with_both_thresholds(
    _fab_campaign,
):
    content = category_report.write_report_for_campaign("fab-campagne").read_text(
        encoding="utf-8"
    )

    assert "Niveaux de lecture par groupe" in content
    assert "Direct" in content
    assert "Vocabulaire" in content


# ---------------------------------------------------------------------------
# EXE-142, critère 5 — pas de section reranker sans aucun run avec reranker
# ---------------------------------------------------------------------------


def test_write_report_for_campaign_omits_reranker_section_without_any_reranker_run(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(report_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        category_report, "ORIGIN_LABELS_PATH", results_dir / "etiquettes-origine.json"
    )
    (results_dir / "campagne-simple").mkdir(parents=True)
    (results_dir / "etiquettes-origine.json").write_text(
        json.dumps(ORIGIN_LABELS), encoding="utf-8"
    )
    _write_gz(
        results_dir / "campagne-simple" / f"{SANS_A['run_name']}-20261006.json.gz",
        SANS_A,
    )

    content = category_report.write_report_for_campaign("campagne-simple").read_text(
        encoding="utf-8"
    )

    assert "Effet du reranker" not in content


# ---------------------------------------------------------------------------
# EXE-142, critère 6 — campagne nommée sans aucun run
# ---------------------------------------------------------------------------


def test_write_report_for_campaign_raises_when_the_campaign_has_no_runs(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(report_module, "RESULTS_DIR", results_dir)
    (results_dir / "vide").mkdir(parents=True)

    with pytest.raises(category_report.NoRunsFound):
        category_report.write_report_for_campaign("vide")

    assert not (results_dir / "vide" / "RAPPORT-PAR-CATEGORIE.md").exists()
