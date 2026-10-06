"""Niveaux de lecture du juge Claude sur les 339 goldens de v2-grid (EXE-137).

Critère 13 : les critères 2, 5, 6, 8, 10 et 12 sont portés par des tests qui
lisent les fichiers réels du dépôt (`results/etiquettes-origine.json`,
`results/v3-juge/jugements-claude.json`, `results/v2-grid/*.json.gz`) —
aucun run ni jugement fabriqué pour ces tests, tous les nombres attendus sont
ceux relevés par le chat le 6 octobre 2026 et cités dans le ticket.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval_scifact import reading_levels
from rag_eval_scifact.compare import load_run
from rag_eval_scifact.report import load_campaign_runs

V2_GRID_DIR = Path("results/v2-grid")
QWEN_PASSAGES_SANS = (
    V2_GRID_DIR / "dense-qwen3-passages-sans-reranker-2026-10-03T09-56-14-983855"
    ".json.gz"
)
QWEN_PASSAGES_AVEC = (
    V2_GRID_DIR / "dense-qwen3-passages-avec-reranker-2026-10-03T11-04-52-196737"
    ".json.gz"
)
BM25_DOCUMENT_SANS = (
    V2_GRID_DIR / "bm25-document-sans-reranker-2026-10-03T09-42-17-772165.json.gz"
)


def _goldens() -> list[dict]:
    goldens = reading_levels.load_goldens()
    assert goldens is not None
    return goldens


# ---------------------------------------------------------------------------
# Critère 2 — effectif des 7 groupes, total 339
# ---------------------------------------------------------------------------


def test_group_counts_matches_the_seven_effectifs_of_the_ticket():
    counts = reading_levels.group_counts(_goldens())

    assert counts == {
        "direct": 60,
        "vocabulaire": 55,
        "raisonnement": 71,
        "avec_preuve_ne_repond_pas": 18,
        "sans_preuve_repond": 22,
        "sans_preuve_ne_repond_pas": 105,
        "non_juge": 8,
    }
    assert sum(counts.values()) == 339


# ---------------------------------------------------------------------------
# Critères 5 et 6 — tableau au seuil « 5 premiers », deux runs de contrôle
# ---------------------------------------------------------------------------


def test_found_counts_at_5_premiers_for_qwen_passages_sans_reranker():
    run_data = load_run(QWEN_PASSAGES_SANS)

    counts = reading_levels.found_counts(_goldens(), run_data, threshold=5)

    assert counts["direct"] == {"found": 60, "total": 60}
    assert counts["vocabulaire"] == {"found": 51, "total": 55}
    assert counts["raisonnement"] == {"found": 63, "total": 71}
    assert counts["avec_preuve_ne_repond_pas"] == {"found": 11, "total": 18}
    assert counts["sans_preuve_repond"] == {"found": 17, "total": 22}
    assert counts["sans_preuve_ne_repond_pas"] == {"found": 60, "total": 105}
    assert counts["non_juge"] == {"found": 8, "total": 8}


def test_found_counts_at_5_premiers_for_bm25_document_sans_reranker():
    run_data = load_run(BM25_DOCUMENT_SANS)

    counts = reading_levels.found_counts(_goldens(), run_data, threshold=5)

    assert counts["direct"] == {"found": 56, "total": 60}
    assert counts["vocabulaire"] == {"found": 46, "total": 55}
    assert counts["raisonnement"] == {"found": 55, "total": 71}
    assert counts["avec_preuve_ne_repond_pas"] == {"found": 11, "total": 18}
    assert counts["sans_preuve_repond"] == {"found": 14, "total": 22}
    assert counts["sans_preuve_ne_repond_pas"] == {"found": 45, "total": 105}
    assert counts["non_juge"] == {"found": 7, "total": 8}


def test_build_table_rows_has_one_row_per_run_of_v2_grid():
    runs = load_campaign_runs("v2-grid")

    rows = reading_levels.build_table_rows(_goldens(), runs, threshold=5)

    assert len(rows) == 34
    by_name = {row["run_name"]: row for row in rows}
    assert by_name["dense-qwen3-passages-sans-reranker"]["counts"]["direct"] == {
        "found": 60,
        "total": 60,
    }


# ---------------------------------------------------------------------------
# Critère 4 — le seuil change le compte retrouvé
# ---------------------------------------------------------------------------


def test_found_counts_changes_with_the_threshold():
    run_data = load_run(QWEN_PASSAGES_SANS)
    goldens = _goldens()

    at_1 = reading_levels.found_counts(goldens, run_data, threshold=1)
    at_5 = reading_levels.found_counts(goldens, run_data, threshold=5)
    at_10 = reading_levels.found_counts(goldens, run_data, threshold=10)

    assert at_1["raisonnement"]["found"] <= at_5["raisonnement"]["found"]
    assert at_5["raisonnement"]["found"] <= at_10["raisonnement"]["found"]
    assert at_5["raisonnement"]["found"] == 63
    assert at_10["raisonnement"]["found"] == 70


# ---------------------------------------------------------------------------
# Critère 7 — tri d'une colonne, de la meilleure à la moins bonne
# ---------------------------------------------------------------------------


def test_sort_rows_by_group_orders_from_best_to_worst_rate():
    rows = [
        {"run_name": "a", "counts": {"direct": {"found": 1, "total": 10}}},
        {"run_name": "b", "counts": {"direct": {"found": 9, "total": 10}}},
        {"run_name": "c", "counts": {"direct": {"found": 5, "total": 10}}},
    ]

    sorted_rows = reading_levels.sort_rows_by_group(rows, "direct")

    assert [row["run_name"] for row in sorted_rows] == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# Critère 8 — restriction confirme / contredit
# ---------------------------------------------------------------------------


def test_filter_goldens_by_categorie_confirme_matches_the_ticket_counts():
    confirme = reading_levels.filter_goldens_by_categorie(_goldens(), "confirme")
    counts = reading_levels.group_counts(confirme)

    assert counts["direct"] == 34
    assert counts["vocabulaire"] == 49
    assert counts["raisonnement"] == 37


def test_filter_goldens_by_categorie_contredit_matches_the_ticket_counts():
    contredit = reading_levels.filter_goldens_by_categorie(_goldens(), "contredit")
    counts = reading_levels.group_counts(contredit)

    assert counts["direct"] == 26
    assert counts["vocabulaire"] == 6
    assert counts["raisonnement"] == 34


def test_filter_goldens_by_categorie_never_mixes_confirme_and_contredit():
    goldens = _goldens()
    confirme = reading_levels.filter_goldens_by_categorie(goldens, "confirme")
    contredit = reading_levels.filter_goldens_by_categorie(goldens, "contredit")

    confirme_ids = {(g["query_id"], g["doc_id"]) for g in confirme}
    contredit_ids = {(g["query_id"], g["doc_id"]) for g in contredit}
    assert confirme_ids.isdisjoint(contredit_ids)
    assert all(g["label"] == "SUPPORT" for g in confirme)
    assert all(g["label"] == "CONTRADICT" for g in contredit)


# ---------------------------------------------------------------------------
# Critère 10 — effet du reranker, 17 stratégies de base
# ---------------------------------------------------------------------------


def test_reranker_diff_rows_covers_the_17_base_strategies():
    runs = load_campaign_runs("v2-grid")

    rows = reading_levels.reranker_diff_rows(_goldens(), runs, threshold=10)

    assert len(rows) == 17


def test_reranker_diff_rows_matches_the_ticket_example_for_qwen_passages():
    runs = load_campaign_runs("v2-grid")

    rows = reading_levels.reranker_diff_rows(_goldens(), runs, threshold=10)
    by_base = {row["base"]: row for row in rows}

    assert by_base["dense-qwen3-passages"]["diffs"]["raisonnement"] == -7
    assert by_base["dense-qwen3-passages"]["diffs"]["direct"] == 0


# ---------------------------------------------------------------------------
# Critère 12 — les goldens hors seuil apparaissent en premier
# ---------------------------------------------------------------------------


def test_sort_rows_beyond_threshold_first_puts_every_not_found_before_found():
    run_data = load_run(QWEN_PASSAGES_SANS)
    rows = reading_levels.build_group_detail_rows(_goldens(), run_data, "raisonnement")
    assert any(r["rank"] is None or r["rank"] > 5 for r in rows)
    assert any(r["rank"] is not None and r["rank"] <= 5 for r in rows)

    sorted_rows = reading_levels.sort_rows_beyond_threshold_first(rows, threshold=5)

    found_flags = [row["rank"] is not None and row["rank"] <= 5 for row in sorted_rows]
    # Toutes les entrées non retrouvées (False) précèdent toutes les entrées
    # retrouvées (True) — une liste de booléens déjà triée.
    assert found_flags == sorted(found_flags)


# ---------------------------------------------------------------------------
# Fichiers manquants — aucune erreur, juste `None`
# ---------------------------------------------------------------------------


def test_load_goldens_returns_none_when_origin_labels_file_is_missing(tmp_path):
    goldens = reading_levels.load_goldens(
        origin_labels_path=tmp_path / "absent.json",
        jugements_path=reading_levels.JUGEMENTS_PATH,
    )
    assert goldens is None


def test_load_goldens_returns_none_when_judgments_file_is_missing(tmp_path):
    goldens = reading_levels.load_goldens(
        origin_labels_path=reading_levels.ORIGIN_LABELS_PATH,
        jugements_path=tmp_path / "absent.json",
    )
    assert goldens is None


# ---------------------------------------------------------------------------
# Critère 11 (données) — contenu d'une ligne de détail
# ---------------------------------------------------------------------------


def test_build_group_detail_rows_has_claim_text_doc_title_rank_and_judgment():
    run_data = load_run(QWEN_PASSAGES_SANS)
    corpus = {"31715818": {"title": "un titre", "text": "un texte"}}
    goldens = _goldens()

    rows = reading_levels.build_group_detail_rows(
        goldens, run_data, "sans_preuve_ne_repond_pas", corpus=corpus
    )

    assert rows
    row = next(r for r in rows if r["doc_id"] == "31715818")
    assert row["claim_text"]
    assert row["doc_title"] == "un titre"
    assert row["verdict"] == "NOT_ENOUGH_INFO"


def test_build_group_detail_rows_reports_out_of_top_100_as_none_rank():
    run_data = load_run(QWEN_PASSAGES_SANS)
    goldens = _goldens()

    rows = reading_levels.build_group_detail_rows(
        goldens, run_data, "sans_preuve_ne_repond_pas"
    )

    assert any(r["rank"] is None for r in rows)


# ---------------------------------------------------------------------------
# H2 — accord du juge avec les annotateurs, lu dans le rapport de campagne
# ---------------------------------------------------------------------------


def test_judge_agreement_with_annotators_matches_the_campaign_report():
    agreement = reading_levels.judge_agreement_with_annotators()

    assert agreement is not None
    assert round(agreement, 4) == 0.8580


def test_judge_agreement_with_annotators_returns_none_when_paires_file_is_missing(
    tmp_path,
):
    agreement = reading_levels.judge_agreement_with_annotators(
        paires_path=tmp_path / "absent.json",
        jugements_path=reading_levels.JUGEMENTS_PATH,
    )
    assert agreement is None
