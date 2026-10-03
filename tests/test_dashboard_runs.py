"""Choix de la source puis du run, et lecture des pages sur le run choisi —
pas seulement sur les deux runs v1 (EXE-113, critères 1, 3 et 5).

Aucun modèle chargé : la liste des sources/runs est une lecture de fichiers,
et les chiffres des pages sont déjà écrits dans les fichiers de run commités.
"""

from __future__ import annotations

from pathlib import Path

import dashboard
from rag_eval_scifact.compare import load_run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_v2_grid_run(run_name: str) -> dict:
    (path,) = RESULTS_DIR.glob(f"v2-grid/{run_name}-*.json.gz")
    return load_run(path)


# ─────────────────────────────────────────────
# Critère 1 — choisir d'abord une source, puis un run de cette source
# ─────────────────────────────────────────────


def test_list_sources_offers_v1_then_each_campaign(tmp_path):
    (tmp_path / "v1-dense-a.json").write_text("{}")
    (tmp_path / "v1-annotations.json").write_text("{}")  # pas une source
    campaign_a = tmp_path / "v1-rejeu"
    campaign_a.mkdir()
    (campaign_a / "baseline-1.json.gz").write_bytes(b"")
    campaign_b = tmp_path / "v2-grid"
    campaign_b.mkdir()
    (campaign_b / "run-1.json.gz").write_bytes(b"")

    assert dashboard.list_sources(tmp_path) == ["v1", "v1-rejeu", "v2-grid"]


def test_runs_for_source_v1_lists_only_v1_dense_files(tmp_path):
    (tmp_path / "v1-dense-a.json").write_text("{}")
    (tmp_path / "v1-dense-b.json").write_text("{}")
    (tmp_path / "v1-buckets.json").write_text("{}")
    (tmp_path / "v1-annotations.json").write_text("{}")

    runs = dashboard.runs_for_source(tmp_path, "v1")

    assert {p.name for p in runs} == {"v1-dense-a.json", "v1-dense-b.json"}


def test_runs_for_source_campaign_lists_its_runs_only_not_derived_files(tmp_path):
    campaign = tmp_path / "v2-grid"
    campaign.mkdir()
    (campaign / "run-a.json.gz").write_bytes(b"")
    (campaign / "run-b.json.gz").write_bytes(b"")
    (campaign / "RAPPORT.md").write_text("")
    passages = campaign / "passages"
    passages.mkdir()
    (passages / "run-a.passages.json.gz").write_bytes(b"")

    runs = dashboard.runs_for_source(tmp_path, "v2-grid")

    assert {p.name for p in runs} == {"run-a.json.gz", "run-b.json.gz"}


def test_v2_grid_campaign_proposes_its_34_runs():
    runs = dashboard.runs_for_source(RESULTS_DIR, "v2-grid")

    assert len(runs) == 34


# ─────────────────────────────────────────────
# Critère 3 — Vue d'ensemble sur un run de campagne choisi dans la barre latérale
# ─────────────────────────────────────────────


def test_overview_metrics_and_rank_distribution_for_chosen_campaign_run():
    run = _load_v2_grid_run("dense-qwen3-passages-sans-reranker")

    assert round(run["metrics"]["ndcg@10"], 4) == 0.7319
    assert round(run["metrics"]["mrr"], 4) == 0.7013

    retriever_cfg = dashboard.effective_retriever_config(run)
    df = dashboard.queries_to_dataframe(run["queries"], retriever_cfg)
    counts = df["category"].value_counts()

    assert len(df) == 300
    assert counts["Rang 1"] == 181
    assert counts["Rang 2-5"] == 66
    assert counts["Rang 6-10"] == 11
    assert counts["Rang 11-50"] == 21
    assert counts["Rang 51-100"] == 7
    assert counts["Non trouvé"] == 14


# ─────────────────────────────────────────────
# Critère 5 — le rang/score d'un claim vient du run choisi, jamais de v1-buckets.json
# ─────────────────────────────────────────────


def test_claim_rank_and_score_reads_the_chosen_run_not_the_frozen_v1_bucket():
    run = _load_v2_grid_run("dense-qwen3-passages-sans-reranker")
    q70 = next(q for q in run["queries"] if q["query_id"] == "70")

    rank, score = dashboard.claim_rank_and_score(q70)

    # v1-buckets.json classe le claim 70 en deep_miss, rang 15, score 0.5350 —
    # figé sur le run v1. Sur ce run de campagne, il remonte rang 1.
    assert rank == 1
    assert round(score, 4) == 0.7105


def test_claim_rank_and_score_is_none_when_not_found_in_top_100():
    q = {
        "per_query_metrics": {"best_rank": None},
        "retrieved_top100": [],
    }

    rank, score = dashboard.claim_rank_and_score(q)

    assert rank is None
    assert score is None
