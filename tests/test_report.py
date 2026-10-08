"""Tests du rapport de campagne (EXE-91, critères 6, 7, 9).

Campagne fabriquée de deux runs face à un run de référence fabriqué sous
`results/v1-rejeu/baseline-*.json.gz` : aucun modèle d'embedding n'est chargé,
tout est du JSON gzip écrit à la main au format `versioning.md`. L'écart et la
p-value attendus sont calculés en appelant directement `compare_runs` (déjà
testé par `test_compare.py`), pour ne pas recalculer la formule à la main ici.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rag_eval_scifact import report as report_module
from rag_eval_scifact.compare import compare_runs

QUERY_IDS = ["q1", "q2", "q3"]
QRELS = {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}}

SIX_METRICS_A = {
    "recall@1": 0.5,
    "recall@5": 0.7,
    "recall@10": 0.8,
    "recall@100": 0.9,
    "ndcg@10": 0.6,
    "mrr": 0.55,
}
SIX_METRICS_B = {
    "recall@1": 0.3,
    "recall@5": 0.5,
    "recall@10": 0.6,
    "recall@100": 0.8,
    "ndcg@10": 0.9,
    "mrr": 0.85,
}
EXTENDED = {
    "bucket_found_at_10_perfect": 1.0,
    "bucket_found_at_10_near_miss": 0.5,
    "bucket_found_at_10_deep_miss": 0.0,
    "bucket_found_at_10_miss_100": 0.0,
    "truncated_pct": 10.0,
    "avg_retrieval_latency_ms": 5.0,
    "indexing_duration_seconds": 2.0,
}

# Perfect : le doc attendu est en rang 1 pour chaque requête (nDCG@10 = 1.0).
PERFECT_RETRIEVAL = {
    qid: [{"doc_id": doc_id, "rank": 1, "score": 0.9}]
    for qid, doc_id in zip(QUERY_IDS, ["d1", "d2", "d3"])
}
# Dégradé : le doc attendu de q1 tombe en rang 2, derrière un distracteur.
DEGRADED_RETRIEVAL = {
    "q1": [
        {"doc_id": "distracteur", "rank": 1, "score": 0.95},
        {"doc_id": "d1", "rank": 2, "score": 0.5},
    ],
    "q2": PERFECT_RETRIEVAL["q2"],
    "q3": PERFECT_RETRIEVAL["q3"],
}


def _run_data(
    run_name: str, campagne: str, retrieved_by_query: dict, metrics: dict
) -> dict:
    queries = [
        {
            "query_id": qid,
            "query_text": f"question {qid}",
            "expected_docs": [
                {"doc_id": doc_id, "token_count": 100} for doc_id in sorted(QRELS[qid])
            ],
            "retrieved_top100": retrieved_by_query[qid],
            "per_query_metrics": {},
        }
        for qid in QUERY_IDS
    ]
    return {
        "campagne": campagne,
        "run_name": run_name,
        "date": "2026-10-02T00:00:00",
        "dataset_hash": "sha256:fake",
        "config": {},
        "metrics": metrics,
        "extended_metrics": EXTENDED,
        "queries": queries,
    }


def _write_gz(path: Path, run_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(run_data, f)


BASELINE_DATA = _run_data("baseline", "v1-rejeu", PERFECT_RETRIEVAL, SIX_METRICS_A)
RUN_A_DATA = _run_data("run-a", "campagne-test", PERFECT_RETRIEVAL, SIX_METRICS_A)
RUN_B_DATA = _run_data("run-b", "campagne-test", DEGRADED_RETRIEVAL, SIX_METRICS_B)


@pytest.fixture
def _campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path / "results")
    results_dir = tmp_path / "results"

    _write_gz(
        results_dir / "v1-rejeu" / "baseline-2026-10-01T00-00-00-000000.json.gz",
        BASELINE_DATA,
    )
    _write_gz(
        results_dir / "campagne-test" / "run-a-2026-10-02T00-00-00-000000.json.gz",
        RUN_A_DATA,
    )
    _write_gz(
        results_dir / "campagne-test" / "run-b-2026-10-02T00-00-01-000000.json.gz",
        RUN_B_DATA,
    )
    return results_dir


# ---------------------------------------------------------------------------
# Critère 6 — tableau trié par nDCG@10 décroissant, avec les champs attendus
# ---------------------------------------------------------------------------


def test_report_sorts_by_ndcg10_descending_and_writes_under_campaign_dir(_campaign):
    path = report_module.write_report("campagne-test")

    assert path == _campaign / "campagne-test" / "RAPPORT.md"
    content = path.read_text(encoding="utf-8")

    pos_a = content.index("run-a")
    pos_b = content.index("run-b")
    assert pos_b < pos_a  # run-b (nDCG@10=0.9) avant run-a (nDCG@10=0.6)


def test_report_contains_the_six_metrics_and_extended_fields_per_run(_campaign):
    content = report_module.write_report("campagne-test").read_text(encoding="utf-8")

    for value in SIX_METRICS_A.values():
        assert f"{value:.4f}" in content
    for value in SIX_METRICS_B.values():
        assert f"{value:.4f}" in content
    assert f"{EXTENDED['truncated_pct']:.1f}" in content
    assert f"{EXTENDED['avg_retrieval_latency_ms']:.1f}" in content
    assert f"{EXTENDED['indexing_duration_seconds']:.1f}" in content
    assert f"{EXTENDED['bucket_found_at_10_perfect']:.4f}" in content


def test_report_diff_and_p_value_match_compare_runs_against_baseline(_campaign):
    content = report_module.write_report("campagne-test").read_text(encoding="utf-8")

    expected_ndcg = compare_runs(
        BASELINE_DATA,
        RUN_B_DATA,
        "ndcg@10",
        n_permutations=report_module.N_PERMUTATIONS,
        seed=report_module.SEED,
    )
    expected_mrr = compare_runs(
        BASELINE_DATA,
        RUN_B_DATA,
        "mrr",
        n_permutations=report_module.N_PERMUTATIONS,
        seed=report_module.SEED,
    )

    assert f"{expected_ndcg['mean_diff']:+.4f}" in content
    assert f"{expected_ndcg['p_value']:.4f}" in content
    assert f"{expected_mrr['mean_diff']:+.4f}" in content
    assert f"{expected_mrr['p_value']:.4f}" in content


# ---------------------------------------------------------------------------
# Critère 7 — régénération à l'identique si les fichiers n'ont pas changé
# ---------------------------------------------------------------------------


def test_report_regenerates_identically_when_results_are_unchanged(_campaign):
    first = report_module.write_report("campagne-test").read_text(encoding="utf-8")
    second = report_module.write_report("campagne-test").read_text(encoding="utf-8")

    assert first == second


# ---------------------------------------------------------------------------
# EXE-156 — un fichier qui n'est pas un run, rangé dans le dossier d'une
# campagne, ne casse ni le chargement ni le rapport.
# ---------------------------------------------------------------------------


def test_load_campaign_runs_ignores_a_non_run_shaped_json_file(_campaign):
    (_campaign / "campagne-test" / "hyde.json").write_text(
        json.dumps([{"claim_id": "1", "hyde_text": "faux résumé"}]), encoding="utf-8"
    )

    runs = report_module.load_campaign_runs("campagne-test")

    assert {r["run_name"] for r in runs} == {"run-a", "run-b"}


def test_write_report_ignores_a_non_run_shaped_json_file_in_the_campaign_dir(
    _campaign,
):
    (_campaign / "campagne-test" / "hyde.json").write_text(
        json.dumps([{"claim_id": "1", "hyde_text": "faux résumé"}]), encoding="utf-8"
    )

    content = report_module.write_report("campagne-test").read_text(encoding="utf-8")

    assert "hyde" not in content
    assert "run-a" in content
    assert "run-b" in content


def test_load_campaign_runs_raises_naming_a_malformed_gz_file(_campaign):
    bad_path = _campaign / "campagne-test" / "pas-un-run-2026-10-03.json.gz"
    _write_gz(bad_path, [{"not": "a run"}])

    with pytest.raises(report_module.InvalidRunFile) as exc_info:
        report_module.load_campaign_runs("campagne-test")

    assert bad_path.name in str(exc_info.value)
