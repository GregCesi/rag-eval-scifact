"""Tests de la comparaison de deux runs par test apparié (EXE-86).

Couvre le chargement (.json et .json.gz) et le recalcul par requête de nDCG@10
et MRR depuis les artefacts de run — ces valeurs ne sont pas stockées par
requête dans le format run (seuls found@k et best_rank le sont), donc
recalculées via rag_eval_scifact.metrics sans en changer le calcul. Aucun
modèle d'embedding chargé : runs fabriqués à la main, comme test_campaign.py.
"""

from __future__ import annotations

import gzip
import json

from rag_eval_scifact.compare import compare_runs, load_run, per_query_metric_values


def _fake_run() -> dict:
    return {
        "campagne": "campagne-test",
        "run_name": "run-1",
        "queries": [
            {
                "query_id": "q1",
                "expected_docs": [{"doc_id": "d1", "token_count": 10}],
                "retrieved_top100": [
                    {"doc_id": "d1", "rank": 1, "score": 0.9},
                    {"doc_id": "d2", "rank": 2, "score": 0.5},
                ],
            },
            {
                "query_id": "q2",
                "expected_docs": [{"doc_id": "d3", "token_count": 10}],
                "retrieved_top100": [
                    {"doc_id": "d9", "rank": 1, "score": 0.8},
                    {"doc_id": "d3", "rank": 2, "score": 0.4},
                ],
            },
        ],
    }


def test_per_query_metric_values_recomputes_ndcg_and_mrr_from_artifact():
    run_data = _fake_run()

    ndcg_values = per_query_metric_values(run_data, "ndcg@10")
    mrr_values = per_query_metric_values(run_data, "mrr")

    assert ndcg_values["q1"] == 1.0  # doc pertinent au rang 1
    assert mrr_values["q2"] == 0.5  # doc pertinent au rang 2


def test_load_run_reads_both_gzip_and_plain_json(tmp_path):
    run_data = _fake_run()

    plain_path = tmp_path / "run.json"
    plain_path.write_text(json.dumps(run_data), encoding="utf-8")
    gz_path = tmp_path / "run.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(run_data, f)

    assert load_run(plain_path) == run_data
    assert load_run(gz_path) == run_data


# ---------------------------------------------------------------------------
# Critère 6 (pipeline complet) — un run comparé à lui-même, pour les deux métriques
# ---------------------------------------------------------------------------


def test_comparing_a_run_to_itself_gives_zero_diff_and_p_value_one():
    run_data = _fake_run()

    for metric in ("ndcg@10", "mrr"):
        result = compare_runs(run_data, run_data, metric, n_permutations=1000, seed=0)
        assert result["mean_diff"] == 0.0
        assert result["p_value"] == 1.0
