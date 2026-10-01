"""Charge deux runs de campagne et les compare par un test de randomisation apparié.

nDCG@10 et MRR ne sont pas stockées par requête dans le format run
(`.claude/rules/versioning.md` : seuls found@k et best_rank le sont) : ce
module les recalcule depuis `retrieved_top100` + `expected_docs`, en
réutilisant `rag_eval_scifact.metrics` sans en changer le calcul.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from rag_eval_scifact.metrics import mrr, ndcg_at_k
from rag_eval_scifact.stats import paired_permutation_test

SUPPORTED_METRICS = ("ndcg@10", "mrr")


def load_run(path: Path | str) -> dict:
    """Charge un run de campagne, compressé (.json.gz) ou non (v1 historique)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_query_metric_values(run_data: dict, metric: str) -> dict[str, float]:
    """Recalcule `metric` (ndcg@10 ou mrr) par requête depuis un run chargé."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"métrique non supportée pour la comparaison : {metric}")

    values: dict[str, float] = {}
    for query in run_data["queries"]:
        retrieved_ids = [d["doc_id"] for d in query["retrieved_top100"]]
        relevant_ids = {d["doc_id"] for d in query["expected_docs"]}
        if metric == "ndcg@10":
            values[query["query_id"]] = ndcg_at_k(retrieved_ids, relevant_ids, 10)
        else:
            values[query["query_id"]] = mrr(retrieved_ids, relevant_ids)
    return values


def compare_runs(
    run_a: dict,
    run_b: dict,
    metric: str,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """Compare deux runs déjà chargés sur `metric`, par test de randomisation apparié."""
    values_a = per_query_metric_values(run_a, metric)
    values_b = per_query_metric_values(run_b, metric)

    if values_a.keys() != values_b.keys():
        raise ValueError("les deux runs ne portent pas sur les mêmes requêtes")

    query_ids = sorted(values_a)
    series_a = [values_a[q] for q in query_ids]
    series_b = [values_b[q] for q in query_ids]
    return paired_permutation_test(
        series_a, series_b, n_permutations=n_permutations, seed=seed
    )
