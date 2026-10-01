"""Tests des artefacts d'un run de campagne (EXE-85).

Ne chargent aucun modèle d'embedding : le retrieval est fabriqué à la main,
`rag_eval_scifact.campaign.run_campaign` ne fait que calculer les métriques et
écrire les artefacts. Couvre les critères d'acceptation 3 (append RESULTS.md),
4 (format JSON gzip), 5 (override Hydra top_k reflété dans la config résolue)
et 6 (deux runs successifs, aucun fichier écrasé).
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from rag_eval_scifact.campaign import run_campaign
from rag_eval_scifact.retrieve import RetrievalResult

RUN_DATE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _fake_results() -> tuple[list[RetrievalResult], dict[str, set[str]]]:
    """Trois requêtes fabriquées, aucune n'invoque de modèle d'embedding."""
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="question un",
            retrieved=[
                {"doc_id": "d1", "rank": 1, "score": 0.9},
                {"doc_id": "d2", "rank": 2, "score": 0.5},
            ],
        ),
        RetrievalResult(
            query_id="q2",
            query_text="question deux",
            retrieved=[
                {"doc_id": "d9", "rank": 1, "score": 0.8},
                {"doc_id": "d3", "rank": 2, "score": 0.4},
            ],
        ),
        RetrievalResult(
            query_id="q3",
            query_text="question trois",
            retrieved=[
                {"doc_id": "d5", "rank": 1, "score": 0.7},
            ],
        ),
    ]
    qrels = {"q1": {"d1"}, "q2": {"d3"}, "q3": {"d5"}}
    return results, qrels


def _base_config() -> dict:
    return {
        "campagne": "campagne-test",
        "run_name": "run-test",
        "retriever": {
            "name": "dense",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "max_seq_length": 256,
            "dim": 384,
        },
        "fusion": None,
        "rerank": None,
        "split": "test",
        "top_k": 100,
    }


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Les artefacts s'écrivent dans tmp_path, jamais dans le dépôt réel."""
    from rag_eval_scifact import campaign

    monkeypatch.setattr(campaign, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(campaign, "RESULTS_MD", tmp_path / "RESULTS.md")
    yield tmp_path


# ---------------------------------------------------------------------------
# Critère 3 — RESULTS.md : exactement une ligne de plus, le reste inchangé
# ---------------------------------------------------------------------------


def test_run_campaign_appends_exactly_one_results_md_line(_isolate_artifacts):
    results, qrels = _fake_results()
    config = _base_config()

    run_campaign(
        "campagne-test", "run-1", config, results, qrels, "sha256:abc", RUN_DATE
    )
    results_md = _isolate_artifacts / "RESULTS.md"
    before = results_md.read_text(encoding="utf-8").splitlines()

    run_campaign(
        "campagne-test", "run-2", config, results, qrels, "sha256:abc", RUN_DATE
    )
    after = results_md.read_text(encoding="utf-8").splitlines()

    assert after[: len(before)] == before
    assert len(after) == len(before) + 1


# ---------------------------------------------------------------------------
# Critère 4 — fichier gzip sous results/<campagne>/ au format versioning.md
# ---------------------------------------------------------------------------


def test_run_campaign_writes_gzip_artifact_with_required_fields(_isolate_artifacts):
    results, qrels = _fake_results()
    config = _base_config()

    outcome = run_campaign(
        "campagne-test",
        "run-1",
        config,
        results,
        qrels,
        "sha256:abc",
        RUN_DATE,
        token_counts={"d1": 120, "d3": 400, "d5": 90},
    )

    json_path = outcome["json_path"]
    assert json_path.parent == _isolate_artifacts / "results" / "campagne-test"
    assert json_path.name.endswith(".json.gz")

    with gzip.open(json_path, "rt", encoding="utf-8") as f:
        run_data = json.load(f)

    assert run_data["campagne"] == "campagne-test"
    assert run_data["run_name"] == "run-1"
    assert run_data["config"] == config
    assert run_data["dataset_hash"] == "sha256:abc"
    assert set(run_data["metrics"]) == {
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
        "ndcg@10",
        "mrr",
    }
    assert len(run_data["queries"]) == len(results)

    query = next(q for q in run_data["queries"] if q["query_id"] == "q1")
    assert query["expected_docs"] == [{"doc_id": "d1", "token_count": 120}]
    assert query["retrieved_top100"] == results[0].retrieved
    assert query["per_query_metrics"]["found@1"] is True


# ---------------------------------------------------------------------------
# Critère 5 — override CLI top_k=10 reflété dans la config résolue
# ---------------------------------------------------------------------------


def test_top_k_override_is_reflected_in_resolved_config(_isolate_artifacts):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["top_k=10"])

    assert cfg.top_k == 10
    resolved = OmegaConf.to_container(cfg, resolve=True)

    results, qrels = _fake_results()
    outcome = run_campaign(
        cfg.campagne,
        cfg.run_name,
        resolved,
        results,
        qrels,
        "sha256:abc",
        RUN_DATE,
    )

    with gzip.open(outcome["json_path"], "rt", encoding="utf-8") as f:
        run_data = json.load(f)

    assert run_data["config"]["top_k"] == 10


# ---------------------------------------------------------------------------
# Critère 6 — deux runs successifs, deux fichiers distincts, aucun écrasé
# ---------------------------------------------------------------------------


def test_two_successive_runs_never_overwrite_each_other(_isolate_artifacts):
    results, qrels = _fake_results()
    config = _base_config()

    outcome_1 = run_campaign(
        "campagne-test", "run-1", config, results, qrels, "sha256:abc", RUN_DATE
    )
    outcome_2 = run_campaign(
        "campagne-test", "run-1", config, results, qrels, "sha256:abc", RUN_DATE
    )

    path_1, path_2 = outcome_1["json_path"], outcome_2["json_path"]
    assert path_1 != path_2
    assert path_1.exists()
    assert path_2.exists()

    with gzip.open(path_1, "rt", encoding="utf-8") as f:
        assert json.load(f)["run_name"] == "run-1"
    with gzip.open(path_2, "rt", encoding="utf-8") as f:
        assert json.load(f)["run_name"] == "run-1"
