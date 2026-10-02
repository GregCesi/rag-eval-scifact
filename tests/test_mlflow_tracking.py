"""Tests du suivi MLflow d'un run de campagne (EXE-86, critères 1 à 4).

N'écrivent jamais dans le `mlruns/` du dépôt : chaque test pointe
`mlflow.set_tracking_uri` vers un dossier temporaire, restauré après coup. Ne
chargent aucun modèle d'embedding : l'artefact comparé est fabriqué à la main,
comme dans test_campaign.py.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import mlflow
import pytest
from mlflow.entities import SpanType
from mlflow.tracking import MlflowClient

from rag_eval_scifact import mlflow_tracking
from rag_eval_scifact.mlflow_tracking import log_campaign_run, log_query_traces
from rag_eval_scifact.retrieve import RetrievalResult

CONFIG = {
    "campagne": "campagne-test",
    "run_name": "run-1",
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

METRICS = {
    "recall@1": 0.4823,
    "recall@5": 0.7379,
    "recall@10": 0.7833,
    "recall@100": 0.9250,
    "ndcg@10": 0.6451,
    "mrr": 0.6110,
}


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking(tmp_path):
    """Pointe MLflow vers un SQLite temporaire : jamais le mlflow.db du dépôt."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(None)


def _write_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "campagne-test" / "run-1-2026-10-01T12-00-00-000000.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"campagne": "campagne-test"}, ensure_ascii=False).encode(
        "utf-8"
    )
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return path


# ---------------------------------------------------------------------------
# Critère 1 — l'expérience MLflow porte le nom de la campagne
# ---------------------------------------------------------------------------


def test_run_is_filed_under_an_experiment_named_after_the_campaign(tmp_path):
    json_path = _write_artifact(tmp_path)

    log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    client = MlflowClient()
    experiment = client.get_experiment_by_name("campagne-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_name == "run-1"


# ---------------------------------------------------------------------------
# Critère 2 — chaque valeur de la config résolue est un paramètre du run
# ---------------------------------------------------------------------------


def test_run_params_cover_every_resolved_config_value(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    params = MlflowClient().get_run(run_id).data.params
    assert params["campagne"] == "campagne-test"
    assert params["run_name"] == "run-1"
    assert params["retriever.name"] == "dense"
    assert params["retriever.model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert params["retriever.max_seq_length"] == "256"
    assert params["retriever.dim"] == "384"
    assert params["fusion"] == "None"
    assert params["rerank"] == "None"
    assert params["split"] == "test"
    assert params["top_k"] == "100"


# ---------------------------------------------------------------------------
# Critère 3 — les 6 métriques du run MLflow valent celles de RESULTS.md
# ---------------------------------------------------------------------------


def test_run_metrics_equal_the_six_results_md_metrics(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    mlflow_metrics = MlflowClient().get_run(run_id).data.metrics
    assert len(mlflow_metrics) == len(METRICS)
    for name, value in METRICS.items():
        assert mlflow_metrics[name.replace("@", "_at_")] == pytest.approx(value)


# ---------------------------------------------------------------------------
# Critère 4 — l'artefact attaché est identique octet pour octet au fichier results/
# ---------------------------------------------------------------------------


def test_artifact_attached_is_byte_identical_to_results_file(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    client = MlflowClient()
    downloaded = client.download_artifacts(run_id, json_path.name, str(tmp_path / "dl"))
    assert Path(downloaded).read_bytes() == json_path.read_bytes()


# ---------------------------------------------------------------------------
# EXE-90 — extended_metrics (bucket v1, part tronquée, latence, indexation)
# journalisées comme métriques MLflow en plus des 6 historiques
# ---------------------------------------------------------------------------


def test_extended_metrics_are_logged_alongside_the_six_metrics(tmp_path):
    json_path = _write_artifact(tmp_path)
    extended_metrics = {
        "bucket_found_at_10_perfect": 1.0,
        "bucket_found_at_10_near_miss": 1.0,
        "bucket_found_at_10_deep_miss": 0.0,
        "bucket_found_at_10_miss_100": 0.0,
        "truncated_pct": 71.0,
        "avg_retrieval_latency_ms": 12.5,
        "indexing_duration_seconds": 3.2,
    }

    run_id = log_campaign_run(
        "campagne-test", "run-1", CONFIG, METRICS, json_path, extended_metrics
    )

    mlflow_metrics = MlflowClient().get_run(run_id).data.metrics
    assert mlflow_metrics["bucket_found_at_10_perfect"] == pytest.approx(1.0)
    assert mlflow_metrics["truncated_pct"] == pytest.approx(71.0)
    assert mlflow_metrics["avg_retrieval_latency_ms"] == pytest.approx(12.5)
    assert mlflow_metrics["indexing_duration_seconds"] == pytest.approx(3.2)
    for name, value in METRICS.items():
        assert mlflow_metrics[name.replace("@", "_at_")] == pytest.approx(value)


def test_extended_metrics_default_to_none_keeps_only_the_six_metrics(tmp_path):
    json_path = _write_artifact(tmp_path)

    run_id = log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)

    mlflow_metrics = MlflowClient().get_run(run_id).data.metrics
    assert len(mlflow_metrics) == len(METRICS)


# ---------------------------------------------------------------------------
# Critère 5 (EXE-88) — tracking SQLite par défaut, sans variable d'environnement
# ---------------------------------------------------------------------------


def test_default_tracking_uri_is_sqlite_mlflow_db_at_repo_root():
    repo_root = Path(mlflow_tracking.__file__).resolve().parent.parent
    expected = f"sqlite:///{repo_root / 'mlflow.db'}"
    assert mlflow_tracking.DEFAULT_TRACKING_URI == expected


def test_no_mlflow_allow_file_store_env_var_is_set():
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


# ---------------------------------------------------------------------------
# EXE-89 — une trace MLflow par requête, rattachée au run
# ---------------------------------------------------------------------------


def _log_run(tmp_path) -> str:
    json_path = _write_artifact(tmp_path)
    return log_campaign_run("campagne-test", "run-1", CONFIG, METRICS, json_path)


# Critère 1 — exactement une trace par requête, retrouvable par l'identifiant du run.


def test_each_query_result_produces_one_trace_searchable_by_run_id(tmp_path):
    run_id = _log_run(tmp_path)
    results = [
        RetrievalResult(
            query_id=f"q{i}",
            query_text=f"claim {i}",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
        for i in range(3)
    ]
    qrels = {f"q{i}": {"d1"} for i in range(3)}

    log_query_traces(run_id, results, qrels, titles={"d1": "Titre D1"})

    traces = mlflow.search_traces(run_id=run_id, return_type="list")
    assert len(traces) == 3


# Critère 2 — l'entrée de la trace porte l'identifiant et le texte du claim.


def test_trace_input_carries_claim_id_and_text(tmp_path):
    run_id = _log_run(tmp_path)
    results = [
        RetrievalResult(
            query_id="q0",
            query_text="0-dimensional biomaterials show inductive properties.",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]
    qrels = {"q0": {"d1"}}

    log_query_traces(run_id, results, qrels, titles={"d1": "Titre D1"})

    trace = mlflow.search_traces(run_id=run_id, return_type="list")[0]
    assert trace.data.spans[0].inputs == {
        "query_id": "q0",
        "query_text": "0-dimensional biomaterials show inductive properties.",
    }


# Critère 3 — une étape de type retriever liste les 10 premiers documents
# (identifiant, titre, rang, score), même quand le classement en contient plus.


def test_trace_has_a_retriever_step_with_the_top_10_ranked_docs(tmp_path):
    run_id = _log_run(tmp_path)
    retrieved = [
        {"doc_id": f"d{i}", "rank": i, "score": round(1.0 - i * 0.01, 2)}
        for i in range(1, 16)
    ]
    results = [RetrievalResult(query_id="q0", query_text="claim", retrieved=retrieved)]
    qrels = {"q0": {"d1"}}
    titles = {f"d{i}": f"Titre {i}" for i in range(1, 16)}

    log_query_traces(run_id, results, qrels, titles)

    span = mlflow.search_traces(run_id=run_id, return_type="list")[0].data.spans[0]
    assert span.span_type == SpanType.RETRIEVER
    top10 = span.outputs["retrieved_top10"]
    assert len(top10) == 10
    assert top10[0] == {"doc_id": "d1", "title": "Titre 1", "rank": 1, "score": 0.99}
    assert [d["doc_id"] for d in top10] == [f"d{i}" for i in range(1, 11)]


# Critère 4 — docs attendus (identifiants + titres) et rang du premier dans le
# top 100, ou son absence quand aucun document attendu n'y figure.


def test_trace_carries_expected_docs_and_best_rank_when_found(tmp_path):
    run_id = _log_run(tmp_path)
    results = [
        RetrievalResult(
            query_id="q0",
            query_text="claim",
            retrieved=[
                {"doc_id": "d1", "rank": 1, "score": 0.9},
                {"doc_id": "d2", "rank": 2, "score": 0.8},
            ],
        )
    ]
    qrels = {"q0": {"d2"}}
    titles = {"d1": "T1", "d2": "T2"}

    log_query_traces(run_id, results, qrels, titles)

    span = mlflow.search_traces(run_id=run_id, return_type="list")[0].data.spans[0]
    assert span.get_attribute("expected_docs") == [{"doc_id": "d2", "title": "T2"}]
    assert span.get_attribute("best_rank_in_top_100") == 2


def test_trace_marks_absence_when_expected_doc_outside_top_100(tmp_path):
    run_id = _log_run(tmp_path)
    results = [
        RetrievalResult(
            query_id="48",
            query_text="claim 48",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.9}],
        )
    ]
    qrels = {"48": {"13734012"}}
    titles = {"d1": "T1", "13734012": "Titre attendu absent"}

    log_query_traces(run_id, results, qrels, titles)

    span = mlflow.search_traces(run_id=run_id, return_type="list")[0].data.spans[0]
    assert span.get_attribute("expected_docs") == [
        {"doc_id": "13734012", "title": "Titre attendu absent"}
    ]
    assert span.get_attribute("best_rank_in_top_100") is None


# ---------------------------------------------------------------------------
# EXE-95 critère 6 — un run hybride ajoute une étape par sous-retriever,
# avant l'étape de fusion (le span existant, classement final).
# ---------------------------------------------------------------------------


def test_hybrid_trace_has_one_retriever_step_per_sub_retriever(tmp_path):
    run_id = _log_run(tmp_path)
    fused = RetrievalResult(
        query_id="q0",
        query_text="claim",
        retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.9}],
    )
    dense = RetrievalResult(
        query_id="q0",
        query_text="claim",
        retrieved=[{"doc_id": "d2", "rank": 1, "score": 0.7}],
    )
    bm25 = RetrievalResult(
        query_id="q0",
        query_text="claim",
        retrieved=[{"doc_id": "d1", "rank": 1, "score": 5.0}],
    )
    qrels = {"q0": {"d1"}}
    titles = {"d1": "T1", "d2": "T2"}

    log_query_traces(
        run_id,
        [fused],
        qrels,
        titles,
        sub_rankings={"dense": [dense], "bm25": [bm25]},
    )

    spans = mlflow.search_traces(run_id=run_id, return_type="list")[0].data.spans
    child_names = {s.name for s in spans if s.name != "retrieve-q0"}
    assert child_names == {"retriever-dense-q0", "retriever-bm25-q0"}

    dense_span = next(s for s in spans if s.name == "retriever-dense-q0")
    bm25_span = next(s for s in spans if s.name == "retriever-bm25-q0")
    assert dense_span.span_type == SpanType.RETRIEVER
    assert dense_span.outputs["retrieved_top10"][0]["doc_id"] == "d2"
    assert bm25_span.outputs["retrieved_top10"][0]["doc_id"] == "d1"

    # L'étape de fusion (span parent, déjà existante) porte toujours le
    # classement final, inchangé par l'ajout des étapes par sous-retriever.
    fusion_span = next(s for s in spans if s.name == "retrieve-q0")
    assert fusion_span.outputs["retrieved_top10"][0]["doc_id"] == "d1"


def test_non_hybrid_trace_has_no_child_retriever_step(tmp_path):
    """Sans `sub_rankings` (run non hybride), le comportement EXE-89 est
    inchangé : un seul span par requête."""
    run_id = _log_run(tmp_path)
    results = [
        RetrievalResult(
            query_id="q0",
            query_text="claim",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.9}],
        )
    ]
    qrels = {"q0": {"d1"}}

    log_query_traces(run_id, results, qrels, titles={"d1": "T1"})

    spans = mlflow.search_traces(run_id=run_id, return_type="list")[0].data.spans
    assert len(spans) == 1
