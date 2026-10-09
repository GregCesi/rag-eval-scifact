"""Tests du lanceur Hydra standard de `run_campaign` (EXE-88, critères 2 et 3).

`--help` est un vrai sous-processus : Hydra affiche l'aide et quitte sans
jamais appeler `main()`, donc sans charger de modèle. Le `--multirun` est
testé en process, avec `retrieve_campaign` et `get_token_counts` remplacés par
des fonctions fabriquées : aucun modèle n'est chargé, rien n'est écrit dans le
`mlflow.db` ni le `RESULTS.md` réels.
"""

from __future__ import annotations

import subprocess
import sys

import mlflow
import pytest
from hydra.core.global_hydra import GlobalHydra
from mlflow.tracking import MlflowClient

import rag_eval_scifact.run_campaign as run_campaign_module
from rag_eval_scifact import campaign
from rag_eval_scifact.retrieve import RetrievalResult


def test_help_shows_hydra_help_with_overridable_config():
    result = subprocess.run(
        [sys.executable, "-m", "rag_eval_scifact.run_campaign", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "powered by Hydra" in result.stdout
    assert "top_k: 100" in result.stdout
    assert "cache_dir:" in result.stdout


@pytest.fixture
def _isolated_cli(tmp_path, monkeypatch):
    """Isole RESULTS_DIR/RESULTS_MD, V1_BUCKETS_PATH et le tracking MLflow dans tmp_path."""
    monkeypatch.setattr(campaign, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(campaign, "RESULTS_MD", tmp_path / "RESULTS.md")
    buckets_path = tmp_path / "v1-buckets.json"
    buckets_path.write_text('{"counts": {}, "queries": {}}', encoding="utf-8")
    monkeypatch.setattr(campaign, "V1_BUCKETS_PATH", buckets_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def _fake_retrieve_campaign(**kwargs):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une question",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.9}],
        )
    ]
    qrels = {"q1": {"d1"}}
    stats = {
        "truncated_pct": 0.0,
        "avg_retrieval_latency_ms": 0.0,
        "indexing_duration_seconds": 0.0,
        "device": "cpu",
    }
    return results, qrels, "sha256:fake", stats


def test_multirun_produces_two_runs_two_files_two_mlflow_runs(
    _isolated_cli, monkeypatch
):
    calls: list[int] = []

    def tracking_retrieve_campaign(*, top_k, **kwargs):
        calls.append(top_k)
        return _fake_retrieve_campaign(top_k=top_k, **kwargs)

    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", tracking_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "--multirun",
            "campagne=dev",
            "top_k=10,20",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert sorted(calls) == [10, 20]

    files = sorted((_isolated_cli / "results" / "dev").glob("*.json.gz"))
    assert len(files) == 2

    client = MlflowClient()
    experiment = client.get_experiment_by_name("dev")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2


# ---------------------------------------------------------------------------
# EXE-93 — retriever.name=bm25 se propage jusqu'à retrieve_campaign
# ---------------------------------------------------------------------------


def test_retriever_name_bm25_and_its_params_reach_retrieve_campaign(
    _isolated_cli, monkeypatch
):
    calls: list[dict] = []

    def tracking_retrieve_campaign(**kwargs):
        calls.append(kwargs)
        return _fake_retrieve_campaign(**kwargs)

    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", tracking_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "retriever.name=bm25",
            "retriever.bm25_k1=2.0",
            "retriever.bm25_b=0.5",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert calls[0]["retriever_name"] == "bm25"
    assert calls[0]["bm25_k1"] == 2.0
    assert calls[0]["bm25_b"] == 0.5


# ---------------------------------------------------------------------------
# Critère 6 (EXE-89) — tracing=false désactive les traces, métriques inchangées
# ---------------------------------------------------------------------------


def test_tracing_false_writes_no_trace_and_keeps_metrics_identical(
    _isolated_cli, monkeypatch
):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "Titre D1", "text": ""}],
    )

    for tracing_value, run_name in [
        ("true", "with-tracing"),
        ("false", "without-tracing"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_campaign.py",
                "campagne=dev",
                f"run_name={run_name}",
                f"tracing={tracing_value}",
            ],
        )
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        run_campaign_module._cli()

    client = MlflowClient()
    experiment = client.get_experiment_by_name("dev")
    runs = {r.info.run_name: r for r in client.search_runs([experiment.experiment_id])}

    traces_with = mlflow.search_traces(
        run_id=runs["with-tracing"].info.run_id, return_type="list"
    )
    traces_without = mlflow.search_traces(
        run_id=runs["without-tracing"].info.run_id, return_type="list"
    )
    assert len(traces_with) == 1
    assert len(traces_without) == 0
    assert runs["with-tracing"].data.metrics == runs["without-tracing"].data.metrics


# ---------------------------------------------------------------------------
# EXE-96 — reranker : config, cache du premier étage, trace, durée
# ---------------------------------------------------------------------------


def test_rerank_not_invoked_when_disabled_by_default(_isolated_cli, monkeypatch):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    calls: list[int] = []
    monkeypatch.setattr(
        run_campaign_module,
        "rerank_campaign_results",
        lambda *a, **kw: calls.append(1),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_campaign.py", "campagne=dev", "tracing=false"]
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert calls == []


def test_rerank_setting_does_not_change_the_retrieve_campaign_call(
    _isolated_cli, monkeypatch
):
    """Critère 4 : le premier étage ne dépend jamais du réglage du reranker —
    `retrieve_campaign` reçoit exactement les mêmes arguments, reranker actif
    ou non, donc sert la même entrée de cache."""
    calls: list[dict] = []

    def tracking_retrieve_campaign(**kwargs):
        calls.append(kwargs)
        return _fake_retrieve_campaign(**kwargs)

    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", tracking_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "T", "text": "x"}],
    )
    monkeypatch.setattr(
        run_campaign_module,
        "rerank_campaign_results",
        lambda results, doc_texts, **kw: (results, 0.01),
    )

    for rerank_value, run_name in [
        ("none", "no-rerank"),
        ("cross-encoder", "with-rerank"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_campaign.py",
                "campagne=dev",
                f"run_name={run_name}",
                f"rerank.name={rerank_value}",
                "tracing=false",
            ],
        )
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        run_campaign_module._cli()

    assert calls[0] == calls[1]


def test_rerank_pairs_use_title_plus_text_at_document_level(_isolated_cli, monkeypatch):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "Titre D1", "text": "Texte D1"}],
    )
    seen_doc_texts: dict = {}

    def _capture_rerank(results, doc_texts, **kwargs):
        seen_doc_texts.update(doc_texts)
        return results, 0.1

    monkeypatch.setattr(run_campaign_module, "rerank_campaign_results", _capture_rerank)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "rerank.name=cross-encoder",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert seen_doc_texts == {"d1": "Titre D1 Texte D1"}


def test_rerank_cross_encoder_duration_is_logged_in_mlflow(_isolated_cli, monkeypatch):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "T1", "text": ""}],
    )
    monkeypatch.setattr(
        run_campaign_module,
        "rerank_campaign_results",
        lambda results, doc_texts, **kwargs: (results, 2.5),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "rerank.name=cross-encoder",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    client = MlflowClient()
    experiment = client.get_experiment_by_name("dev")
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.data.metrics["rerank_duration_seconds"] == pytest.approx(2.5)


def test_rerank_instruction_and_half_precision_are_read_from_config(
    _isolated_cli, monkeypatch
):
    """EXE-157 critères 5, 6 : `rerank.instruction` et `rerank.half_precision`
    de la config Hydra atteignent `rerank_campaign_results`."""
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "T1", "text": ""}],
    )
    captured: dict = {}

    def _capture_rerank(results, doc_texts, **kwargs):
        captured.update(kwargs)
        return results, 0.1

    monkeypatch.setattr(run_campaign_module, "rerank_campaign_results", _capture_rerank)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "rerank.name=cross-encoder",
            "rerank.instruction='Given a scientific claim, retrieve documents that support or refute it'",
            "rerank.half_precision=true",
            "rerank.batch_size=4",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert captured["instruction"] == (
        "Given a scientific claim, retrieve documents that support or refute it"
    )
    assert captured["half_precision"] is True
    assert captured["batch_size"] == 4


# ---------------------------------------------------------------------------
# EXE-159 critère 5 — dépassement mémoire du reranker : une phrase, jamais
# une trace Python, sur le chemin d'un run complet
# ---------------------------------------------------------------------------


def test_rerank_out_of_memory_stops_with_a_sentence_not_a_traceback(
    _isolated_cli, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_campaign_module, "retrieve_campaign", _fake_retrieve_campaign
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        run_campaign_module,
        "load_corpus",
        lambda path: [{"_id": "d1", "title": "T1", "text": ""}],
    )

    def _raise_oom(results, doc_texts, **kwargs):
        raise RuntimeError(
            "MPS backend out of memory (MPS allocated: 17.79 GiB, other "
            "allocations: 1.70 MiB, max allowed: 18.13 GiB). Tried to "
            "allocate 528.00 MiB"
        )

    monkeypatch.setattr(run_campaign_module, "rerank_campaign_results", _raise_oom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "run_name=rerank-qwen3-4b-top20",
            "rerank.name=cross-encoder",
            "rerank.model=Qwen/Qwen3-Reranker-4B",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with pytest.raises(SystemExit):
        run_campaign_module._cli()

    out = capsys.readouterr().out
    assert "rerank-qwen3-4b-top20" in out
    assert "528.00 MiB" in out
    assert "18.13 GiB" in out
    assert "rerank.batch_size" in out


# ---------------------------------------------------------------------------
# EXE-141, critère 10 — refus de lancer un run « query_source=hyde » tant que
# hyde.json est absent ou incomplet
# ---------------------------------------------------------------------------


def test_hyde_query_source_refuses_to_launch_when_hyde_file_is_missing(
    _isolated_cli, monkeypatch
):
    monkeypatch.setattr(run_campaign_module, "load_hyde_texts", lambda path: {})
    calls: list[dict] = []
    monkeypatch.setattr(
        run_campaign_module,
        "retrieve_campaign",
        lambda **kw: calls.append(kw) or _fake_retrieve_campaign(**kw),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "retriever.query_source=hyde",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with pytest.raises(SystemExit):
        run_campaign_module._cli()

    assert calls == []
    assert list((_isolated_cli / "results" / "dev").glob("*.json.gz")) == []


def test_hyde_query_source_refuses_to_launch_when_hyde_file_is_incomplete(
    capsys, _isolated_cli, monkeypatch
):
    monkeypatch.setattr(
        run_campaign_module, "load_hyde_texts", lambda path: {"1": "texte"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "retriever.query_source=hyde",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with pytest.raises(SystemExit):
        run_campaign_module._cli()

    out = capsys.readouterr().out
    assert "1" in out
    assert "300" in out
    assert list((_isolated_cli / "results" / "dev").glob("*.json.gz")) == []


def test_hyde_query_source_launches_when_hyde_file_is_complete(
    _isolated_cli, monkeypatch
):
    hyde_texts = {str(i): f"texte-{i}" for i in range(1, 301)}
    monkeypatch.setattr(run_campaign_module, "load_hyde_texts", lambda path: hyde_texts)
    calls: list[dict] = []
    monkeypatch.setattr(
        run_campaign_module,
        "retrieve_campaign",
        lambda **kw: calls.append(kw) or _fake_retrieve_campaign(**kw),
    )
    monkeypatch.setattr(run_campaign_module, "get_token_counts", lambda ids: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "campagne=dev",
            "retriever.query_source=hyde",
            "tracing=false",
        ],
    )

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    run_campaign_module._cli()

    assert calls[0]["query_source"] == "hyde"
    assert calls[0]["query_texts_override"] == hyde_texts
