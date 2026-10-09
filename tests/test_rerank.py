"""Tests du reranker cross-encoder (EXE-96 critères 2, 3, 6, 8).

Aucun modèle cross-encoder chargé : `score_fn` est toujours fabriqué.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

import rag_eval_scifact.rerank as rerank_module
from rag_eval_scifact.rerank import (
    essai_rerank_campaign,
    format_rerank_oom_message,
    is_rerank_out_of_memory,
    rerank_campaign_results,
    rerank_ranking,
)
from rag_eval_scifact.retrieve import RetrievalResult

# ---------------------------------------------------------------------------
# Critère 2 — chaque paire reclassée est (claim, titre + texte du document)
# ---------------------------------------------------------------------------


def test_pairs_combine_claim_text_and_document_text():
    captured: list[tuple[str, str]] = []

    def fake_score_fn(pairs):
        captured.extend(pairs)
        return [0.0] * len(pairs)

    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.5},
        {"doc_id": "d2", "rank": 2, "score": 0.4},
    ]
    doc_texts = {"d1": "Titre1 Texte1", "d2": "Titre2 Texte2"}

    rerank_ranking("mon claim", retrieved, doc_texts, top_n=10, score_fn=fake_score_fn)

    assert captured == [
        ("mon claim", "Titre1 Texte1"),
        ("mon claim", "Titre2 Texte2"),
    ]


def test_no_pairs_scored_when_retrieved_is_empty():
    calls = []

    def fake_score_fn(pairs):
        calls.append(pairs)
        return []

    reranked = rerank_ranking("claim", [], {}, top_n=10, score_fn=fake_score_fn)

    assert reranked == []
    assert calls == []


# ---------------------------------------------------------------------------
# Critère 3 — la liste reclassée contient exactement les mêmes documents
# ---------------------------------------------------------------------------


def test_reranked_list_contains_exactly_the_same_documents_reordered():
    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.9},
        {"doc_id": "d2", "rank": 2, "score": 0.8},
        {"doc_id": "d3", "rank": 3, "score": 0.7},
    ]
    doc_texts = {"d1": "t1", "d2": "t2", "d3": "t3"}

    def fake_score_fn(pairs):
        # Score fabriqué qui inverse l'ordre d'entrée.
        return list(range(len(pairs)))

    reranked = rerank_ranking(
        "claim", retrieved, doc_texts, top_n=3, score_fn=fake_score_fn
    )

    assert {d["doc_id"] for d in reranked} == {"d1", "d2", "d3"}
    assert [d["doc_id"] for d in reranked] == ["d3", "d2", "d1"]
    assert [d["rank"] for d in reranked] == [1, 2, 3]


def test_documents_beyond_top_n_are_appended_unchanged():
    retrieved = [
        {"doc_id": "d1", "rank": 1, "score": 0.9},
        {"doc_id": "d2", "rank": 2, "score": 0.8},
        {"doc_id": "d3", "rank": 3, "score": 0.1},
    ]
    doc_texts = {"d1": "t1", "d2": "t2", "d3": "t3"}

    def fake_score_fn(pairs):
        assert len(pairs) == 2  # seuls d1 et d2 entrent dans top_n=2
        return [0.0, 1.0]

    reranked = rerank_ranking(
        "claim", retrieved, doc_texts, top_n=2, score_fn=fake_score_fn
    )

    assert [d["doc_id"] for d in reranked] == ["d2", "d1", "d3"]
    assert [d["rank"] for d in reranked] == [1, 2, 3]
    assert reranked[2]["score"] == 0.1  # score original conservé pour la queue


# ---------------------------------------------------------------------------
# Critère 6 — la durée du reclassement est mesurée
# ---------------------------------------------------------------------------


def test_rerank_campaign_results_returns_reranked_results_and_duration(monkeypatch):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="claim1",
            retrieved=[
                {"doc_id": "d1", "rank": 1, "score": 0.5},
                {"doc_id": "d2", "rank": 2, "score": 0.4},
            ],
        ),
    ]
    doc_texts = {"d1": "t1", "d2": "t2"}

    def fake_score_fn(pairs):
        return [0.0, 1.0]  # inverse l'ordre

    clock = iter([10.0, 10.5])
    monkeypatch.setattr(rerank_module.time, "perf_counter", lambda: next(clock))

    reranked, duration = rerank_campaign_results(
        results, doc_texts, top_n=2, score_fn=fake_score_fn
    )

    assert [d["doc_id"] for d in reranked[0].retrieved] == ["d2", "d1"]
    assert duration == pytest.approx(0.5)


def test_rerank_campaign_results_reranks_every_query():
    results = [
        RetrievalResult(
            query_id=f"q{i}",
            query_text=f"claim{i}",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
        for i in range(3)
    ]
    calls: list[int] = []

    def fake_score_fn(pairs):
        calls.append(len(pairs))
        return [0.0] * len(pairs)

    reranked, _ = rerank_campaign_results(
        results, {"d1": "t1"}, top_n=100, score_fn=fake_score_fn
    )

    assert len(reranked) == 3
    assert calls == [1, 1, 1]


# ---------------------------------------------------------------------------
# EXE-157 critères 5, 6 — instruction et demi-précision du scorer par défaut
# (aucun modèle réel chargé : `CrossEncoder` est remplacé par un faux qui
# enregistre ce qu'il reçoit)
# ---------------------------------------------------------------------------


class _FakeCrossEncoder:
    instances: ClassVar[list[_FakeCrossEncoder]] = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.predict_calls: list[dict] = []
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, **kwargs):
        self.predict_calls.append({"pairs": list(pairs), **kwargs})
        return [0.0 for _ in pairs]


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    _FakeCrossEncoder.instances = []
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    return _FakeCrossEncoder


def test_default_scorer_passes_instruction_as_predict_prompt(fake_cross_encoder):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results,
        {"d1": "t1"},
        top_n=20,
        model_name="Qwen/Qwen3-Reranker-0.6B",
        instruction="Given a scientific claim, retrieve documents that support or refute it",
    )

    instance = fake_cross_encoder.instances[0]
    assert instance.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert instance.predict_calls[0]["prompt"] == (
        "Given a scientific claim, retrieve documents that support or refute it"
    )


def test_default_scorer_passes_no_prompt_when_instruction_is_empty(fake_cross_encoder):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results,
        {"d1": "t1"},
        top_n=20,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    instance = fake_cross_encoder.instances[0]
    assert "prompt" not in instance.predict_calls[0]


def test_default_scorer_loads_model_in_half_precision_when_asked(fake_cross_encoder):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results,
        {"d1": "t1"},
        top_n=20,
        model_name="Qwen/Qwen3-Reranker-4B",
        half_precision=True,
    )

    instance = fake_cross_encoder.instances[0]
    assert instance.init_kwargs.get("model_kwargs") == {"torch_dtype": "float16"}


def test_default_scorer_does_not_set_half_precision_by_default(fake_cross_encoder):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results, {"d1": "t1"}, top_n=20, model_name="BAAI/bge-reranker-v2-m3"
    )

    instance = fake_cross_encoder.instances[0]
    assert "model_kwargs" not in instance.init_kwargs


# ---------------------------------------------------------------------------
# EXE-159 critère 3 — taille de lot envoyée à `CrossEncoder.predict`
# ---------------------------------------------------------------------------


def test_default_scorer_passes_the_configured_batch_size_to_predict(
    fake_cross_encoder,
):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results,
        {"d1": "t1"},
        top_n=20,
        model_name="Qwen/Qwen3-Reranker-4B",
        batch_size=4,
    )

    instance = fake_cross_encoder.instances[0]
    assert instance.predict_calls[0]["batch_size"] == 4


def test_default_scorer_uses_32_as_the_default_batch_size(fake_cross_encoder):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]

    rerank_campaign_results(
        results,
        {"d1": "t1"},
        top_n=20,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    instance = fake_cross_encoder.instances[0]
    assert instance.predict_calls[0]["batch_size"] == 32


# ---------------------------------------------------------------------------
# EXE-157 critère 7 — essai du reranker : échantillon de requêtes, durée
# mesurée et extrapolée, rien n'est écrit
# ---------------------------------------------------------------------------


def test_essai_rerank_campaign_times_only_the_sample_and_extrapolates(monkeypatch):
    results = [
        RetrievalResult(
            query_id=f"q{i}",
            query_text=f"claim{i}",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
        for i in range(15)
    ]
    seen_sample_sizes: list[int] = []

    def fake_score_fn(pairs):
        seen_sample_sizes.append(len(pairs))
        return [0.0] * len(pairs)

    clock = iter([10.0, 11.0])
    monkeypatch.setattr(rerank_module.time, "perf_counter", lambda: next(clock))

    stats = essai_rerank_campaign(
        results,
        {"d1": "t1"},
        top_n=20,
        score_fn=fake_score_fn,
        sample_size=10,
        n_claims=300,
    )

    assert len(seen_sample_sizes) == 10  # seules les 10 premières affirmations
    assert stats["duration_seconds"] == pytest.approx(1.0)
    assert stats["extrapolated_minutes"] == pytest.approx(1.0 * (300 / 10) / 60)


# ---------------------------------------------------------------------------
# EXE-159 critère 2 — mémoire occupée sur l'accélérateur, rapportée en Go
# ---------------------------------------------------------------------------


def test_essai_rerank_campaign_reports_accelerator_memory_in_gb(monkeypatch):
    results = [
        RetrievalResult(
            query_id="q1",
            query_text="une affirmation",
            retrieved=[{"doc_id": "d1", "rank": 1, "score": 0.5}],
        )
    ]
    monkeypatch.setattr(rerank_module, "_accelerator_memory_gb", lambda: 8.5)

    stats = essai_rerank_campaign(
        results,
        {"d1": "t1"},
        top_n=20,
        score_fn=lambda pairs: [0.0] * len(pairs),
    )

    assert stats["memory_gb"] == 8.5


# ---------------------------------------------------------------------------
# EXE-159 critère 1 — la demi-précision charge réellement des poids float16,
# vérifié sur le modèle construit par le code de chargement (pas sur les
# arguments passés), avec un petit modèle de test, sans réseau
# ---------------------------------------------------------------------------


def _write_tiny_sequence_classification_model(path) -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        BertConfig,
        BertTokenizerFast,
    )

    config = BertConfig(
        vocab_size=99,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        num_labels=1,
    )
    AutoModelForSequenceClassification.from_config(config).save_pretrained(path)

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"tok{i}" for i in range(94)
    ]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(path)


def test_half_precision_loads_real_float16_weights_on_the_built_model(tmp_path):
    import torch

    _write_tiny_sequence_classification_model(tmp_path)

    model = rerank_module._load_cross_encoder(str(tmp_path), half_precision=True)

    assert next(model.model.parameters()).dtype is torch.float16


def test_without_half_precision_the_built_model_keeps_full_precision_weights(
    tmp_path,
):
    import torch

    _write_tiny_sequence_classification_model(tmp_path)

    model = rerank_module._load_cross_encoder(str(tmp_path), half_precision=False)

    assert next(model.model.parameters()).dtype is torch.float32


# ---------------------------------------------------------------------------
# EXE-159 critère 5 — dépassement mémoire du reranker : détection et message
# de remplacement de la trace Python, sans charger aucun modèle
# ---------------------------------------------------------------------------


def test_is_rerank_out_of_memory_true_for_the_mps_oom_runtime_error():
    exc = RuntimeError(
        "MPS backend out of memory (MPS allocated: 17.79 GiB, other "
        "allocations: 1.70 MiB, max allowed: 18.13 GiB). Tried to allocate "
        "528.00 MiB"
    )

    assert is_rerank_out_of_memory(exc) is True


def test_is_rerank_out_of_memory_false_for_an_unrelated_runtime_error():
    assert is_rerank_out_of_memory(RuntimeError("modèle introuvable")) is False


def test_format_rerank_oom_message_names_the_run_and_the_reported_memory():
    exc = RuntimeError(
        "MPS backend out of memory (MPS allocated: 17.79 GiB, other "
        "allocations: 1.70 MiB, max allowed: 18.13 GiB). Tried to allocate "
        "528.00 MiB"
    )

    message = format_rerank_oom_message("rerank-qwen3-4b-top20", exc)

    assert "rerank-qwen3-4b-top20" in message
    assert "528.00 MiB" in message
    assert "18.13 GiB" in message
    assert "rerank.batch_size" in message
