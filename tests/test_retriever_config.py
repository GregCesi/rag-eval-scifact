"""Tests des valeurs de configuration du retriever (EXE-92 critère 1 ; EXE-93 critère 1).

Compose `conf/config.yaml` via Hydra, sans rien lancer ni charger de modèle.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from rag_eval_scifact.run_campaign import CONF_DIR


def test_defaults_reproduce_v1_document_unit():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.retriever.unit == "document"
    assert cfg.retriever.chunk_size == 128
    assert cfg.retriever.chunk_overlap == 32
    assert cfg.retriever.grouping == "max"
    assert cfg.retriever.grouping_top_n == 1000


def test_unit_and_chunking_are_overridable_from_the_command_line():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "retriever.unit=passages",
                "retriever.chunk_size=64",
                "retriever.chunk_overlap=16",
                "retriever.grouping=sum",
                "retriever.grouping_top_n=500",
            ],
        )

    assert cfg.retriever.unit == "passages"
    assert cfg.retriever.chunk_size == 64
    assert cfg.retriever.chunk_overlap == 16
    assert cfg.retriever.grouping == "sum"
    assert cfg.retriever.grouping_top_n == 500


# ---------------------------------------------------------------------------
# EXE-93 critère 1 — BM25 sélectionnable, k1 et b configurables
# ---------------------------------------------------------------------------


def test_bm25_defaults_are_elasticsearch_defaults():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.bm25_k1 == 1.2
    assert cfg.retriever.bm25_b == 0.75


def test_bm25_is_selectable_and_its_parameters_are_overridable():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "retriever.name=bm25",
                "retriever.bm25_k1=2.0",
                "retriever.bm25_b=0.5",
            ],
        )

    assert cfg.retriever.name == "bm25"
    assert cfg.retriever.bm25_k1 == 2.0
    assert cfg.retriever.bm25_b == 0.5


# ---------------------------------------------------------------------------
# EXE-94 critère 1, 2 — Qwen3-Embedding sélectionnable, MiniLM reste le défaut
# ---------------------------------------------------------------------------


def test_defaults_select_minilm_with_no_query_instruction():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.retriever.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.max_seq_length == 256
    assert cfg.retriever.query_instruction == ""
    assert cfg.retriever.batch_size == 64


def test_batch_size_is_overridable():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["retriever.batch_size=8"])

    assert cfg.retriever.batch_size == 8


# ---------------------------------------------------------------------------
# EXE-95 critère 1 — hybride sélectionnable, méthode et k RRF configurables
# ---------------------------------------------------------------------------


def test_hybrid_defaults_select_union_with_k_60():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.retriever.fusion_mode == "union"
    assert cfg.retriever.rrf_k == 60


def test_hybrid_is_selectable_and_its_fusion_mode_is_overridable():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "retriever.name=hybrid",
                "retriever.fusion_mode=rrf",
                "retriever.rrf_k=30",
            ],
        )

    assert cfg.retriever.name == "hybrid"
    assert cfg.retriever.fusion_mode == "rrf"
    assert cfg.retriever.rrf_k == 30


def test_qwen_model_and_query_instruction_are_overridable():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "retriever.model=Qwen/Qwen3-Embedding-0.6B",
                "retriever.max_seq_length=2048",
                "retriever.query_instruction=Instruct: retrieve relevant documents. Query: ",
            ],
        )

    assert cfg.retriever.model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.retriever.max_seq_length == 2048
    assert (
        cfg.retriever.query_instruction
        == "Instruct: retrieve relevant documents. Query: "
    )


# ---------------------------------------------------------------------------
# EXE-96 critère 1 — reranker sélectionnable, modèle et top_n configurables
# ---------------------------------------------------------------------------


def test_rerank_defaults_to_none_with_cross_encoder_values_ready():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.rerank.name == "none"
    assert cfg.rerank.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.rerank.top_n == 100


def test_rerank_is_selectable_and_its_parameters_are_overridable():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "rerank.name=cross-encoder",
                "rerank.model=other-cross-encoder",
                "rerank.top_n=50",
            ],
        )

    assert cfg.rerank.name == "cross-encoder"
    assert cfg.rerank.model == "other-cross-encoder"
    assert cfg.rerank.top_n == 50


def test_rerank_null_override_still_disables_it():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["rerank=null"])

    assert cfg.rerank is None
