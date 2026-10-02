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
