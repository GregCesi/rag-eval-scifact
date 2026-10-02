"""Tests de la grille v2-grid : fichier déclaratif + détection des runs déjà faits.

N'importe aucun module d'embedding : `load_grid_combos` lit un YAML via OmegaConf,
`has_existing_result` ne fait que lister des fichiers (EXE-91, critères 1 et 5 ;
EXE-92, critère 8 — la combinaison dense/MiniLM/passages s'ajoute à côté de
celle du document entier ; EXE-93, critère 6 — les deux combinaisons BM25 —
sans toucher ce module).
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from rag_eval_scifact.grid import has_existing_result, load_grid_combos
from rag_eval_scifact.run_campaign import CONF_DIR


def _compose_combo(combo: dict):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(
            config_name="config",
            overrides=[f"run_name={combo['run_name']}", *combo["overrides"]],
        )


# ---------------------------------------------------------------------------
# Critère 1 — la combinaison document entier, déjà réalisable
# ---------------------------------------------------------------------------


def test_v2_grid_declares_the_document_combo_dense_minilm_256_no_rerank():
    combos = load_grid_combos()
    combo = next(c for c in combos if c["run_name"] == "dense-minilm-256-sans-reranker")

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.max_seq_length == 256
    assert cfg.retriever.unit == "document"
    assert cfg.rerank is None


# ---------------------------------------------------------------------------
# EXE-92 critère 8 — la combinaison dense/MiniLM/passages, sans reranker
# ---------------------------------------------------------------------------


def test_v2_grid_declares_the_passages_combo_dense_minilm_passages_no_rerank():
    combos = load_grid_combos()
    combo = next(
        c for c in combos if c["run_name"] == "dense-minilm-passages-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.unit == "passages"
    assert cfg.retriever.chunk_size == 128
    assert cfg.retriever.chunk_overlap == 32
    assert cfg.rerank is None


# ---------------------------------------------------------------------------
# EXE-93 critère 6 — les deux combinaisons BM25, sans reranker
# ---------------------------------------------------------------------------


def test_v2_grid_declares_the_bm25_document_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(c for c in combos if c["run_name"] == "bm25-document-sans-reranker")

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "bm25"
    assert cfg.retriever.unit == "document"
    assert cfg.rerank is None


def test_v2_grid_declares_the_bm25_passages_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(c for c in combos if c["run_name"] == "bm25-passages-sans-reranker")

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "bm25"
    assert cfg.retriever.unit == "passages"
    assert cfg.retriever.chunk_size == 128
    assert cfg.retriever.chunk_overlap == 32
    assert cfg.rerank is None


def test_v2_grid_declares_exactly_seven_combos():
    combos = load_grid_combos()

    assert len(combos) == 7
    assert {c["run_name"] for c in combos} == {
        "dense-minilm-256-sans-reranker",
        "dense-minilm-passages-sans-reranker",
        "bm25-document-sans-reranker",
        "bm25-passages-sans-reranker",
        "dense-qwen3-256-sans-reranker",
        "dense-qwen3-abstract-entier-sans-reranker",
        "dense-qwen3-passages-sans-reranker",
    }


# ---------------------------------------------------------------------------
# EXE-94 critère 8 — les trois combinaisons Qwen3-Embedding, sans reranker
# ---------------------------------------------------------------------------


def test_v2_grid_declares_the_qwen3_256_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(c for c in combos if c["run_name"] == "dense-qwen3-256-sans-reranker")

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.retriever.max_seq_length == 256
    assert cfg.retriever.unit == "document"
    assert cfg.retriever.query_instruction != ""
    assert cfg.rerank is None


def test_v2_grid_declares_the_qwen3_abstract_entier_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(
        c
        for c in combos
        if c["run_name"] == "dense-qwen3-abstract-entier-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.retriever.max_seq_length == 2048
    assert cfg.retriever.unit == "document"
    assert cfg.retriever.query_instruction != ""
    assert cfg.rerank is None


def test_v2_grid_declares_the_qwen3_passages_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(
        c for c in combos if c["run_name"] == "dense-qwen3-passages-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.retriever.unit == "passages"
    assert cfg.retriever.chunk_size == 128
    assert cfg.retriever.chunk_overlap == 32
    assert cfg.retriever.query_instruction != ""
    assert cfg.rerank is None


# ---------------------------------------------------------------------------
# Critère 5 — détection d'un résultat déjà présent pour un run_name
# ---------------------------------------------------------------------------


def test_has_existing_result_true_when_a_matching_file_is_present(tmp_path):
    campaign_dir = tmp_path / "v2-grid"
    campaign_dir.mkdir()
    (
        campaign_dir
        / "dense-minilm-256-sans-reranker-2026-10-02T00-00-00-000000.json.gz"
    ).touch()

    assert has_existing_result(campaign_dir, "dense-minilm-256-sans-reranker")
    assert not has_existing_result(campaign_dir, "une-autre-combinaison")


def test_has_existing_result_false_when_campaign_dir_is_absent(tmp_path):
    assert not has_existing_result(
        tmp_path / "absent", "dense-minilm-256-sans-reranker"
    )
