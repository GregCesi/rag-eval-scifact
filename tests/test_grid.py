"""Tests de la grille v2-grid : fichier déclaratif + détection des runs déjà faits.

N'importe aucun module d'embedding : `load_grid_combos` lit un YAML via OmegaConf,
`has_existing_result` ne fait que lister des fichiers (EXE-91, critères 1 et 5 ;
EXE-92, critère 8 — la combinaison dense/MiniLM/passages s'ajoute à côté de
celle du document entier ; EXE-93, critère 6 — les deux combinaisons BM25 —
sans toucher ce module).
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

from rag_eval_scifact.grid import (
    GRID_DIR,
    GridFileMissing,
    grid_path_for_campagne,
    has_existing_result,
    load_grid_combos,
)
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


def test_v2_grid_declares_exactly_seventeen_combos_without_reranker():
    combos = load_grid_combos()

    sans_reranker = {
        c["run_name"] for c in combos if c["run_name"].endswith("sans-reranker")
    }
    assert len(sans_reranker) == 17
    assert sans_reranker == {
        "dense-minilm-256-sans-reranker",
        "dense-minilm-passages-sans-reranker",
        "bm25-document-sans-reranker",
        "bm25-passages-sans-reranker",
        "dense-qwen3-256-sans-reranker",
        "dense-qwen3-abstract-entier-sans-reranker",
        "dense-qwen3-passages-sans-reranker",
        "hybrid-minilm-256-union-sans-reranker",
        "hybrid-minilm-256-rrf-sans-reranker",
        "hybrid-minilm-passages-union-sans-reranker",
        "hybrid-minilm-passages-rrf-sans-reranker",
        "hybrid-qwen3-256-union-sans-reranker",
        "hybrid-qwen3-256-rrf-sans-reranker",
        "hybrid-qwen3-abstract-entier-union-sans-reranker",
        "hybrid-qwen3-abstract-entier-rrf-sans-reranker",
        "hybrid-qwen3-passages-union-sans-reranker",
        "hybrid-qwen3-passages-rrf-sans-reranker",
    }


# ---------------------------------------------------------------------------
# EXE-96 critère 7 — chaque combinaison obtient sa jumelle avec reranker :
# 34 combinaisons déclarées, toutes distinctes.
# ---------------------------------------------------------------------------


def test_v2_grid_declares_exactly_thirty_four_combos_all_distinct():
    combos = load_grid_combos()
    run_names = [c["run_name"] for c in combos]

    assert len(run_names) == 34
    assert len(set(run_names)) == 34  # toutes distinctes

    sans_reranker = {n for n in run_names if n.endswith("sans-reranker")}
    avec_reranker = {n for n in run_names if n.endswith("avec-reranker")}
    assert len(sans_reranker) == 17
    assert len(avec_reranker) == 17
    # Chaque combinaison "sans-reranker" a exactement sa jumelle "avec-reranker".
    assert {n[: -len("sans-reranker")] for n in sans_reranker} == {
        n[: -len("avec-reranker")] for n in avec_reranker
    }


def test_v2_grid_reranker_twin_overrides_only_the_reranker_setting():
    combos = load_grid_combos()
    sans = next(c for c in combos if c["run_name"] == "dense-minilm-256-sans-reranker")
    avec = next(c for c in combos if c["run_name"] == "dense-minilm-256-avec-reranker")

    cfg_sans = _compose_combo(sans)
    cfg_avec = _compose_combo(avec)

    assert cfg_sans.rerank is None
    assert cfg_avec.rerank.name == "cross-encoder"
    # Le reste de la config (retriever) est identique entre les deux jumelles.
    assert cfg_sans.retriever == cfg_avec.retriever


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
# EXE-95 critère 7 — les 10 combinaisons hybrides, sans reranker
# ---------------------------------------------------------------------------


def test_v2_grid_declares_the_hybrid_minilm_256_union_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(
        c for c in combos if c["run_name"] == "hybrid-minilm-256-union-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "hybrid"
    assert cfg.retriever.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.unit == "document"
    assert cfg.retriever.fusion_mode == "union"
    assert cfg.rerank is None


def test_v2_grid_declares_the_hybrid_minilm_256_rrf_combo_no_rerank():
    combos = load_grid_combos()
    combo = next(
        c for c in combos if c["run_name"] == "hybrid-minilm-256-rrf-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.retriever.name == "hybrid"
    assert cfg.retriever.fusion_mode == "rrf"
    assert cfg.retriever.rrf_k == 60
    assert cfg.rerank is None


def test_v2_grid_declares_the_hybrid_minilm_passages_combos_no_rerank():
    combos = load_grid_combos()
    for run_name, mode in [
        ("hybrid-minilm-passages-union-sans-reranker", "union"),
        ("hybrid-minilm-passages-rrf-sans-reranker", "rrf"),
    ]:
        combo = next(c for c in combos if c["run_name"] == run_name)
        cfg = _compose_combo(combo)

        assert cfg.retriever.name == "hybrid"
        assert cfg.retriever.unit == "passages"
        assert cfg.retriever.chunk_size == 128
        assert cfg.retriever.chunk_overlap == 32
        assert cfg.retriever.fusion_mode == mode
        assert cfg.rerank is None


def test_v2_grid_declares_the_hybrid_qwen3_combos_no_rerank():
    combos = load_grid_combos()
    for run_name, unit, max_seq_length, mode in [
        ("hybrid-qwen3-256-union-sans-reranker", "document", 256, "union"),
        ("hybrid-qwen3-256-rrf-sans-reranker", "document", 256, "rrf"),
        (
            "hybrid-qwen3-abstract-entier-union-sans-reranker",
            "document",
            2048,
            "union",
        ),
        ("hybrid-qwen3-abstract-entier-rrf-sans-reranker", "document", 2048, "rrf"),
        ("hybrid-qwen3-passages-union-sans-reranker", "passages", 2048, "union"),
        ("hybrid-qwen3-passages-rrf-sans-reranker", "passages", 2048, "rrf"),
    ]:
        combo = next(c for c in combos if c["run_name"] == run_name)
        cfg = _compose_combo(combo)

        assert cfg.retriever.name == "hybrid"
        assert cfg.retriever.model == "Qwen/Qwen3-Embedding-0.6B"
        assert cfg.retriever.unit == unit
        assert cfg.retriever.max_seq_length == max_seq_length
        assert cfg.retriever.query_instruction != ""
        assert cfg.retriever.fusion_mode == mode
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


# ---------------------------------------------------------------------------
# EXE-140 — une grille par campagne, jamais seulement v2-grid
# ---------------------------------------------------------------------------


def test_grid_path_for_campagne_points_at_conf_grid_campagne_yaml():
    assert grid_path_for_campagne("v4-leviers") == GRID_DIR / "v4-leviers.yaml"


def test_load_grid_combos_defaults_to_v2_grid_without_a_campagne():
    combos = load_grid_combos()
    assert len(combos) == 34


def test_load_grid_combos_reads_the_named_campagne_grid_file():
    combos = load_grid_combos(campagne="v4-leviers")
    run_names = {c["run_name"] for c in combos}
    assert run_names == {
        "qwen3-passages-reference",
        "qwen3-passages-sans-instruction",
        "qwen3-4b-passages",
        "medcpt-passages",
        "qwen3-passages-hyde",
    }


def test_load_grid_combos_raises_grid_file_missing_without_a_traceback_cause(tmp_path):
    with pytest.raises(GridFileMissing) as exc_info:
        load_grid_combos(campagne="campagne-sans-grille")

    assert "campagne-sans-grille" in str(exc_info.value)


def test_load_grid_combos_with_explicit_path_still_works(tmp_path):
    grid_file = tmp_path / "ad-hoc.yaml"
    grid_file.write_text(
        "runs:\n  - run_name: x\n    overrides: []\n", encoding="utf-8"
    )

    combos = load_grid_combos(path=grid_file)

    assert combos == [{"run_name": "x", "overrides": []}]


# ---------------------------------------------------------------------------
# EXE-140 critères 6, 7, 8 — v4-leviers : référence et ses deux premières
# variantes à un seul levier changé
# ---------------------------------------------------------------------------


def test_v4_leviers_reference_matches_v2_grid_dense_qwen3_passages_key_for_key():
    v2_combos = load_grid_combos()
    v4_combos = load_grid_combos(campagne="v4-leviers")

    v2_reference = next(
        c for c in v2_combos if c["run_name"] == "dense-qwen3-passages-sans-reranker"
    )
    v4_reference = next(
        c for c in v4_combos if c["run_name"] == "qwen3-passages-reference"
    )

    cfg_v2 = _compose_combo(v2_reference)
    cfg_v4 = _compose_combo(v4_reference)

    assert cfg_v2.retriever == cfg_v4.retriever
    assert cfg_v2.rerank == cfg_v4.rerank


def test_v4_leviers_sans_instruction_only_empties_the_query_instruction():
    v4_combos = load_grid_combos(campagne="v4-leviers")
    reference = next(
        c for c in v4_combos if c["run_name"] == "qwen3-passages-reference"
    )
    sans_instruction = next(
        c for c in v4_combos if c["run_name"] == "qwen3-passages-sans-instruction"
    )

    cfg_reference = _compose_combo(reference)
    cfg_sans_instruction = _compose_combo(sans_instruction)

    assert cfg_sans_instruction.retriever.query_instruction == ""
    assert cfg_reference.retriever.query_instruction != ""

    diff = {
        k: v
        for k, v in cfg_sans_instruction.retriever.items()
        if cfg_reference.retriever[k] != v
    }
    assert set(diff) == {"query_instruction"}


def test_v4_leviers_4b_only_changes_model_and_batch_size():
    v4_combos = load_grid_combos(campagne="v4-leviers")
    reference = next(
        c for c in v4_combos if c["run_name"] == "qwen3-passages-reference"
    )
    model_4b = next(c for c in v4_combos if c["run_name"] == "qwen3-4b-passages")

    cfg_reference = _compose_combo(reference)
    cfg_4b = _compose_combo(model_4b)

    assert cfg_4b.retriever.model == "Qwen/Qwen3-Embedding-4B"
    assert cfg_4b.retriever.batch_size == 2

    diff = {
        k: v for k, v in cfg_4b.retriever.items() if cfg_reference.retriever[k] != v
    }
    assert set(diff) == {"model", "batch_size"}


# ---------------------------------------------------------------------------
# EXE-140 critères 9, 10 — medcpt-passages : double encodeur, sans instruction
# ---------------------------------------------------------------------------


def test_v4_leviers_medcpt_uses_two_named_encoders_without_query_instruction():
    v4_combos = load_grid_combos(campagne="v4-leviers")
    combo = next(c for c in v4_combos if c["run_name"] == "medcpt-passages")

    cfg = _compose_combo(combo)

    assert cfg.retriever.model == "ncbi/MedCPT-Article-Encoder"
    assert cfg.retriever.query_model == "ncbi/MedCPT-Query-Encoder"
    assert cfg.retriever.query_instruction == ""
    assert cfg.retriever.unit == "passages"
    assert cfg.retriever.chunk_size == 128
    assert cfg.retriever.chunk_overlap == 32
    assert cfg.rerank is None


def test_v4_leviers_grid_has_no_reranker_bm25_or_hybrid():
    v4_combos = load_grid_combos(campagne="v4-leviers")

    for combo in v4_combos:
        cfg = _compose_combo(combo)
        assert cfg.rerank is None
        assert cfg.retriever.name == "dense"


# ---------------------------------------------------------------------------
# EXE-157 critères 1, 2, 3, 5, 6 — v5-rerankers : six combinaisons, même
# recherche que la référence v4-leviers, seul le reranker change
# ---------------------------------------------------------------------------

V5_RERANKER_RUN_NAMES = {
    "qwen3-passages-sans-reranker",
    "rerank-minilm-top20",
    "rerank-bge-m3-top20",
    "rerank-medcpt-top20",
    "rerank-qwen3-0.6b-top20",
    "rerank-qwen3-4b-top20",
}


def test_load_grid_combos_reads_v5_rerankers_six_named_runs():
    combos = load_grid_combos(campagne="v5-rerankers")
    run_names = {c["run_name"] for c in combos}
    assert run_names == V5_RERANKER_RUN_NAMES


def test_v5_rerankers_runs_match_v4_leviers_reference_retriever_key_for_key():
    v4_combos = load_grid_combos(campagne="v4-leviers")
    v5_combos = load_grid_combos(campagne="v5-rerankers")

    reference = next(
        c for c in v4_combos if c["run_name"] == "qwen3-passages-reference"
    )
    cfg_reference = _compose_combo(reference)

    for combo in v5_combos:
        cfg = _compose_combo(combo)
        assert cfg.retriever == cfg_reference.retriever, combo["run_name"]


def test_v5_rerankers_sans_reranker_run_has_no_reranker():
    v5_combos = load_grid_combos(campagne="v5-rerankers")
    combo = next(
        c for c in v5_combos if c["run_name"] == "qwen3-passages-sans-reranker"
    )

    cfg = _compose_combo(combo)

    assert cfg.rerank is None


def test_v5_rerankers_reranker_runs_reread_top_20_with_their_named_model():
    v5_combos = load_grid_combos(campagne="v5-rerankers")
    expected_models = {
        "rerank-minilm-top20": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank-bge-m3-top20": "BAAI/bge-reranker-v2-m3",
        "rerank-medcpt-top20": "ncbi/MedCPT-Cross-Encoder",
        "rerank-qwen3-0.6b-top20": "Qwen/Qwen3-Reranker-0.6B",
        "rerank-qwen3-4b-top20": "Qwen/Qwen3-Reranker-4B",
    }

    for run_name, model in expected_models.items():
        combo = next(c for c in v5_combos if c["run_name"] == run_name)
        cfg = _compose_combo(combo)

        assert cfg.rerank.name == "cross-encoder"
        assert cfg.rerank.model == model
        assert cfg.rerank.top_n == 20


def test_v5_rerankers_only_the_two_qwen3_rerankers_get_the_instruction():
    v5_combos = load_grid_combos(campagne="v5-rerankers")
    instructed = {"rerank-qwen3-0.6b-top20", "rerank-qwen3-4b-top20"}

    for combo in v5_combos:
        if combo["run_name"] == "qwen3-passages-sans-reranker":
            continue
        cfg = _compose_combo(combo)
        if combo["run_name"] in instructed:
            assert cfg.rerank.instruction == (
                "Given a scientific claim, retrieve documents that support or refute it"
            )
        else:
            assert cfg.rerank.instruction == ""


def test_v5_rerankers_only_the_4b_run_uses_half_precision():
    v5_combos = load_grid_combos(campagne="v5-rerankers")

    for combo in v5_combos:
        if combo["run_name"] == "qwen3-passages-sans-reranker":
            continue
        cfg = _compose_combo(combo)
        if combo["run_name"] == "rerank-qwen3-4b-top20":
            assert cfg.rerank.half_precision is True
        else:
            assert cfg.rerank.half_precision is False


# ---------------------------------------------------------------------------
# EXE-157 critère 9 — v2-grid garde sa configuration de reranker inchangée
# ---------------------------------------------------------------------------


def test_v2_grid_reranker_runs_keep_their_original_top_n_and_no_instruction():
    combos = load_grid_combos()
    combo = next(c for c in combos if c["run_name"] == "dense-minilm-256-avec-reranker")

    cfg = _compose_combo(combo)

    assert cfg.rerank.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.rerank.top_n == 100
    assert cfg.rerank.instruction == ""
    assert cfg.rerank.half_precision is False
