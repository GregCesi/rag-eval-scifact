"""Tests de la grille v2-grid : fichier déclaratif + détection des runs déjà faits.

N'importe aucun module d'embedding : `load_grid_combos` lit un YAML via OmegaConf,
`has_existing_result` ne fait que lister des fichiers (EXE-91, critères 1 et 5).
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir

from rag_eval_scifact.grid import has_existing_result, load_grid_combos
from rag_eval_scifact.run_campaign import CONF_DIR

# ---------------------------------------------------------------------------
# Critère 1 — une seule combinaison déclarée, celle déjà réalisable
# ---------------------------------------------------------------------------


def test_v2_grid_declares_exactly_one_combo_dense_minilm_256_no_rerank():
    combos = load_grid_combos()

    assert len(combos) == 1
    combo = combos[0]
    assert combo["run_name"]

    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"run_name={combo['run_name']}", *combo["overrides"]],
        )

    assert cfg.retriever.name == "dense"
    assert cfg.retriever.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retriever.max_seq_length == 256
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
