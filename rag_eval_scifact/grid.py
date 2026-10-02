"""Grille de campagne v2-grid : combinaisons déclarées, une par entrée.

Un fichier déclaratif (`conf/grid/v2-grid.yaml`) plutôt qu'un sweep Hydra : la
grille est une liste de stratégies choisies à la main, pas un produit
cartésien. Chaque entrée ajoute des surcharges Hydra (`retriever.model=...`)
au-dessus de `conf/config.yaml`. Les fiches suivantes (EXE-92 à EXE-96)
étendent ce fichier au fur et à mesure que leur code existe, jamais ce module.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

GRID_PATH = Path("conf/grid/v2-grid.yaml")


def load_grid_combos(path: Path = GRID_PATH) -> list[dict]:
    """Charge la liste des combinaisons déclarées (`run_name` + surcharges Hydra)."""
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return data["runs"]


def has_existing_result(campaign_dir: Path, run_name: str) -> bool:
    """Un fichier de résultat `<run_name>-*.json(.gz)` existe déjà dans la campagne."""
    if not campaign_dir.exists():
        return False
    return any(campaign_dir.glob(f"{run_name}-*.json*"))
