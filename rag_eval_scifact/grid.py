"""Grille de campagne : combinaisons déclarées, une par entrée, une par campagne.

Un fichier déclaratif par campagne (`conf/grid/<campagne>.yaml`) plutôt qu'un
sweep Hydra : la grille est une liste de stratégies choisies à la main, pas un
produit cartésien. Chaque entrée ajoute des surcharges Hydra
(`retriever.model=...`) au-dessus de `conf/config.yaml`. `conf/grid/v2-grid.yaml`
reste la grille historique (34 runs, EXE-91 à EXE-96) ; `conf/grid/v4-leviers.yaml`
(EXE-140) est la première grille d'une autre campagne, même format.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

GRID_DIR = Path("conf/grid")
DEFAULT_CAMPAGNE = "v2-grid"


class GridFileMissing(Exception):
    """Levée quand une campagne n'a pas de fichier `conf/grid/<campagne>.yaml`."""


def grid_path_for_campagne(campagne: str, grid_dir: Path = GRID_DIR) -> Path:
    """Chemin attendu du fichier de grille d'une campagne."""
    return grid_dir / f"{campagne}.yaml"


def load_grid_combos(
    path: Path | str | None = None, campagne: str = DEFAULT_CAMPAGNE
) -> list[dict]:
    """Charge la liste des combinaisons déclarées (`run_name` + surcharges Hydra).

    `path`, quand fourni, prime sur `campagne` (compatibilité des appels qui
    pointent un chemin explicite, ex. tests). Sinon résolu depuis `campagne`
    (`grid_path_for_campagne`, défaut `v2-grid` — comportement inchangé pour
    un appel sans argument). Lève `GridFileMissing` si le fichier résolu est
    absent : au CLI (`run_grid.py`), ceci devient une phrase, jamais une trace
    Python brute.
    """
    resolved = Path(path) if path is not None else grid_path_for_campagne(campagne)
    if not resolved.exists():
        raise GridFileMissing(str(resolved))
    data = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    return data["runs"]


def has_existing_result(campaign_dir: Path, run_name: str) -> bool:
    """Un fichier de résultat `<run_name>-*.json(.gz)` existe déjà dans la campagne."""
    if not campaign_dir.exists():
        return False
    return any(campaign_dir.glob(f"{run_name}-*.json*"))
