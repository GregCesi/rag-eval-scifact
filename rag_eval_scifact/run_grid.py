"""Point d'entrée CLI de la grille v2-grid : lister ou lancer ses combinaisons.

Mode liste : affiche les combinaisons déclarées dans `conf/grid/v2-grid.yaml`,
une par ligne, puis leur nombre, sans rien lancer. Mode lancement : compose
chaque combinaison avec `conf/config.yaml` (lanceur Hydra) et la lance comme un
run de campagne, en réutilisant `run_campaign.main` tel quel — un levier reste
une dimension de config Hydra, jamais un chemin de code dédié.

Usage : python -m rag_eval_scifact.run_grid --list
        python -m rag_eval_scifact.run_grid
        python -m rag_eval_scifact.run_grid --campagne dev
"""

from __future__ import annotations

import argparse
import subprocess

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import rag_eval_scifact.campaign as campaign_module
import rag_eval_scifact.run_campaign as run_campaign_module
from rag_eval_scifact.grid import has_existing_result, load_grid_combos
from rag_eval_scifact.run_campaign import CONF_DIR

GATED_CAMPAGNE = "v2-grid"
PREDICTION_PATH = "results/v2-grid/PREDICTION.md"


def _prediction_committed() -> bool:
    """`PREDICTION_PATH` est présent dans le dernier commit (`git show HEAD:...`)."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{PREDICTION_PATH}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Affiche les combinaisons déclarées et leur nombre, sans rien lancer.",
    )
    parser.add_argument(
        "--campagne",
        default=GATED_CAMPAGNE,
        help=f"Campagne cible des runs lancés (défaut : {GATED_CAMPAGNE}).",
    )
    args = parser.parse_args(argv)

    combos = load_grid_combos()

    if args.list_only:
        for combo in combos:
            print(combo["run_name"])
        print(f"{len(combos)} combinaison(s)")
        return

    if args.campagne == GATED_CAMPAGNE and not _prediction_committed():
        print(
            f"Lancement refusé : {PREDICTION_PATH} n'est pas dans le dernier "
            "commit (.claude/rules/methodologie.md)."
        )
        raise SystemExit(1)

    campaign_dir = campaign_module.RESULTS_DIR / args.campagne
    for combo in combos:
        run_name = combo["run_name"]
        if has_existing_result(campaign_dir, run_name):
            print(f"{run_name} : déjà fait, ignoré")
            continue

        overrides = [
            f"campagne={args.campagne}",
            f"run_name={run_name}",
            *combo["overrides"],
        ]
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)
        run_campaign_module.main(cfg)


if __name__ == "__main__":
    main()
