"""Point d'entrée CLI d'un run de campagne : une stratégie se déclare par une config Hydra.

Sans argument, reproduit v1 à l'identique (`conf/config.yaml`). Une stratégie se
surcharge en ligne de commande (`top_k=10`) ou par un fichier de config dérivé,
jamais en modifiant le pipeline.

Usage : python -m rag_eval_scifact.run_campaign
        python -m rag_eval_scifact.run_campaign top_k=10
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from rag_eval_scifact.campaign import run_campaign
from rag_eval_scifact.retrieve import retrieve
from rag_eval_scifact.run_output import get_token_counts


def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    run_date = datetime.now(UTC)

    print(f"Campagne : {cfg.campagne} — run : {cfg.run_name}")
    results, qrels, dataset_hash = retrieve(
        top_k=cfg.top_k,
        model_name=cfg.retriever.model,
        max_seq_length=cfg.retriever.max_seq_length,
    )

    all_relevant_ids: set[str] = set()
    for relevant_set in qrels.values():
        all_relevant_ids.update(relevant_set)
    token_counts = get_token_counts(list(all_relevant_ids))

    outcome = run_campaign(
        campagne=cfg.campagne,
        run_name=cfg.run_name,
        config=config,
        results=results,
        qrels=qrels,
        dataset_hash=dataset_hash,
        run_date=run_date,
        token_counts=token_counts,
    )

    print("\n" + "=" * 60)
    print(
        f"  RUN DE CAMPAGNE — {cfg.campagne}/{cfg.run_name} — {run_date.isoformat(timespec='seconds')}"
    )
    print("=" * 60)
    print(f"  dataset_hash : {dataset_hash}")
    for k, v in outcome["metrics"].items():
        print(f"  {k:12s} = {v:.4f}")
    print()
    print(f"  JSON  : {outcome['json_path']}")
    print("  TABLE : RESULTS.md (ligne ajoutée)")
    print("=" * 60)
    print("\n  Rappel : commiter les artefacts (run non commité = run inexistant)")


if __name__ == "__main__":
    # hydra.main() construit son propre argparse.ArgumentParser au chargement du
    # module ; sous Python 3.14 cette construction lève TypeError (argparse
    # vérifie `'%' in help_string` sur un objet non-str que Hydra 1.3 lui passe).
    # On compose la config nous-mêmes avec l'API compose() (celle que Hydra
    # documente pour les notebooks/tests) : Hydra reste le moyen de déclarer et
    # surcharger la config, sans passer par le CLI décoré qui plante ici.
    with initialize(version_base=None, config_path="../conf"):
        resolved_cfg = compose(config_name="config", overrides=sys.argv[1:])
    main(resolved_cfg)
