"""Point d'entrée CLI d'un run de campagne : une stratégie se déclare par une config Hydra.

Sans argument, reproduit v1 à l'identique (`conf/config.yaml`). Une stratégie se
surcharge en ligne de commande (`top_k=10`) ou par un fichier de config dérivé,
jamais en modifiant le pipeline. Plusieurs stratégies se lancent d'une seule
commande avec le sweeper Hydra standard (`--multirun`, alias `-m`).

Usage : python -m rag_eval_scifact.run_campaign
        python -m rag_eval_scifact.run_campaign top_k=10
        python -m rag_eval_scifact.run_campaign --multirun top_k=10,20
        python -m rag_eval_scifact.run_campaign --help
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from rag_eval_scifact.campaign import run_campaign
from rag_eval_scifact.ingest import CORPUS_PATH, load_corpus
from rag_eval_scifact.mlflow_tracking import log_campaign_run, log_query_traces
from rag_eval_scifact.retrieve import retrieve_campaign
from rag_eval_scifact.run_output import get_token_counts

# Chemin absolu : résolu en filesystem pur par Hydra, sans dépendre de la
# détection de module appelant (qui diffère entre `python -m` et un appel direct
# de `_cli()`, ce second cas servant les tests du multirun — TCK/EXE-88).
CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    run_date = datetime.now(UTC)

    print(f"Campagne : {cfg.campagne} — run : {cfg.run_name}")
    results, qrels, dataset_hash, stats = retrieve_campaign(
        top_k=cfg.top_k,
        model_name=cfg.retriever.model,
        max_seq_length=cfg.retriever.max_seq_length,
        split=cfg.split,
        cache_dir=cfg.cache_dir,
        unit=cfg.retriever.unit,
        chunk_size=cfg.retriever.chunk_size,
        chunk_overlap=cfg.retriever.chunk_overlap,
        grouping=cfg.retriever.grouping,
        grouping_top_n=cfg.retriever.grouping_top_n,
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
        truncated_pct=stats["truncated_pct"],
        avg_retrieval_latency_ms=stats["avg_retrieval_latency_ms"],
        indexing_duration_seconds=stats["indexing_duration_seconds"],
        n_passages=stats.get("n_passages", 0.0),
    )

    mlflow_run_id = log_campaign_run(
        campagne=cfg.campagne,
        run_name=cfg.run_name,
        config=config,
        metrics=outcome["metrics"],
        json_path=outcome["json_path"],
        extended_metrics=outcome["extended_metrics"],
    )

    n_traces = 0
    if cfg.tracing:
        titles = {doc["_id"]: doc["title"] for doc in load_corpus(CORPUS_PATH)}
        log_query_traces(
            run_id=mlflow_run_id, results=results, qrels=qrels, titles=titles
        )
        n_traces = len(results)

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
    print(f"  MLflow: expérience '{cfg.campagne}', run {mlflow_run_id}")
    print(
        f"  Traces: {n_traces} requêtes tracées"
        if cfg.tracing
        else "  Traces: désactivées (tracing=false)"
    )
    print("=" * 60)
    print("\n  Rappel : commiter les artefacts (run non commité = run inexistant)")


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def _cli(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    # Le venv du projet est pinné à Python 3.13 (scripts/preflight.sh) précisément
    # pour ce lanceur : sous Python 3.14, la construction de l'argparse.ArgumentParser
    # interne de hydra.main() lève TypeError. Sous 3.13, le CLI décoré standard
    # fonctionne et porte --help et --multirun (sweeper Hydra standard).
    _cli()
