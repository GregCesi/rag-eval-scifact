"""Point d'entrée CLI de la grille d'une campagne : lister, lancer, essayer.

Mode liste : affiche les combinaisons déclarées dans `conf/grid/<campagne>.yaml`,
une par ligne, puis leur nombre, sans rien lancer. Mode lancement : compose
chaque combinaison avec `conf/config.yaml` (lanceur Hydra) et la lance comme un
run de campagne, en réutilisant `run_campaign.main` tel quel — un levier reste
une dimension de config Hydra, jamais un chemin de code dédié. Mode essai :
chronomètre l'encodage d'un échantillon d'un run nommé, sans rien écrire.

Usage : python -m rag_eval_scifact.run_grid --list
        python -m rag_eval_scifact.run_grid --list --campagne v4-leviers
        python -m rag_eval_scifact.run_grid
        python -m rag_eval_scifact.run_grid --campagne dev
        python -m rag_eval_scifact.run_grid --campagne v4-leviers --run qwen3-passages-reference
        python -m rag_eval_scifact.run_grid --campagne v4-leviers --essai medcpt-passages
        python -m rag_eval_scifact.run_grid --campagne v5-rerankers --essai-rerank rerank-minilm-top20
"""

from __future__ import annotations

import argparse
import subprocess

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import rag_eval_scifact.campaign as campaign_module
import rag_eval_scifact.run_campaign as run_campaign_module
from rag_eval_scifact.grid import GridFileMissing, has_existing_result, load_grid_combos
from rag_eval_scifact.hyde import N_CLAIMS
from rag_eval_scifact.ingest import CORPUS_PATH, load_corpus
from rag_eval_scifact.rerank import essai_rerank_campaign
from rag_eval_scifact.retrieve import essai_embedding, retrieve_campaign
from rag_eval_scifact.run_campaign import CONF_DIR

DEFAULT_CAMPAGNE = "v2-grid"
UNGATED_CAMPAGNE = "dev"


def _prediction_committed(campagne: str) -> bool:
    """`results/<campagne>/PREDICTION.md` est présent dans le dernier commit."""
    path = f"results/{campagne}/PREDICTION.md"
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _load_combos_or_exit(campagne: str) -> list[dict]:
    """Charge la grille d'une campagne ; une phrase, jamais une trace Python,
    quand le fichier attendu est absent (critère 4)."""
    try:
        return load_grid_combos(campagne=campagne)
    except GridFileMissing as exc:
        print(
            f"Aucun fichier de grille pour la campagne « {campagne} » (attendu : {exc})."
        )
        raise SystemExit(1)


def _find_combo_or_exit(combos: list[dict], run_name: str, campagne: str) -> dict:
    """Résout un run nommé dans la grille ; liste les noms connus s'il est
    absent (critère 5)."""
    combo = next((c for c in combos if c["run_name"] == run_name), None)
    if combo is None:
        known = ", ".join(c["run_name"] for c in combos)
        print(f"Run inconnu dans {campagne} : « {run_name} ». Runs connus : {known}.")
        raise SystemExit(1)
    return combo


def _compose_cfg(campagne: str, combo: dict):
    overrides = [
        f"campagne={campagne}",
        f"run_name={combo['run_name']}",
        *combo["overrides"],
    ]
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


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
        default=DEFAULT_CAMPAGNE,
        help=f"Campagne cible (défaut : {DEFAULT_CAMPAGNE}).",
    )
    parser.add_argument(
        "--run",
        dest="run_name",
        default=None,
        help="Lance uniquement ce run nommé (sinon, tous les runs de la campagne).",
    )
    parser.add_argument(
        "--essai",
        dest="essai_run_name",
        default=None,
        help=(
            "Essai rapide (50 premiers documents) de ce run nommé : affiche la "
            "durée et son extrapolation au corpus entier, n'écrit rien."
        ),
    )
    parser.add_argument(
        "--essai-rerank",
        dest="essai_rerank_run_name",
        default=None,
        help=(
            "Essai rapide du reranker de ce run nommé : reclasse les top_n "
            "premiers documents des 10 premières affirmations, affiche la "
            f"durée et son extrapolation aux {N_CLAIMS} affirmations, n'écrit "
            "rien."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_only:
        combos = _load_combos_or_exit(args.campagne)
        for combo in combos:
            print(combo["run_name"])
        print(f"{len(combos)} combinaison(s)")
        return

    if args.essai_run_name is not None:
        combos = _load_combos_or_exit(args.campagne)
        combo = _find_combo_or_exit(combos, args.essai_run_name, args.campagne)
        cfg = _compose_cfg(args.campagne, combo)
        docs = load_corpus(CORPUS_PATH)
        stats = essai_embedding(
            corpus=docs,
            model_name=cfg.retriever.model,
            max_seq_length=cfg.retriever.max_seq_length,
            unit=cfg.retriever.unit,
            chunk_size=cfg.retriever.chunk_size,
            chunk_overlap=cfg.retriever.chunk_overlap,
            batch_size=cfg.retriever.batch_size,
            pooling=cfg.retriever.pooling,
        )
        print(
            f"Essai {combo['run_name']} : {int(stats['n_units'])} passages (50 premiers documents)"
        )
        print(f"  Durée           : {stats['duration_seconds']:.2f} s")
        print(
            f"  Extrapolation   : {stats['extrapolated_minutes']:.1f} min (corpus entier)"
        )
        return

    if args.essai_rerank_run_name is not None:
        combos = _load_combos_or_exit(args.campagne)
        combo = _find_combo_or_exit(combos, args.essai_rerank_run_name, args.campagne)
        cfg = _compose_cfg(args.campagne, combo)
        if cfg.rerank is None or cfg.rerank.name != "cross-encoder":
            print(
                f"{combo['run_name']} ne configure aucun reranker "
                "(rerank.name=cross-encoder attendu)."
            )
            raise SystemExit(1)

        results, _qrels, _dataset_hash, _stats = retrieve_campaign(
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
            retriever_name=cfg.retriever.name,
            bm25_k1=cfg.retriever.bm25_k1,
            bm25_b=cfg.retriever.bm25_b,
            fusion_mode=cfg.retriever.fusion_mode,
            rrf_k=cfg.retriever.rrf_k,
            query_instruction=cfg.retriever.query_instruction,
            batch_size=cfg.retriever.batch_size,
            query_model_name=cfg.retriever.query_model or None,
            query_max_seq_length=cfg.retriever.query_max_seq_length or None,
            pooling=cfg.retriever.pooling,
            query_source=cfg.retriever.query_source,
        )
        doc_texts = {
            doc["_id"]: doc["title"] + " " + doc["text"]
            for doc in load_corpus(CORPUS_PATH)
        }
        stats = essai_rerank_campaign(
            results,
            doc_texts,
            top_n=cfg.rerank.top_n,
            model_name=cfg.rerank.model,
            instruction=cfg.rerank.instruction,
            half_precision=cfg.rerank.half_precision,
            n_claims=N_CLAIMS,
        )
        print(f"Essai reranker {combo['run_name']} : 10 premières affirmations")
        print(f"  Durée           : {stats['duration_seconds']:.2f} s")
        print(
            f"  Extrapolation   : {stats['extrapolated_minutes']:.1f} min "
            f"({N_CLAIMS} affirmations)"
        )
        return

    combos = _load_combos_or_exit(args.campagne)

    if args.run_name is not None:
        combos = [_find_combo_or_exit(combos, args.run_name, args.campagne)]

    if args.campagne != UNGATED_CAMPAGNE and not _prediction_committed(args.campagne):
        print(
            f"Lancement refusé : results/{args.campagne}/PREDICTION.md n'est pas "
            "dans le dernier commit (.claude/rules/methodologie.md)."
        )
        raise SystemExit(1)

    campaign_dir = campaign_module.RESULTS_DIR / args.campagne
    for combo in combos:
        run_name = combo["run_name"]
        if has_existing_result(campaign_dir, run_name):
            print(f"{run_name} : déjà fait, ignoré")
            continue

        cfg = _compose_cfg(args.campagne, combo)
        run_campaign_module.main(cfg)


if __name__ == "__main__":
    main()
