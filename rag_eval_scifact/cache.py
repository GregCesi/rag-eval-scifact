"""Cache disque des calculs lourds d'une campagne : embeddings et classement.

Hors du dépôt (défaut `~/.cache/rag-eval-scifact`, valeur de configuration
`cache_dir`). Chaque entrée est rangée par les paramètres qui la déterminent ;
changer l'un d'eux ne réutilise jamais l'entrée d'un autre. `chroma_data/`
reste l'index de `run_eval.py` (v1 historique) — ce module ne le touche pas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _safe(component: str) -> str:
    """Rend une chaîne sûre pour un segment de chemin (modèle, hash)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", component)


def document_embeddings_cache_path(
    cache_dir: Path | str,
    dataset_hash: str,
    model_name: str,
    max_seq_length: int,
    unit: str = "document",
) -> Path:
    return (
        Path(cache_dir).expanduser()
        / "embeddings"
        / _safe(dataset_hash)
        / _safe(model_name)
        / f"window-{max_seq_length}"
        / _safe(unit)
        / "documents.npz"
    )


def ranking_cache_path(
    cache_dir: Path | str,
    dataset_hash: str,
    model_name: str,
    max_seq_length: int,
    top_k: int,
    split: str,
    unit: str = "document",
) -> Path:
    return (
        Path(cache_dir).expanduser()
        / "rankings"
        / _safe(dataset_hash)
        / _safe(model_name)
        / f"window-{max_seq_length}"
        / _safe(unit)
        / f"top{top_k}"
        / f"{split}.json"
    )


def get_document_embeddings(
    cache_dir: Path | str,
    dataset_hash: str,
    model_name: str,
    max_seq_length: int,
    doc_ids: list[str],
    compute_fn: Callable[[], np.ndarray],
    unit: str = "document",
) -> tuple[np.ndarray, bool]:
    """Embeddings des documents (ou des passages), depuis le cache si présent.

    `unit` range le découpage (`"document"` ou `"passages-{taille}-{chevauchement}"`,
    EXE-92) : changer de découpage ne réutilise jamais l'entrée d'un autre.
    `compute_fn` n'est appelé qu'en cas d'absence du cache. Retourne
    (embeddings, cache_hit).
    """
    path = document_embeddings_cache_path(
        cache_dir, dataset_hash, model_name, max_seq_length, unit
    )
    if path.exists():
        data = np.load(path, allow_pickle=False)
        cached_ids = [str(d) for d in data["doc_ids"]]
        if cached_ids == list(doc_ids):
            print(f"  [cache embeddings] trouvé ({path}) — aucun document ré-embeddé.")
            return data["embeddings"].astype(np.float32), True
        print(
            f"  [cache embeddings] {path} présent mais doc_ids différents — recalcul."
        )

    print(f"  [cache embeddings] absent — calcul de {len(doc_ids)} documents...")
    embeddings = np.asarray(compute_fn(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, doc_ids=np.array(doc_ids), embeddings=embeddings)
    return embeddings, False


def get_ranking(
    cache_dir: Path | str,
    dataset_hash: str,
    model_name: str,
    max_seq_length: int,
    top_k: int,
    split: str,
    compute_fn: Callable[[], list[dict[str, Any]]],
    unit: str = "document",
) -> tuple[list[dict[str, Any]], bool]:
    """Classement de premier étage par requête, depuis le cache si présent.

    `unit` range le classement comme `document_embeddings_cache_path` (EXE-92) :
    pour les passages, inclut aussi le regroupement (`max`/`sum`, N) puisqu'il
    déterminent le classement final, contrairement aux embeddings.
    `compute_fn` n'est appelé qu'en cas d'absence du cache (aucune similarité
    recalculée sur un hit). Retourne (ranking, cache_hit) ; chaque élément de
    `ranking` est {"query_id", "query_text", "retrieved": [...]}.
    """
    path = ranking_cache_path(
        cache_dir, dataset_hash, model_name, max_seq_length, top_k, split, unit
    )
    if path.exists():
        print(f"  [cache classement] trouvé ({path}) — aucune similarité recalculée.")
        with open(path, encoding="utf-8") as f:
            return json.load(f), True

    print("  [cache classement] absent — calcul du classement de premier étage...")
    ranking = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranking, f)
    return ranking, False
