"""BM25 fait-main sur le tokenizer, scores calculés par `rank_bm25`.

`.claude/rules/methodologie.md` interdit les libs d'éval (métriques, fusion,
test statistique) mais autorise explicitement une librairie BM25 — seule la
tokenisation (minuscules, ponctuation retirée, sans racinisation, H2 EXE-93)
est fixée ici ; k1 et b restent des valeurs de configuration de l'appelant.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Minuscules, ponctuation retirée, sans racinisation."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(corpus_texts: list[str], k1: float, b: float) -> BM25Okapi:
    """Index BM25 sur un corpus (documents ou passages), tokenisé à la main."""
    return BM25Okapi([tokenize(t) for t in corpus_texts], k1=k1, b=b)


def score_queries(index: BM25Okapi, query_texts: list[str]) -> np.ndarray:
    """Matrice (n_requêtes, n_corpus) des scores BM25 pour `query_texts`."""
    return np.array(
        [index.get_scores(tokenize(q)) for q in query_texts], dtype=np.float32
    )
