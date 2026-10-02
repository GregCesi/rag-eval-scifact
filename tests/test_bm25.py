"""Tests de `rag_eval_scifact.bm25` (EXE-93, critère 7).

Corpus fabriqué de 3 documents, aucun modèle chargé : `rank_bm25` est une
librairie BM25, permise par `.claude/rules/methodologie.md` (seules les libs
d'éval sont interdites).
"""

from __future__ import annotations

from rag_eval_scifact.bm25 import build_bm25_index, score_queries, tokenize

CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a rare xylophone solo echoed through the cat sanctuary",
]
QUERY = ["xylophone"]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Cats, dogs -- and MATS!") == ["cats", "dogs", "and", "mats"]


def test_document_with_the_rare_query_term_ranks_first():
    index = build_bm25_index(CORPUS, k1=1.2, b=0.75)
    scores = score_queries(index, QUERY)[0]

    assert int(scores.argmax()) == 2


def test_changing_k1_changes_the_scores():
    index_default = build_bm25_index(CORPUS, k1=1.2, b=0.75)
    index_other_k1 = build_bm25_index(CORPUS, k1=5.0, b=0.75)

    scores_default = score_queries(index_default, QUERY)[0]
    scores_other = score_queries(index_other_k1, QUERY)[0]

    assert list(scores_default) != list(scores_other)


def test_changing_b_changes_the_scores():
    index_default = build_bm25_index(CORPUS, k1=1.2, b=0.75)
    index_other_b = build_bm25_index(CORPUS, k1=1.2, b=0.1)

    scores_default = score_queries(index_default, QUERY)[0]
    scores_other = score_queries(index_other_b, QUERY)[0]

    assert list(scores_default) != list(scores_other)
