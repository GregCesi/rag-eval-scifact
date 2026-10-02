"""Tests du regroupement passage -> document (EXE-92, critères 3 et 4).

Arithmétique pure sur des scores fabriqués : aucun modèle, aucun corpus,
aucune similarité recalculée.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_eval_scifact.retrieve import group_passage_scores

# d1 a 2 passages, d2 en a 1, d3 en a 3.
PASSAGE_DOC_IDS = ["d1", "d1", "d2", "d3", "d3", "d3"]
SCORES = np.array([0.2, 0.9, 0.5, 0.1, 0.8, 0.3], dtype=np.float32)


# ---------------------------------------------------------------------------
# Critère 4 — max (défaut) : le meilleur score parmi tous les passages du doc
# ---------------------------------------------------------------------------


def test_max_grouping_takes_the_best_score_among_all_of_a_document_passages():
    doc_scores = group_passage_scores(
        PASSAGE_DOC_IDS, SCORES, grouping="max", grouping_top_n=1000
    )

    assert doc_scores == pytest.approx({"d1": 0.9, "d2": 0.5, "d3": 0.8})


def test_max_grouping_ignores_grouping_top_n():
    # N=1 ne restreint jamais le max : chaque doc garde son vrai meilleur score.
    doc_scores = group_passage_scores(
        PASSAGE_DOC_IDS, SCORES, grouping="max", grouping_top_n=1
    )

    assert doc_scores == pytest.approx({"d1": 0.9, "d2": 0.5, "d3": 0.8})


# ---------------------------------------------------------------------------
# Critère 4 — sum : somme des scores parmi les N premiers passages de la requête
# ---------------------------------------------------------------------------


def test_sum_grouping_sums_scores_of_passages_within_top_n():
    # Top-3 passages (par score) : idx 1 (0.9, d1), idx 4 (0.8, d3), idx 2 (0.5, d2).
    doc_scores = group_passage_scores(
        PASSAGE_DOC_IDS, SCORES, grouping="sum", grouping_top_n=3
    )

    assert doc_scores == pytest.approx({"d1": 0.9, "d3": 0.8, "d2": 0.5})


def test_sum_grouping_sums_multiple_passages_of_the_same_document_within_top_n():
    # Top-4 : idx 1 (0.9, d1), idx 4 (0.8, d3), idx 2 (0.5, d2), idx 5 (0.3, d3).
    doc_scores = group_passage_scores(
        PASSAGE_DOC_IDS, SCORES, grouping="sum", grouping_top_n=4
    )

    assert doc_scores["d3"] == pytest.approx(0.8 + 0.3)
    assert doc_scores["d1"] == pytest.approx(0.9)
    assert doc_scores["d2"] == pytest.approx(0.5)


def test_sum_grouping_excludes_documents_with_no_passage_in_top_n():
    doc_scores = group_passage_scores(
        PASSAGE_DOC_IDS, SCORES, grouping="sum", grouping_top_n=1
    )

    # Seul idx 1 (0.9, d1) est dans le top-1.
    assert doc_scores == pytest.approx({"d1": 0.9})
