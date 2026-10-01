"""Tests du test de randomisation apparié, codé à la main (EXE-86, critères 6 à 8).

Aucune lib de stats importée (scipy, statsmodels interdits par le ticket) : le
module sous test n'utilise que numpy.
"""

from __future__ import annotations

import pytest

from rag_eval_scifact.stats import paired_permutation_test

N_QUERIES = 300
N_PERMUTATIONS = 10_000


# ---------------------------------------------------------------------------
# Critère 6 — une série comparée à elle-même : diff moyenne nulle, p-value 1
# ---------------------------------------------------------------------------


def test_identical_series_have_zero_mean_diff_and_p_value_one():
    values = [0.1 * i for i in range(N_QUERIES)]

    result = paired_permutation_test(
        values, values, n_permutations=N_PERMUTATIONS, seed=0
    )

    assert result["mean_diff"] == 0.0
    assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Critère 7 — écart constant de 0,1 sur chaque requête : p-value < 0.001
# ---------------------------------------------------------------------------


def test_constant_gap_of_one_tenth_is_detected_as_significant():
    values_a = [0.5] * N_QUERIES
    values_b = [0.6] * N_QUERIES

    result = paired_permutation_test(
        values_a, values_b, n_permutations=N_PERMUTATIONS, seed=0
    )

    assert result["mean_diff"] == pytest.approx(0.1)
    assert result["p_value"] < 0.001


# ---------------------------------------------------------------------------
# Critère 8 — même graine, deux lancements : p-values identiques
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_p_value_across_runs():
    values_a = [0.5] * N_QUERIES
    values_b = [0.6] * N_QUERIES

    first = paired_permutation_test(
        values_a, values_b, n_permutations=N_PERMUTATIONS, seed=42
    )
    second = paired_permutation_test(
        values_a, values_b, n_permutations=N_PERMUTATIONS, seed=42
    )

    assert first["p_value"] == second["p_value"]
