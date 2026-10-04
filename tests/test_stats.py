"""Tests du test de randomisation apparié, codé à la main (EXE-86, critères 6 à 8).

Aucune lib de stats importée (scipy, statsmodels interdits par le ticket) : le
module sous test n'utilise que numpy.
"""

from __future__ import annotations

import pytest

from rag_eval_scifact.stats import cohen_kappa, paired_permutation_test

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


# ---------------------------------------------------------------------------
# EXE-122, critère 9 — kappa de Cohen, codé à la main
# ---------------------------------------------------------------------------

# Tableau croisé fabriqué, 10 paires, 8 accords :
#   A\B   S    R    N
#   S     4    1    0
#   R     0    3    0
#   N     1    0    1
# Marges A : S=5, R=3, N=2. Marges B : S=5, R=4, N=1.
# p_o = 8/10 = 0.8 ; p_e = (5*5 + 3*4 + 2*1) / 100 = 0.39
# kappa = (0.8 - 0.39) / (1 - 0.39) = 0.41 / 0.61
LABELS_A = ["S"] * 5 + ["R"] * 3 + ["N"] * 2
LABELS_B = ["S", "S", "S", "S", "R", "R", "R", "R", "S", "N"]


def test_cohen_kappa_on_handcrafted_8_of_10_agreement_table():
    result = cohen_kappa(LABELS_A, LABELS_B)

    assert result["agreement"] == pytest.approx(0.8)
    assert result["kappa"] == pytest.approx(0.41 / 0.61)


def test_cohen_kappa_is_one_for_two_identical_judges():
    labels = ["S", "S", "R", "N", "S", "R", "N", "N"]

    result = cohen_kappa(labels, labels)

    assert result["agreement"] == 1.0
    assert result["kappa"] == pytest.approx(1.0)


def test_cohen_kappa_is_undefined_when_both_judges_always_pick_one_category():
    labels = ["S"] * 5

    result = cohen_kappa(labels, labels)

    assert result["agreement"] == 1.0
    assert result["kappa"] is None
