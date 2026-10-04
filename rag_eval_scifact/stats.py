"""Test de randomisation apparié entre deux runs, et accord entre deux juges,
codés à la main (EXE-86, EXE-122).

Interdiction d'importer une lib de stats (scipy, statsmodels) pour ce test :
methodologie.md proscrit déjà les libs qui calculent une métrique d'éval à
notre place, ce ticket étend l'interdiction au test statistique. numpy reste
autorisé.

Standard en recherche d'information pour comparer deux systèmes sur les mêmes
requêtes (cf. Smucker et al., "A comparison of statistical significance tests
for information retrieval evaluation") : sous H0, le signe de la différence
par requête entre les deux systèmes est arbitraire. On estime la distribution
de la statistique sous H0 en tirant au hasard ce signe, requête par requête.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def paired_permutation_test(
    values_a: list[float],
    values_b: list[float],
    n_permutations: int = 10000,
    seed: int | None = None,
) -> dict[str, float]:
    """p-value bilatérale d'un test de randomisation apparié (b contre a).

    H0 : le signe de chaque différence (b - a), par requête, est tiré au
    hasard. Statistique observée : différence moyenne des deux séries.
    `n_permutations` tirages aléatoires de signes par requête estiment la
    distribution de cette statistique sous H0. Le run observé compte comme
    une permutation de plus, pour que la p-value ne tombe jamais à 0 :

        p = (1 + #{permutations aussi extrêmes que l'observée}) / (n_permutations + 1)
    """
    if len(values_a) != len(values_b):
        raise ValueError("les deux séries doivent porter sur les mêmes requêtes")

    diffs = np.asarray(values_b, dtype=np.float64) - np.asarray(
        values_a, dtype=np.float64
    )
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diffs.shape[0]))
    permuted_means = (signs * diffs).mean(axis=1)

    extreme = int(np.sum(np.abs(permuted_means) >= abs(observed)))
    p_value = (1 + extreme) / (n_permutations + 1)

    return {
        "mean_diff": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """Accord entre deux juges sur les mêmes items (EXE-122, H3).

    `kappa = (p_o - p_e) / (1 - p_e)` : `p_o` est l'accord observé (part de la
    diagonale du tableau croisé), `p_e` l'accord attendu par hasard à partir
    des fréquences marginales de chaque juge. Quand `p_e` vaut 1 (chaque juge
    a toujours donné la même unique catégorie), le dénominateur est nul ;
    `kappa` est alors `None`, à écrire « non défini » plutôt qu'un nombre.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("les deux séries doivent porter sur les mêmes items")
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    confusion = Counter(zip(labels_a, labels_b))
    row_totals = Counter(labels_a)
    col_totals = Counter(labels_b)

    observed = sum(confusion[(c, c)] for c in categories) / n
    expected = sum((row_totals[c] / n) * (col_totals[c] / n) for c in categories)
    denom = 1 - expected
    kappa = None if denom == 0 else (observed - expected) / denom

    return {
        "n": n,
        "categories": categories,
        "confusion": dict(confusion),
        "agreement": observed,
        "kappa": kappa,
    }
