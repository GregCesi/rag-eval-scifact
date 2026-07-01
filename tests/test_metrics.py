"""Tests unitaires des métriques fait-main — oracles numériques en dur.

Chaque valeur oracle a été calculée indépendamment du code testé.
Tolérance : 1e-4 sur toutes les comparaisons flottantes.

Cas couverts (cf. IMPLEMENTATION_eval_system.md §Phase 1, livrable 1.4) :
  trivial : 1 doc pertinent au rang 1
  rang 2  : 1 doc pertinent au rang 2
  rang 3  : 1 doc pertinent au rang 3
  absent  : 1 doc pertinent hors top-100
  multi   : 2 docs pertinents aux rangs 2 et 5
"""

import pytest

from rag_eval_scifact.metrics import mrr, ndcg_at_k, recall_at_k

TOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures : listes de retrieved_ids simulées
# ---------------------------------------------------------------------------

def _make_retrieved(relevant_at_ranks: dict[str, int], total: int = 100) -> list[str]:
    """Construit une liste de `total` doc_ids avec les docs pertinents placés aux rangs voulus.

    relevant_at_ranks : {doc_id: rang (1-based)}
    Les autres positions sont remplies par des ids non pertinents.
    """
    result = [f"noise_{i}" for i in range(total)]
    for doc_id, rank in relevant_at_ranks.items():
        result[rank - 1] = doc_id
    return result


# --- Cas trivial : 1 doc pertinent au rang 1 ---

TRIVIAL_RETRIEVED = _make_retrieved({"rel_1": 1})
TRIVIAL_RELEVANT = {"rel_1"}

# --- Cas rang 2 : 1 doc pertinent au rang 2 ---

RANK2_RETRIEVED = _make_retrieved({"rel_1": 2})
RANK2_RELEVANT = {"rel_1"}

# --- Cas rang 3 : 1 doc pertinent au rang 3 ---

RANK3_RETRIEVED = _make_retrieved({"rel_1": 3})
RANK3_RELEVANT = {"rel_1"}

# --- Cas absent : 1 doc pertinent hors top-100 ---

ABSENT_RETRIEVED = _make_retrieved({}, total=100)  # aucun doc pertinent dans la liste
ABSENT_RELEVANT = {"rel_1"}

# --- Cas multi : 2 docs pertinents aux rangs 2 et 5 ---

MULTI_RETRIEVED = _make_retrieved({"rel_1": 2, "rel_2": 5})
MULTI_RELEVANT = {"rel_1", "rel_2"}


# ===========================================================================
# Tests Recall@k
# ===========================================================================

class TestRecallAtK:
    """Recall@k = |{pertinents ∩ top-k}| / |{pertinents}|"""

    @pytest.mark.parametrize("k, expected", [(1, 1.0), (5, 1.0), (10, 1.0), (100, 1.0)])
    def test_trivial(self, k, expected):
        assert recall_at_k(TRIVIAL_RETRIEVED, TRIVIAL_RELEVANT, k) == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize("k, expected", [(1, 0.0), (5, 1.0), (10, 1.0), (100, 1.0)])
    def test_rank2(self, k, expected):
        assert recall_at_k(RANK2_RETRIEVED, RANK2_RELEVANT, k) == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize("k, expected", [(1, 0.0), (5, 1.0), (10, 1.0), (100, 1.0)])
    def test_rank3(self, k, expected):
        assert recall_at_k(RANK3_RETRIEVED, RANK3_RELEVANT, k) == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize("k, expected", [(1, 0.0), (5, 0.0), (10, 0.0), (100, 0.0)])
    def test_absent(self, k, expected):
        assert recall_at_k(ABSENT_RETRIEVED, ABSENT_RELEVANT, k) == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize("k, expected", [(1, 0.0), (5, 1.0), (10, 1.0), (100, 1.0)])
    def test_multi(self, k, expected):
        assert recall_at_k(MULTI_RETRIEVED, MULTI_RELEVANT, k) == pytest.approx(expected, abs=TOL)


# ===========================================================================
# Tests nDCG@10
# ===========================================================================

class TestNdcgAtK:
    """nDCG@10 avec relevance binaire.

    Oracles calculés à la main :
      trivial : DCG = 1/log2(2) = 1 ; IDCG = 1 ; nDCG = 1.0
      rang 2  : DCG = 1/log2(3) = 0.6309 ; IDCG = 1 ; nDCG = 0.6309
      rang 3  : DCG = 1/log2(4) = 0.5 ; IDCG = 1 ; nDCG = 0.5
      absent  : DCG = 0 ; nDCG = 0.0
      multi   : DCG = 1/log2(3) + 1/log2(6) = 1.0178 ; IDCG = 1/log2(2) + 1/log2(3) = 1.6309 ; nDCG = 0.6240
    """

    def test_trivial(self):
        assert ndcg_at_k(TRIVIAL_RETRIEVED, TRIVIAL_RELEVANT, 10) == pytest.approx(1.0, abs=TOL)

    def test_rank2(self):
        assert ndcg_at_k(RANK2_RETRIEVED, RANK2_RELEVANT, 10) == pytest.approx(0.6309, abs=TOL)

    def test_rank3(self):
        assert ndcg_at_k(RANK3_RETRIEVED, RANK3_RELEVANT, 10) == pytest.approx(0.5, abs=TOL)

    def test_absent(self):
        assert ndcg_at_k(ABSENT_RETRIEVED, ABSENT_RELEVANT, 10) == pytest.approx(0.0, abs=TOL)

    def test_multi(self):
        assert ndcg_at_k(MULTI_RETRIEVED, MULTI_RELEVANT, 10) == pytest.approx(0.6240, abs=TOL)


# ===========================================================================
# Tests MRR
# ===========================================================================

class TestMrr:
    """MRR = 1/rang du premier doc pertinent (0 si absent)."""

    def test_trivial(self):
        assert mrr(TRIVIAL_RETRIEVED, TRIVIAL_RELEVANT) == pytest.approx(1.0, abs=TOL)

    def test_rank2(self):
        assert mrr(RANK2_RETRIEVED, RANK2_RELEVANT) == pytest.approx(0.5, abs=TOL)

    def test_rank3(self):
        assert mrr(RANK3_RETRIEVED, RANK3_RELEVANT) == pytest.approx(0.3333, abs=TOL)

    def test_absent(self):
        assert mrr(ABSENT_RETRIEVED, ABSENT_RELEVANT) == pytest.approx(0.0, abs=TOL)

    def test_multi(self):
        assert mrr(MULTI_RETRIEVED, MULTI_RELEVANT) == pytest.approx(0.5, abs=TOL)


# ===========================================================================
# Garde-fou : IDCG = 0
# ===========================================================================

class TestEdgeCases:
    """nDCG doit retourner 0.0 si aucun doc pertinent (IDCG = 0), pas une division par zéro."""

    def test_ndcg_no_relevant_docs(self):
        retrieved = [f"doc_{i}" for i in range(10)]
        assert ndcg_at_k(retrieved, set(), 10) == 0.0
