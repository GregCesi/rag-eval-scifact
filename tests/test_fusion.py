"""Tests de la fusion hybride dense + BM25, faits main (EXE-95 critère 8).

Classements fabriqués, aucun modèle chargé.
"""

from __future__ import annotations

from rag_eval_scifact.fusion import rrf_fuse, union_fuse


def _ranked(doc_ids: list[str]) -> list[dict]:
    return [
        {"doc_id": doc_id, "rank": rank, "score": 1.0 / rank}
        for rank, doc_id in enumerate(doc_ids, start=1)
    ]


# ---------------------------------------------------------------------------
# Union par alternance (critère 2)
# ---------------------------------------------------------------------------


def test_union_alternates_dense_and_bm25_starting_with_dense():
    dense = _ranked(["d1", "d2", "d3"])
    bm25 = _ranked(["b1", "b2", "b3"])

    fused = union_fuse(dense, bm25, top_k=100)

    assert [e["doc_id"] for e in fused] == ["d1", "b1", "d2", "b2", "d3", "b3"]
    assert [e["rank"] for e in fused] == [1, 2, 3, 4, 5, 6]


def test_union_skips_a_document_already_placed():
    dense = _ranked(["d1", "d2"])
    bm25 = _ranked(["d1", "b2"])  # d1 déjà placé par dense, sauté côté BM25

    fused = union_fuse(dense, bm25, top_k=100)

    assert [e["doc_id"] for e in fused] == ["d1", "d2", "b2"]


def test_union_caps_at_100_documents():
    dense = _ranked([f"d{i}" for i in range(1, 60)])
    bm25 = _ranked([f"b{i}" for i in range(1, 60)])

    fused = union_fuse(dense, bm25, top_k=100)

    assert len(fused) == 100


def test_union_only_considers_the_first_50_of_each():
    dense = _ranked([f"d{i}" for i in range(1, 60)])  # 59 docs, au-delà de 50
    bm25 = _ranked([f"b{i}" for i in range(1, 60)])

    fused = union_fuse(dense, bm25, top_k=200)

    doc_ids = {e["doc_id"] for e in fused}
    assert "d51" not in doc_ids
    assert "b51" not in doc_ids
    assert len(fused) == 100  # 50 + 50, pas plus même avec top_k=200


# ---------------------------------------------------------------------------
# Fusion RRF (critère 3)
# ---------------------------------------------------------------------------


def test_rrf_score_matches_hand_computed_sum():
    dense = _ranked(["d1", "d2"])  # rangs 1, 2
    bm25 = _ranked(["d1", "d3"])  # rangs 1, 2 ; d1 en commun

    fused = rrf_fuse(dense, bm25, k=60, top_k=100)

    scores = {e["doc_id"]: e["score"] for e in fused}
    assert scores["d1"] == 1 / 61 + 1 / 61
    assert scores["d2"] == 1 / 62
    assert scores["d3"] == 1 / 62


def test_rrf_orders_by_descending_fused_score():
    dense = _ranked(["d1", "d2", "d3"])  # rangs 1, 2, 3
    bm25 = _ranked(["d3", "d1", "d2"])  # rangs 1, 2, 3

    fused = rrf_fuse(dense, bm25, k=60, top_k=100)

    hand_computed = {
        "d1": 1 / 61 + 1 / 62,  # dense rang 1, bm25 rang 2
        "d2": 1 / 62 + 1 / 63,  # dense rang 2, bm25 rang 3
        "d3": 1 / 63 + 1 / 61,  # dense rang 3, bm25 rang 1
    }
    expected_order = sorted(
        hand_computed, key=lambda doc_id: hand_computed[doc_id], reverse=True
    )

    assert [e["doc_id"] for e in fused] == expected_order
    assert [e["score"] for e in fused] == [hand_computed[d] for d in expected_order]


def test_rrf_only_considers_the_first_100_of_each():
    dense = _ranked([f"d{i}" for i in range(1, 150)])
    bm25: list[dict] = []

    fused = rrf_fuse(dense, bm25, k=60, top_k=200)

    doc_ids = {e["doc_id"] for e in fused}
    assert "d101" not in doc_ids
    assert len(fused) == 100


def test_rrf_caps_the_output_at_top_k():
    dense = _ranked([f"d{i}" for i in range(1, 10)])
    bm25 = _ranked([f"b{i}" for i in range(1, 10)])

    fused = rrf_fuse(dense, bm25, k=60, top_k=5)

    assert len(fused) == 5
