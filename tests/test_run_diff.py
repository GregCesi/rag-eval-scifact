"""Tests de la comparaison de deux runs claim par claim (EXE-109).

Couvre les critères 3, 5 et 10 du ticket : les trois compteurs (meilleur,
moins bon, identique), les quatre filtres (perd/gagne le rang 1, sort/entre
dans le top 10), et le cas du claim 70. Aucun modèle chargé, aucun Streamlit
lancé : runs fabriqués à la main pour la partie unitaire, fichiers réels de
results/v2-grid pour la partie de non-régression sur les chiffres du ticket.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval_scifact.compare import load_run
from rag_eval_scifact.run_diff import (
    changed_claims,
    compare_claims,
    filter_claims,
    find_claim,
    rank_comparison_counts,
    unit_badge,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_v2_grid_run(run_name: str) -> dict:
    (path,) = RESULTS_DIR.glob(f"v2-grid/{run_name}-*.json.gz")
    return load_run(path)


def _run(name: str, ranks: dict[str, int | None]) -> dict:
    """Fabrique un run minimal : seul `per_query_metrics.best_rank` compte ici."""
    return {
        "campagne": "campagne-test",
        "run_name": name,
        "queries": [
            {
                "query_id": qid,
                "query_text": f"claim {qid}",
                "expected_docs": [{"doc_id": "d1", "token_count": 10}],
                "retrieved_top100": [],
                "per_query_metrics": {"best_rank": rank},
            }
            for qid, rank in ranks.items()
        ],
    }


def _load_v2_grid_pair() -> tuple[dict, dict]:
    (run_a_path,) = RESULTS_DIR.glob(
        "v2-grid/dense-qwen3-passages-sans-reranker-*.json.gz"
    )
    (run_b_path,) = RESULTS_DIR.glob(
        "v2-grid/dense-qwen3-passages-avec-reranker-*.json.gz"
    )
    return load_run(run_a_path), load_run(run_b_path)


# ─────────────────────────────────────────────
# Critère 3 — compteurs, sur des runs fabriqués
# ─────────────────────────────────────────────
def test_rank_comparison_counts_on_fabricated_runs():
    run_a = _run("A", {"q1": 1, "q2": 5, "q3": None, "q4": 10, "q5": None})
    run_b = _run("B", {"q1": 3, "q2": 2, "q3": None, "q4": 10, "q5": 50})

    rows = compare_claims(run_a, run_b)
    counts = rank_comparison_counts(rows)

    # q1 : 1 -> 3, pire. q2 : 5 -> 2, meilleur. q3 : absent des deux, identique.
    # q4 : 10 -> 10, identique. q5 : absent -> 50, meilleur (un rang trouvé
    # vaut toujours mieux qu'absent du top 100).
    assert counts == {"better": 2, "worse": 1, "same": 2}


# ─────────────────────────────────────────────
# Critère 3 — compteurs, sur le vrai couple du ticket
# ─────────────────────────────────────────────
def test_rank_comparison_counts_on_real_v2_grid_pair():
    run_a, run_b = _load_v2_grid_pair()
    rows = compare_claims(run_a, run_b)
    counts = rank_comparison_counts(rows)

    assert counts == {"better": 51, "worse": 79, "same": 170}


# ─────────────────────────────────────────────
# Critère 5 — quatre filtres, sur des runs fabriqués
# ─────────────────────────────────────────────
def test_filter_claims_on_fabricated_runs():
    run_a = _run(
        "A",
        {
            "perd": 1,  # perd le rang 1
            "gagne": 3,  # gagne le rang 1
            "sort": 7,  # sort du top 10
            "entre": 15,  # entre dans le top 10
            "stable": 20,  # ne bouge pas assez pour matcher un filtre
        },
    )
    run_b = _run(
        "B",
        {
            "perd": 2,
            "gagne": 1,
            "sort": None,
            "entre": 9,
            "stable": 20,
        },
    )
    rows = compare_claims(run_a, run_b)

    assert [r["query_id"] for r in filter_claims(rows, "perd_rang1")] == ["perd"]
    assert [r["query_id"] for r in filter_claims(rows, "gagne_rang1")] == ["gagne"]
    assert [r["query_id"] for r in filter_claims(rows, "sort_top10")] == ["sort"]
    assert [r["query_id"] for r in filter_claims(rows, "entre_top10")] == ["entre"]

    changed = changed_claims(rows)
    assert {r["query_id"] for r in changed} == {"perd", "gagne", "sort", "entre"}


# ─────────────────────────────────────────────
# Critère 5 — quatre filtres, sur le vrai couple du ticket
# ─────────────────────────────────────────────
def test_filter_claims_on_real_v2_grid_pair():
    run_a, run_b = _load_v2_grid_pair()
    rows = compare_claims(run_a, run_b)

    assert len(filter_claims(rows, "perd_rang1")) == 34
    assert len(filter_claims(rows, "gagne_rang1")) == 28
    assert len(filter_claims(rows, "sort_top10")) == 16
    assert len(filter_claims(rows, "entre_top10")) == 12


# ─────────────────────────────────────────────
# Critère 10 — claim 70, sur le vrai couple du ticket
# ─────────────────────────────────────────────
def test_claim_70_on_real_v2_grid_pair():
    run_a, run_b = _load_v2_grid_pair()
    rows = compare_claims(run_a, run_b)

    claim_70 = next(r for r in rows if r["query_id"] == "70")
    assert claim_70["rank_a"] == 1
    assert claim_70["rank_b"] == 21

    query_a = next(q for q in run_a["queries"] if q["query_id"] == "70")
    top2 = [d["doc_id"] for d in query_a["retrieved_top100"][:2]]
    assert top2 == ["5956380", "4414547"]


# ─────────────────────────────────────────────
# Critères 1 et 2 — trouver un claim par son numéro, indépendamment du filtre actif
# ─────────────────────────────────────────────
def test_find_claim_by_number_ignores_active_filter():
    run_a = _load_v2_grid_run("dense-qwen3-256-sans-reranker")
    run_b = _load_v2_grid_run("dense-qwen3-passages-sans-reranker")
    rows = compare_claims(run_a, run_b)

    # Le filtre « perd le rang 1 » exclut le claim 70 : son rang A est 15, pas 1.
    filtered_qids = {r["query_id"] for r in filter_claims(rows, "perd_rang1")}
    assert "70" not in filtered_qids

    claim = find_claim(run_a, "70")
    assert claim is not None
    assert claim["query_id"] == "70"


def test_find_claim_returns_none_for_unknown_number():
    run_a = _load_v2_grid_run("dense-qwen3-256-sans-reranker")
    assert find_claim(run_a, "999999") is None


# ─────────────────────────────────────────────
# Critère 7 — badge d'unité/fenêtre selon la config du run
# ─────────────────────────────────────────────
def test_unit_badge_document_256():
    retriever = {"name": "dense", "unit": "document", "max_seq_length": 256}
    assert unit_badge(retriever, 423) == "tronqué"
    assert unit_badge(retriever, 200) == "complet"


def test_unit_badge_document_2048():
    retriever = {"name": "dense", "unit": "document", "max_seq_length": 2048}
    assert unit_badge(retriever, 423) == "lu en entier"


def test_unit_badge_passages():
    retriever = {"name": "dense", "unit": "passages", "max_seq_length": 2048}
    assert unit_badge(retriever, 9999) == "découpé en passages"


def test_unit_badge_bm25_document():
    retriever = {"name": "bm25", "unit": "document", "max_seq_length": 256}
    assert unit_badge(retriever, 9999) == "lu en entier"


# ─────────────────────────────────────────────
# Critère 9 — claim 70, paire 256 / abstract-entier, valeurs du ticket
# ─────────────────────────────────────────────
def test_claim_70_badges_on_256_vs_abstract_entier_pair():
    run_a = _load_v2_grid_run("dense-qwen3-256-sans-reranker")
    run_b = _load_v2_grid_run("dense-qwen3-abstract-entier-sans-reranker")

    expected_a = {
        d["doc_id"]: d["token_count"] for d in find_claim(run_a, "70")["expected_docs"]
    }
    expected_b = {
        d["doc_id"]: d["token_count"] for d in find_claim(run_b, "70")["expected_docs"]
    }

    assert expected_a["4414547"] == 423
    assert expected_a["5956380"] == 272
    assert expected_b["4414547"] == 423

    retriever_a = run_a["config"]["retriever"]
    retriever_b = run_b["config"]["retriever"]
    assert unit_badge(retriever_a, expected_a["4414547"]) == "tronqué"
    assert unit_badge(retriever_b, expected_b["4414547"]) == "lu en entier"
