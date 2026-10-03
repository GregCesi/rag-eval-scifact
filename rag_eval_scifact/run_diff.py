"""Compare deux runs de campagne claim par claim (EXE-109).

Aucune métrique n'est recalculée : le meilleur rang par claim est repris de
`per_query_metrics.best_rank`, déjà écrit dans le format run
(`.claude/rules/versioning.md`). Un document attendu absent du top 100
(`best_rank` None) compte comme moins bon que n'importe quel rang trouvé.
"""

from __future__ import annotations

from pathlib import Path

NOT_FOUND_RANK = 101  # sentinelle de tri/affichage : au-delà du top 100


def list_campaign_dirs(results_dir: Path) -> list[str]:
    """Sous-dossiers de `results_dir` portant au moins un run de campagne (*.json.gz)."""
    if not results_dir.exists():
        return []
    return sorted(
        d.name for d in results_dir.iterdir() if d.is_dir() and any(d.glob("*.json.gz"))
    )


def list_campaign_run_files(results_dir: Path, campaign: str) -> list[Path]:
    """Fichiers de run d'une campagne."""
    return sorted((results_dir / campaign).glob("*.json.gz"))


def _sortable_rank(rank: int | None) -> int:
    return rank if rank is not None else NOT_FOUND_RANK


def compare_claims(run_a: dict, run_b: dict) -> list[dict]:
    """Rang du meilleur document attendu, claim par claim, pour A et B."""
    queries_b = {q["query_id"]: q for q in run_b["queries"]}
    rows = []
    for query_a in run_a["queries"]:
        qid = query_a["query_id"]
        query_b = queries_b[qid]
        rows.append(
            {
                "query_id": qid,
                "query_text": query_a["query_text"],
                "rank_a": query_a["per_query_metrics"]["best_rank"],
                "rank_b": query_b["per_query_metrics"]["best_rank"],
            }
        )
    return rows


def changed_claims(rows: list[dict]) -> list[dict]:
    """Claims dont le meilleur rang diffère entre A et B."""
    return [row for row in rows if row["rank_a"] != row["rank_b"]]


def rank_comparison_counts(rows: list[dict]) -> dict[str, int]:
    """Nombre de claims où B est meilleur, moins bon, ou identique à A."""
    better = worse = same = 0
    for row in rows:
        rank_a, rank_b = _sortable_rank(row["rank_a"]), _sortable_rank(row["rank_b"])
        if rank_b < rank_a:
            better += 1
        elif rank_b > rank_a:
            worse += 1
        else:
            same += 1
    return {"better": better, "worse": worse, "same": same}


FILTER_LABELS = {
    "perd_rang1": "Perd le rang 1",
    "gagne_rang1": "Gagne le rang 1",
    "sort_top10": "Sort du top 10",
    "entre_top10": "Entre dans le top 10",
}


def filter_claims(rows: list[dict], filter_key: str) -> list[dict]:
    """Applique l'un des quatre filtres nommés de `FILTER_LABELS` à `rows`."""
    if filter_key == "perd_rang1":
        return [r for r in rows if r["rank_a"] == 1 and r["rank_b"] != 1]
    if filter_key == "gagne_rang1":
        return [r for r in rows if r["rank_b"] == 1 and r["rank_a"] != 1]
    if filter_key == "sort_top10":
        return [
            r
            for r in rows
            if r["rank_a"] is not None
            and r["rank_a"] <= 10
            and (r["rank_b"] is None or r["rank_b"] > 10)
        ]
    if filter_key == "entre_top10":
        return [
            r
            for r in rows
            if r["rank_b"] is not None
            and r["rank_b"] <= 10
            and (r["rank_a"] is None or r["rank_a"] > 10)
        ]
    raise ValueError(f"filtre inconnu : {filter_key}")
