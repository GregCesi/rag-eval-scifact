"""Rapport d'une campagne : tableau des runs face à la référence v1-rejeu/baseline.

Ne charge aucun modèle d'embedding : relit les JSON déjà écrits par les runs de
campagne (`results/<campagne>/*.json.gz`) et les compare au run de référence
par le test de randomisation apparié de `stats.py`, déjà codé à la main
(`.claude/rules/methodologie.md` proscrit toute lib de stats). Le rapport ne
contient que des chiffres et des noms de runs, jamais de phrase d'analyse.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval_scifact.compare import compare_runs, load_run

RESULTS_DIR = Path("results")
BASELINE_GLOB = "v1-rejeu/baseline-*.json*"

SIX_METRICS = ("recall@1", "recall@5", "recall@10", "recall@100", "ndcg@10", "mrr")
BUCKET_NAMES = ("perfect", "near_miss", "deep_miss", "miss_100")

# Graine et nombre de permutations fixes (H3, EXE-91) : un rapport regénéré sur
# les mêmes fichiers de résultats rend le même texte, au caractère près.
SEED = 0
N_PERMUTATIONS = 10000


def find_baseline_path() -> Path:
    """Chemin du run de référence `results/v1-rejeu/baseline-*.json(.gz)`."""
    matches = sorted(RESULTS_DIR.glob(BASELINE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"aucune référence trouvée sous {RESULTS_DIR / BASELINE_GLOB}"
        )
    return matches[-1]


class InvalidRunFile(Exception):
    """Levée par `load_campaign_runs` quand un `.json.gz` n'a pas la forme
    d'un run (EXE-156) : ce format ne devrait jamais porter autre chose
    qu'un run commité, à la différence d'un `.json` nu (ex. `hyde.json`)."""


def _looks_like_run(data: object) -> bool:
    return isinstance(data, dict) and "run_name" in data and "queries" in data


def load_campaign_runs(campagne: str) -> list[dict]:
    """Charge tous les runs de `campagne`, triés par nom de fichier (ordre stable).

    Un `.json` sans forme de run est ignoré en silence (ex. `hyde.json`, une
    liste de faux résumés) : ce format sert aussi à des sous-produits qui ne
    sont pas des runs. Un `.json.gz` sans forme de run lève `InvalidRunFile`
    en nommant le fichier (EXE-156) : ce format est toujours un run commité.
    """
    campaign_dir = RESULTS_DIR / campagne
    runs_by_path: dict[Path, dict] = {}

    for path in sorted(campaign_dir.glob("*.json.gz")):
        data = load_run(path)
        if not _looks_like_run(data):
            raise InvalidRunFile(f"fichier de run invalide : {path}")
        runs_by_path[path] = data

    for path in sorted(campaign_dir.glob("*.json")):
        data = load_run(path)
        if _looks_like_run(data):
            runs_by_path[path] = data

    return [runs_by_path[p] for p in sorted(runs_by_path, key=lambda p: p.name)]


def _bucket_values(run_data: dict) -> dict[str, float]:
    extended = run_data.get("extended_metrics", {})
    return {
        name: extended.get(f"bucket_found_at_10_{name}", 0.0) for name in BUCKET_NAMES
    }


def build_report_rows(runs: list[dict], baseline: dict) -> list[dict]:
    """Une ligne par run : ses 6 métriques, son écart et sa p-value face à `baseline`."""
    rows = []
    for run_data in runs:
        metrics = run_data["metrics"]
        extended = run_data.get("extended_metrics", {})
        ndcg_cmp = compare_runs(
            baseline, run_data, "ndcg@10", n_permutations=N_PERMUTATIONS, seed=SEED
        )
        mrr_cmp = compare_runs(
            baseline, run_data, "mrr", n_permutations=N_PERMUTATIONS, seed=SEED
        )
        rows.append(
            {
                "run_name": run_data["run_name"],
                "metrics": metrics,
                "ndcg_diff": ndcg_cmp["mean_diff"],
                "ndcg_p_value": ndcg_cmp["p_value"],
                "mrr_diff": mrr_cmp["mean_diff"],
                "mrr_p_value": mrr_cmp["p_value"],
                "buckets": _bucket_values(run_data),
                "truncated_pct": extended.get("truncated_pct", 0.0),
                "avg_retrieval_latency_ms": extended.get(
                    "avg_retrieval_latency_ms", 0.0
                ),
                "indexing_duration_seconds": extended.get(
                    "indexing_duration_seconds", 0.0
                ),
            }
        )
    rows.sort(key=lambda r: (-r["metrics"]["ndcg@10"], r["run_name"]))
    return rows


def render_report(rows: list[dict], baseline_run_name: str) -> str:
    """Rend le tableau markdown. Aucun mot hors noms de run, unités et libellés."""
    lines = [
        f"Référence : {baseline_run_name}\n",
        "\n",
        (
            "| run | nDCG@10 | MRR | R@1 | R@5 | R@10 | R@100 "
            "| Δ nDCG@10 | p(nDCG@10) | Δ MRR | p(MRR) "
            "| perfect | near_miss | deep_miss | miss_100 "
            "| tronqué % | latence ms | indexation s |\n"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        m = r["metrics"]
        b = r["buckets"]
        lines.append(
            f"| {r['run_name']} "
            f"| {m['ndcg@10']:.4f} | {m['mrr']:.4f} "
            f"| {m['recall@1']:.4f} | {m['recall@5']:.4f} "
            f"| {m['recall@10']:.4f} | {m['recall@100']:.4f} "
            f"| {r['ndcg_diff']:+.4f} | {r['ndcg_p_value']:.4f} "
            f"| {r['mrr_diff']:+.4f} | {r['mrr_p_value']:.4f} "
            f"| {b['perfect']:.4f} | {b['near_miss']:.4f} "
            f"| {b['deep_miss']:.4f} | {b['miss_100']:.4f} "
            f"| {r['truncated_pct']:.1f} | {r['avg_retrieval_latency_ms']:.1f} "
            f"| {r['indexing_duration_seconds']:.1f} |\n"
        )
    return "".join(lines)


def write_report(campagne: str) -> Path:
    """Écrit `results/<campagne>/RAPPORT.md` et renvoie son chemin."""
    baseline = load_run(find_baseline_path())
    runs = load_campaign_runs(campagne)
    rows = build_report_rows(runs, baseline)
    content = render_report(rows, baseline["run_name"])

    report_path = RESULTS_DIR / campagne / "RAPPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
