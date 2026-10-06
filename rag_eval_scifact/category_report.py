"""Rapport de v2-grid par catégorie d'étiquette d'origine (EXE-124).

Un document attendu BEIR n'est pas forcément une preuve : il peut confirmer le
claim, le contredire, ou simplement être cité sans trancher. Ce module relit
les 34 runs déjà commités de `results/v2-grid/` et le fichier des étiquettes
d'origine (`rag_eval_scifact.origin_labels`) pour lire les métriques
séparément selon cette catégorie, sans jamais relancer un run.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_scifact import report as report_module
from rag_eval_scifact.metrics import ndcg_at_k
from rag_eval_scifact.report import load_campaign_runs
from rag_eval_scifact.stats import paired_permutation_test

CAMPAGNE = "v2-grid"
V2_GRID_DIR = Path("results") / CAMPAGNE
ORIGIN_LABELS_PATH = Path("results/etiquettes-origine.json")

CATEGORY_NAMES = ("confirme", "contredit", "sans_preuve")


class NoRunsFound(Exception):
    """Levée par `write_report_for_campaign` quand la campagne nommée n'a
    aucun run (EXE-142, critère 6)."""


# Graine et nombre de permutations fixes (même convention que `report.py`,
# EXE-91 H3) : un rapport regénéré sur les mêmes fichiers rend le même texte.
SEED = 0
N_PERMUTATIONS = 10000


def load_origin_labels(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _label_docs_by_claim(origin_labels: dict) -> dict[str, dict[str, set[str]]]:
    """Pour chaque claim, ses documents cités regroupés par étiquette d'origine."""
    by_claim: dict[str, dict[str, set[str]]] = {}
    for pair in origin_labels["pairs"]:
        qid = pair["query_id"]
        by_claim.setdefault(
            qid, {"SUPPORT": set(), "CONTRADICT": set(), "SANS_PREUVE": set()}
        )
        by_claim[qid][pair["label"]].add(pair["doc_id"])
    return by_claim


def claims_by_categorie(origin_labels: dict) -> dict[str, list[str]]:
    """Claims groupés par catégorie (critère 3), triés numériquement."""
    result: dict[str, list[str]] = {name: [] for name in CATEGORY_NAMES}
    for claim in origin_labels["claims"]:
        result[claim["categorie"]].append(claim["query_id"])
    for qids in result.values():
        qids.sort(key=int)
    return result


def expected_docs_for_rank_metrics(
    origin_labels: dict,
) -> dict[str, dict[str, set[str]]]:
    """Critère 5 : documents attendus par catégorie, pour rang1/top10/top100.

    confirme → ses documents SUPPORT seulement ; contredit → ses documents
    CONTRADICT seulement ; sans_preuve → tous ses documents cités (ils sont
    tous SANS_PREUVE, par construction de la catégorie).
    """
    label_docs = _label_docs_by_claim(origin_labels)
    by_cat = claims_by_categorie(origin_labels)
    return {
        "confirme": {qid: label_docs[qid]["SUPPORT"] for qid in by_cat["confirme"]},
        "contredit": {
            qid: label_docs[qid]["CONTRADICT"] for qid in by_cat["contredit"]
        },
        "sans_preuve": {
            qid: label_docs[qid]["SANS_PREUVE"] for qid in by_cat["sans_preuve"]
        },
    }


def expected_docs_for_ndcg_sets(origin_labels: dict) -> dict[str, dict[str, set[str]]]:
    """Critère 6 : les trois ensembles de claims, documents attendus pour nDCG@10.

    `300` : tous les claims, documents attendus = tous les documents cités.
    `188_avec_preuve` : claims confirme + contredit, documents attendus =
    leurs documents SUPPORT ou CONTRADICT seulement.
    `112_sans_preuve` : claims sans_preuve, documents attendus = cités (= le
    même ensemble que pour `expected_docs_for_rank_metrics`).
    """
    label_docs = _label_docs_by_claim(origin_labels)
    by_cat = claims_by_categorie(origin_labels)
    all_cited = {
        qid: docs["SUPPORT"] | docs["CONTRADICT"] | docs["SANS_PREUVE"]
        for qid, docs in label_docs.items()
    }
    avec_preuve_qids = by_cat["confirme"] + by_cat["contredit"]
    return {
        "300": all_cited,
        "188_avec_preuve": {
            qid: label_docs[qid]["SUPPORT"] | label_docs[qid]["CONTRADICT"]
            for qid in avec_preuve_qids
        },
        "112_sans_preuve": {qid: all_cited[qid] for qid in by_cat["sans_preuve"]},
    }


def _best_rank(retrieved_ids: list[str], expected_ids: set[str]) -> int | None:
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            return i
    return None


def category_rank_rates(
    run_data: dict, expected_by_category: dict[str, dict[str, set[str]]]
) -> dict[str, dict[str, float]]:
    """Critère 5 : pour un run, n/rang1/top10/top100 par catégorie."""
    queries_by_id = {q["query_id"]: q for q in run_data["queries"]}
    result: dict[str, dict[str, float]] = {}
    for categorie, expected_by_qid in expected_by_category.items():
        n = len(expected_by_qid)
        rank1 = top10 = top100 = 0
        for qid, expected_ids in expected_by_qid.items():
            retrieved_ids = [
                d["doc_id"] for d in queries_by_id[qid]["retrieved_top100"]
            ]
            rank = _best_rank(retrieved_ids, expected_ids)
            if rank == 1:
                rank1 += 1
            if rank is not None and rank <= 10:
                top10 += 1
            if rank is not None and rank <= 100:
                top100 += 1
        result[categorie] = {
            "n": n,
            "rang1": rank1 / n if n else 0.0,
            "top10": top10 / n if n else 0.0,
            "top100": top100 / n if n else 0.0,
        }
    return result


def ndcg_per_query_values(
    run_data: dict, expected_by_qid: dict[str, set[str]]
) -> dict[str, float]:
    """nDCG@10 par requête, restreint à `expected_by_qid` (son propre ensemble
    de documents attendus, pas celui du run) — réutilise `metrics.ndcg_at_k`
    sans en changer le calcul, comme `compare.per_query_metric_values`."""
    queries_by_id = {q["query_id"]: q for q in run_data["queries"]}
    values = {}
    for qid, expected_ids in expected_by_qid.items():
        retrieved_ids = [d["doc_id"] for d in queries_by_id[qid]["retrieved_top100"]]
        values[qid] = ndcg_at_k(retrieved_ids, expected_ids, 10)
    return values


def ndcg_set_values(
    run_data: dict, expected_by_set: dict[str, dict[str, set[str]]]
) -> dict[str, float]:
    """Critère 6 : nDCG@10 moyen pour chacun des trois ensembles de claims."""
    result = {}
    for set_name, expected_by_qid in expected_by_set.items():
        values = ndcg_per_query_values(run_data, expected_by_qid)
        result[set_name] = sum(values.values()) / len(values) if values else 0.0
    return result


def base_strategy_name(run_name: str) -> tuple[str, str] | None:
    """Scinde un nom de run en (stratégie de base, variante avec/sans reranker).

    `None` si `run_name` ne porte aucun des deux suffixes (H4, EXE-124 : une
    stratégie de base est un nom de run sans son suffixe -sans-reranker ou
    -avec-reranker)."""
    for suffix, variant in (("-avec-reranker", "avec"), ("-sans-reranker", "sans")):
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)], variant
    return None


def group_runs_by_base_strategy(runs: list[dict]) -> dict[str, dict[str, dict]]:
    """Regroupe les runs par stratégie de base, avec leur variante avec/sans reranker."""
    grouped: dict[str, dict[str, dict]] = {}
    for run_data in runs:
        split = base_strategy_name(run_data["run_name"])
        if split is None:
            continue
        base, variant = split
        grouped.setdefault(base, {})[variant] = run_data
    return grouped


def reranker_effect_rows(
    runs: list[dict],
    expected_188: dict[str, set[str]],
    expected_112: dict[str, set[str]],
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> list[dict]:
    """Critère 8 : écart de nDCG@10 (avec − sans), par stratégie de base, sur
    les claims avec preuve et sans preuve, avec la p-value du test de
    randomisation apparié déjà codé à la main (`rag_eval_scifact.stats`)."""
    grouped = group_runs_by_base_strategy(runs)
    rows = []
    for base in sorted(grouped):
        variants = grouped[base]
        if "avec" not in variants or "sans" not in variants:
            continue
        sans_run, avec_run = variants["sans"], variants["avec"]

        qids_188 = sorted(expected_188)
        values_sans_188 = ndcg_per_query_values(sans_run, expected_188)
        values_avec_188 = ndcg_per_query_values(avec_run, expected_188)
        cmp_188 = paired_permutation_test(
            [values_sans_188[q] for q in qids_188],
            [values_avec_188[q] for q in qids_188],
            n_permutations=n_permutations,
            seed=seed,
        )

        qids_112 = sorted(expected_112)
        values_sans_112 = ndcg_per_query_values(sans_run, expected_112)
        values_avec_112 = ndcg_per_query_values(avec_run, expected_112)
        cmp_112 = paired_permutation_test(
            [values_sans_112[q] for q in qids_112],
            [values_avec_112[q] for q in qids_112],
            n_permutations=n_permutations,
            seed=seed,
        )

        rows.append(
            {
                "base": base,
                "diff_188": cmp_188["mean_diff"],
                "p_188": cmp_188["p_value"],
                "diff_112": cmp_112["mean_diff"],
                "p_112": cmp_112["p_value"],
            }
        )
    return rows


CATEGORY_TITLES = {
    "confirme": "confirme",
    "contredit": "contredit",
    "sans_preuve": "sans preuve",
}


def _render_rank_sections(rank_rows: dict[str, list[dict]]) -> list[str]:
    lines: list[str] = []
    for categorie in CATEGORY_NAMES:
        lines.append(f"## {CATEGORY_TITLES[categorie]}\n\n")
        lines.append("| run | n | rang 1 | top 10 | top 100 |\n")
        lines.append("|---|---|---|---|---|\n")
        for row in rank_rows[categorie]:
            lines.append(
                f"| {row['run_name']} | {row['n']} "
                f"| {row['rang1']:.4f} | {row['top10']:.4f} | {row['top100']:.4f} |\n"
            )
        lines.append("\n")
    return lines


def _render_ndcg_section(ndcg_rows: list[dict]) -> list[str]:
    lines: list[str] = [
        "## nDCG@10 par ensemble de claims\n\n",
        "| run | 300 claims | 188 claims avec preuve | 112 claims sans preuve |\n",
        "|---|---|---|---|\n",
    ]
    for row in ndcg_rows:
        lines.append(
            f"| {row['run_name']} | {row['300']:.4f} "
            f"| {row['188_avec_preuve']:.4f} | {row['112_sans_preuve']:.4f} |\n"
        )
    lines.append("\n")
    return lines


def _render_reranker_section(reranker_rows: list[dict]) -> list[str]:
    lines: list[str] = [
        "## Effet du reranker (avec − sans), par stratégie de base\n\n",
        (
            "| stratégie | Δ nDCG@10 (188 avec preuve) | p (188) "
            "| Δ nDCG@10 (112 sans preuve) | p (112) |\n"
        ),
        "|---|---|---|---|---|\n",
    ]
    for row in reranker_rows:
        lines.append(
            f"| {row['base']} | {row['diff_188']:+.4f} | {row['p_188']:.4f} "
            f"| {row['diff_112']:+.4f} | {row['p_112']:.4f} |\n"
        )
    return lines


def render_category_report(
    rank_rows: dict[str, list[dict]],
    ndcg_rows: list[dict],
    reranker_rows: list[dict],
) -> str:
    """Rend le rapport markdown. Aucun mot hors noms de run/stratégie et chiffres."""
    lines: list[str] = []
    lines += _render_rank_sections(rank_rows)
    lines += _render_ndcg_section(ndcg_rows)
    lines += _render_reranker_section(reranker_rows)
    return "".join(lines)


def write_report() -> Path:
    """Écrit `results/v2-grid/RAPPORT-PAR-CATEGORIE.md` et renvoie son chemin.

    Chemin historique (EXE-124), inchangé par EXE-142 — critère 2 : une
    commande sans campagne nommée régénère ce rapport à l'identique."""
    origin_labels = load_origin_labels(ORIGIN_LABELS_PATH)
    runs = load_campaign_runs(CAMPAGNE)

    expected_rank = expected_docs_for_rank_metrics(origin_labels)
    expected_ndcg = expected_docs_for_ndcg_sets(origin_labels)

    rank_rows = {
        categorie: [
            {
                "run_name": run["run_name"],
                **category_rank_rates(run, expected_rank)[categorie],
            }
            for run in runs
        ]
        for categorie in CATEGORY_NAMES
    }
    ndcg_rows = [
        {"run_name": run["run_name"], **ndcg_set_values(run, expected_ndcg)}
        for run in runs
    ]
    reranker_rows = reranker_effect_rows(
        runs, expected_ndcg["188_avec_preuve"], expected_ndcg["112_sans_preuve"]
    )

    content = render_category_report(rank_rows, ndcg_rows, reranker_rows)

    report_path = V2_GRID_DIR / "RAPPORT-PAR-CATEGORIE.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path


def reference_diff_rows(
    runs: list[dict],
    reference_run: str,
    expected_188: dict[str, set[str]],
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> list[dict]:
    """EXE-142, critère 3 : pour chaque run autre que `reference_run`, écart
    de nDCG@10 (ce run − référence) sur les 188 claims avec preuve, et la
    p-value du test de randomisation apparié déjà codé à la main."""
    runs_by_name = {r["run_name"]: r for r in runs}
    reference = runs_by_name[reference_run]
    qids = sorted(expected_188)
    reference_values = ndcg_per_query_values(reference, expected_188)

    rows = []
    for run_data in runs:
        if run_data["run_name"] == reference_run:
            continue
        values = ndcg_per_query_values(run_data, expected_188)
        cmp = paired_permutation_test(
            [reference_values[q] for q in qids],
            [values[q] for q in qids],
            n_permutations=n_permutations,
            seed=seed,
        )
        rows.append(
            {
                "run_name": run_data["run_name"],
                "diff_188": cmp["mean_diff"],
                "p_188": cmp["p_value"],
            }
        )
    return rows


def _render_reference_section(reference_run: str, rows: list[dict]) -> list[str]:
    lines: list[str] = [
        (
            f"## Écart de nDCG@10 face à la référence {reference_run} "
            "(188 claims avec preuve)\n\n"
        ),
        "| run | Δ nDCG@10 | p |\n",
        "|---|---|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['diff_188']:+.4f} | {row['p_188']:.4f} |\n"
        )
    lines.append("\n")
    return lines


def group_rank_rows(runs: list[dict]) -> list[dict]:
    """EXE-142, critère 4 : pour chaque run, le `found_counts` de
    `reading_levels` aux seuils 5 et 10 premiers, par groupe de la page
    « Niveaux de lecture ». Import différé : `reading_levels` importe déjà
    `category_report.group_runs_by_base_strategy`."""
    from rag_eval_scifact import reading_levels

    goldens = reading_levels.load_goldens(
        reading_levels.ORIGIN_LABELS_PATH, reading_levels.JUGEMENTS_PATH
    )
    threshold_5 = reading_levels.THRESHOLD_RANKS["5 premiers"]
    threshold_10 = reading_levels.THRESHOLD_RANKS["10 premiers"]
    return [
        {
            "run_name": run["run_name"],
            "counts_5": reading_levels.found_counts(goldens, run, threshold_5),
            "counts_10": reading_levels.found_counts(goldens, run, threshold_10),
        }
        for run in runs
    ]


def _render_group_section(group_rows: list[dict]) -> list[str]:
    from rag_eval_scifact import reading_levels

    lines: list[str] = ["## Niveaux de lecture par groupe\n\n"]
    for row in group_rows:
        lines.append(f"### {row['run_name']}\n\n")
        lines.append(
            "| groupe | 5 premiers | part (5 premiers) "
            "| 10 premiers | part (10 premiers) |\n"
        )
        lines.append("|---|---|---|---|---|\n")
        for name in reading_levels.GROUP_NAMES:
            c5, c10 = row["counts_5"][name], row["counts_10"][name]
            rate5 = c5["found"] / c5["total"] if c5["total"] else 0.0
            rate10 = c10["found"] / c10["total"] if c10["total"] else 0.0
            lines.append(
                f"| {reading_levels.GROUP_LABELS[name]} "
                f"| {c5['found']}/{c5['total']} | {rate5:.4f} "
                f"| {c10['found']}/{c10['total']} | {rate10:.4f} |\n"
            )
        lines.append("\n")
    return lines


def render_category_report_for_campaign(
    campagne: str,
    rank_rows: dict[str, list[dict]],
    ndcg_rows: list[dict],
    reranker_rows: list[dict] | None,
    reference_run: str | None,
    reference_rows: list[dict] | None,
    group_rows: list[dict],
) -> str:
    """EXE-142 : rapport par catégorie d'une campagne nommée, avec les
    niveaux de lecture par groupe (critère 4), l'écart face à un run de
    référence s'il est nommé (critère 3), et la section reranker omise
    si aucun run de la campagne ne la porte (critère 5)."""
    lines: list[str] = [f"Campagne : {campagne}\n\n"]
    lines += _render_rank_sections(rank_rows)
    lines += _render_ndcg_section(ndcg_rows)
    if reference_run is not None:
        lines += _render_reference_section(reference_run, reference_rows or [])
    if reranker_rows is not None:
        lines += _render_reranker_section(reranker_rows)
        lines.append("\n")
    lines += _render_group_section(group_rows)
    return "".join(lines)


def write_report_for_campaign(campagne: str, reference_run: str | None = None) -> Path:
    """EXE-142 : écrit `results/<campagne>/RAPPORT-PAR-CATEGORIE.md` pour une
    campagne nommée (critère 1). Lève `NoRunsFound` sans rien écrire si la
    campagne n'a aucun run (critère 6)."""
    runs = load_campaign_runs(campagne)
    if not runs:
        raise NoRunsFound(campagne)

    origin_labels = load_origin_labels(ORIGIN_LABELS_PATH)
    expected_rank = expected_docs_for_rank_metrics(origin_labels)
    expected_ndcg = expected_docs_for_ndcg_sets(origin_labels)

    rank_rows = {
        categorie: [
            {
                "run_name": run["run_name"],
                **category_rank_rates(run, expected_rank)[categorie],
            }
            for run in runs
        ]
        for categorie in CATEGORY_NAMES
    }
    ndcg_rows = [
        {"run_name": run["run_name"], **ndcg_set_values(run, expected_ndcg)}
        for run in runs
    ]

    reference_rows = None
    if reference_run is not None:
        reference_rows = reference_diff_rows(
            runs, reference_run, expected_ndcg["188_avec_preuve"]
        )

    has_reranker = any(r["run_name"].endswith("-avec-reranker") for r in runs)
    reranker_rows = (
        reranker_effect_rows(
            runs, expected_ndcg["188_avec_preuve"], expected_ndcg["112_sans_preuve"]
        )
        if has_reranker
        else None
    )

    group_rows = group_rank_rows(runs)

    content = render_category_report_for_campaign(
        campagne,
        rank_rows,
        ndcg_rows,
        reranker_rows,
        reference_run,
        reference_rows,
        group_rows,
    )

    report_path = report_module.RESULTS_DIR / campagne / "RAPPORT-PAR-CATEGORIE.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
