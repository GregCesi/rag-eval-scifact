"""Rapport comparant les juges de v3-juge entre eux et aux étiquettes d'error
analysis v1 (EXE-122).

Ne lance aucun jugement : relit les jugements déjà écrits par `run_judge.py`
(`results/<campagne>/jugements-<juge>.json`) et les paires déjà écrites par
`run_judge_pairs.py` (`results/<campagne>/paires.json`). Fonctionne avec un,
deux ou trois juges présents ; dit lesquels manquent. Le rapport ne contient
que des chiffres et des noms, jamais de phrase d'analyse — comme `report.py`
pour la campagne v2-grid, qui ne change pas.
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

from rag_eval_scifact.judge_pairs import load_pairs
from rag_eval_scifact.stats import cohen_kappa

RESULTS_DIR = Path("results")
JUDGE_NAMES = ("local", "claude", "etapes")
RESPONDING_VERDICTS = ("SUPPORTS", "REFUTES")
VALID_VERDICTS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
CAUSES = ("hors sujet", "vocabulaire", "inférence", "sujet voisin")
ORIGIN_LABELS = ("SUPPORT", "CONTRADICT", "SANS_PREUVE")
VERDICT_AGREES_WITH_LABEL = {
    "SUPPORTS": "SUPPORT",
    "REFUTES": "CONTRADICT",
    "NOT_ENOUGH_INFO": "SANS_PREUVE",
}
LEVELS = ("DIRECT", "VOCABULARY", "REASONING")
ANNOTATION_FILES = (
    Path("results/v1-annotations.json"),
    Path("results/v1-deep-miss-annotations.json"),
)
BINARY_LABELS = {
    "SUPPORTS": "répond",
    "REFUTES": "répond",
    "NOT_ENOUGH_INFO": "ne répond pas",
}


def _pairs_path(campagne: str) -> Path:
    return RESULTS_DIR / campagne / "paires.json"


def _judgments_path(campagne: str, juge: str) -> Path:
    return RESULTS_DIR / campagne / f"jugements-{juge}.json"


def load_judgments(campagne: str, juge: str) -> list[dict] | None:
    """`None` quand le fichier de jugements de ce juge n'existe pas (critère 1)."""
    path = _judgments_path(campagne, juge)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _responds(verdict: str) -> bool:
    return verdict in RESPONDING_VERDICTS


def judge_summary(judgments: list[dict]) -> dict:
    """Critère 2 : effectifs de verdicts, citations introuvables, refus du
    juge, durée moyenne.

    EXE-131, critères 7 et 8 : « citation introuvable » est un indicateur à
    part sur un jugement qui garde son verdict réel, pas un verdict en soi ;
    « refus du juge » reste un verdict (aucune réponse du modèle à compter)."""
    counts = Counter(j["verdict"] for j in judgments)
    durations = [j["duration_seconds"] for j in judgments]
    return {
        "n_judged": len(judgments),
        "supports": counts.get("SUPPORTS", 0),
        "refutes": counts.get("REFUTES", 0),
        "not_enough_info": counts.get("NOT_ENOUGH_INFO", 0),
        "illisible": counts.get("illisible", 0),
        "citation_introuvable": sum(
            1 for j in judgments if j.get("citation_introuvable")
        ),
        "refus_du_juge": counts.get("refus du juge", 0),
        "avg_duration_seconds": statistics.mean(durations) if durations else 0.0,
    }


def dataset_summary(judgments: list[dict], pairs_by_id: dict[str, dict]) -> dict:
    """Critère 3 : ce que le juge dit du jeu de données (documents attendus et
    intrus classés devant), par claim et par paire."""
    expected_total = expected_responds = 0
    non_expected_total = non_expected_responds = 0
    claims: set[str] = set()
    claims_with_expected_responds: set[str] = set()
    claims_with_non_expected_responds: set[str] = set()

    for j in judgments:
        pair = pairs_by_id[j["pair_id"]]
        claims.add(j["claim_id"])
        responds = _responds(j["verdict"])
        if pair["document_attendu"]:
            expected_total += 1
            if responds:
                expected_responds += 1
                claims_with_expected_responds.add(j["claim_id"])
        else:
            non_expected_total += 1
            if responds:
                non_expected_responds += 1
                claims_with_non_expected_responds.add(j["claim_id"])

    return {
        "expected_responds_rate": (
            expected_responds / expected_total if expected_total else 0.0
        ),
        "non_expected_responds_rate": (
            non_expected_responds / non_expected_total if non_expected_total else 0.0
        ),
        "n_claims_with_non_expected_responds": len(claims_with_non_expected_responds),
        "n_claims_without_expected_responds": len(
            claims - claims_with_expected_responds
        ),
    }


def cause_summary(judgments: list[dict]) -> dict[str, int]:
    """Critère 6 : jugements du juge par étapes, par cause (seulement quand le
    verdict vaut NOT_ENOUGH_INFO ; les autres jugements n'ont pas de cause)."""
    counts = Counter(j["cause"] for j in judgments if j.get("cause"))
    return {cause: counts.get(cause, 0) for cause in CAUSES}


def expected_crosstab(judgments: list[dict], pairs_by_id: dict) -> dict:
    """EXE-125, critère 8 : tableau croisé 3 × 3 verdict / étiquette d'origine
    sur les documents attendus, et part d'accord. Restreint aux verdicts
    valides (« illisible » et « citation introuvable » ne sont pas un verdict
    sur l'étiquette)."""
    confusion: Counter[tuple[str, str]] = Counter()
    n_valid = 0
    n_agree = 0
    for j in judgments:
        pair = pairs_by_id[j["pair_id"]]
        if not pair["document_attendu"] or j["verdict"] not in VALID_VERDICTS:
            continue
        label = pair["etiquette_origine"]
        confusion[(j["verdict"], label)] += 1
        n_valid += 1
        if VERDICT_AGREES_WITH_LABEL[j["verdict"]] == label:
            n_agree += 1
    return {
        "confusion": confusion,
        "n_valid": n_valid,
        "agreement": n_agree / n_valid if n_valid else 0.0,
    }


def refused_pairs(judgments: list[dict], pairs_by_id: dict) -> list[dict]:
    """EXE-131, critère 9 : paires où ce juge a essuyé un refus du modèle,
    avec leur famille et leur étiquette d'origine."""
    rows = []
    for j in judgments:
        if j["verdict"] != "refus du juge":
            continue
        pair = pairs_by_id[j["pair_id"]]
        rows.append(
            {
                "pair_id": j["pair_id"],
                "famille": pair["famille"],
                "etiquette_origine": pair["etiquette_origine"],
            }
        )
    return rows


def devant_summary(judgments: list[dict], pairs_by_id: dict) -> dict:
    """EXE-125, critère 9 : effectifs de verdicts sur les paires « devant », et
    les claims dont au moins un document classé devant répond au claim."""
    counts: Counter[str] = Counter()
    claims_with_responding_devant: set[str] = set()
    for j in judgments:
        pair = pairs_by_id[j["pair_id"]]
        if "devant" not in pair["famille"]:
            continue
        if j["verdict"] in VALID_VERDICTS:
            counts[j["verdict"]] += 1
        if _responds(j["verdict"]):
            claims_with_responding_devant.add(j["claim_id"])
    return {
        "supports": counts.get("SUPPORTS", 0),
        "refutes": counts.get("REFUTES", 0),
        "not_enough_info": counts.get("NOT_ENOUGH_INFO", 0),
        "claims": sorted(claims_with_responding_devant, key=int),
    }


def level_distribution(
    judgments: list[dict], pairs_by_id: dict, famille: str
) -> dict[str, int] | None:
    """EXE-125, critère 10 : répartition DIRECT/VOCABULARY/REASONING des
    documents de la famille donnée (« attendu », restreint à SUPPORT/CONTRADICT,
    ou « devant ») que le juge juge répondre au claim. `None` si aucun jugement
    de `judgments` ne porte de niveau (juge par étapes, EXE-121 — sans niveau)."""
    if not any("level" in j for j in judgments):
        return None
    counts: Counter[str] = Counter()
    for j in judgments:
        if "level" not in j or not _responds(j["verdict"]):
            continue
        pair = pairs_by_id[j["pair_id"]]
        if famille not in pair["famille"]:
            continue
        if famille == "attendu" and pair["etiquette_origine"] not in (
            "SUPPORT",
            "CONTRADICT",
        ):
            continue
        counts[j["level"]] += 1
    return {level: counts.get(level, 0) for level in LEVELS}


def pairwise_comparison(
    judgments_a: list[dict], judgments_b: list[dict]
) -> dict | None:
    """Critères 4 et 5 : tableau croisé 3 × 3, accord et kappa, puis la même
    chose en deux classes (répond au claim ou non). Restreint aux paires jugées
    par les deux juges, avec un verdict valide (SUPPORTS, REFUTES ou
    NOT_ENOUGH_INFO) des deux côtés — « illisible » et « citation introuvable »
    ne sont pas des verdicts sur le claim. `None` s'il n'y a aucune paire
    commune valide."""
    verdict_a = {j["pair_id"]: j["verdict"] for j in judgments_a}
    verdict_b = {j["pair_id"]: j["verdict"] for j in judgments_b}
    common = [
        pid
        for pid in sorted(set(verdict_a) & set(verdict_b))
        if verdict_a[pid] in VALID_VERDICTS and verdict_b[pid] in VALID_VERDICTS
    ]
    if not common:
        return None

    labels_a = [verdict_a[pid] for pid in common]
    labels_b = [verdict_b[pid] for pid in common]
    binary_a = [BINARY_LABELS[v] for v in labels_a]
    binary_b = [BINARY_LABELS[v] for v in labels_b]

    return {
        "n_common": len(common),
        "full": cohen_kappa(labels_a, labels_b),
        "binary": cohen_kappa(binary_a, binary_b),
    }


def annotation_crosstab(
    pairs: list[dict], judgments_by_judge: dict[str, list[dict]]
) -> list[dict]:
    """Critère 7 : croise les étiquettes des deux fichiers d'annotation v1 avec
    les juges présents. Une étiquette est le texte `categorie` tel qu'écrit
    dans son fichier, jamais regroupé ni renommé."""
    pairs_by_claim: dict[str, list[dict]] = {}
    for p in pairs:
        pairs_by_claim.setdefault(p["claim_id"], []).append(p)

    responds_by_judge: dict[str, dict[str, bool]] = {
        juge: {j["pair_id"]: _responds(j["verdict"]) for j in judgments}
        for juge, judgments in judgments_by_judge.items()
    }

    rows: list[dict] = []
    for annotation_path in ANNOTATION_FILES:
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        claims_by_label: dict[str, list[str]] = {}
        for claim_id, annotation in annotations.items():
            if claim_id not in pairs_by_claim:
                continue
            claims_by_label.setdefault(annotation["categorie"], []).append(claim_id)

        for label, claim_ids in sorted(claims_by_label.items()):
            row = {
                "fichier": annotation_path.name,
                "etiquette": label,
                "n_claims": len(claim_ids),
                "par_juge": {},
            }
            for juge, responds_by_pair in responds_by_judge.items():
                n_expected_responds = 0
                n_non_expected_responds = 0
                for claim_id in claim_ids:
                    claim_pairs = pairs_by_claim[claim_id]
                    if any(
                        responds_by_pair.get(p["pair_id"], False)
                        for p in claim_pairs
                        if p["document_attendu"]
                    ):
                        n_expected_responds += 1
                    if any(
                        responds_by_pair.get(p["pair_id"], False)
                        for p in claim_pairs
                        if not p["document_attendu"]
                    ):
                        n_non_expected_responds += 1
                row["par_juge"][juge] = {
                    "expected_responds": n_expected_responds,
                    "non_expected_responds": n_non_expected_responds,
                }
            rows.append(row)
    return rows


def _kappa_str(value: float | None) -> str:
    return "non défini" if value is None else f"{value:.4f}"


def _render_judge_section(juge: str, judgments: list[dict] | None) -> list[str]:
    if judgments is None:
        return [f"### {juge}\n", "\n", "absent\n"]
    summary = judge_summary(judgments)
    lines = [
        f"### {juge}\n",
        "\n",
        f"- paires jugées : {summary['n_judged']}\n",
        f"- SUPPORTS : {summary['supports']}\n",
        f"- REFUTES : {summary['refutes']}\n",
        f"- NOT_ENOUGH_INFO : {summary['not_enough_info']}\n",
        f"- illisibles : {summary['illisible']}\n",
        f"- citations introuvables : {summary['citation_introuvable']}\n",
        f"- refus du juge : {summary['refus_du_juge']}\n",
        f"- durée moyenne par jugement : {summary['avg_duration_seconds']:.4f} s\n",
    ]
    return lines


def _render_dataset_section(
    juge: str, judgments: list[dict], pairs_by_id: dict
) -> list[str]:
    summary = dataset_summary(judgments, pairs_by_id)
    return [
        f"### {juge}\n",
        "\n",
        (
            "- part des documents attendus jugés SUPPORTS ou REFUTES : "
            f"{summary['expected_responds_rate']:.4f}\n"
        ),
        (
            "- part des documents non attendus classés devant jugés SUPPORTS "
            f"ou REFUTES : {summary['non_expected_responds_rate']:.4f}\n"
        ),
        (
            "- claims avec au moins un document non attendu jugé SUPPORTS ou "
            f"REFUTES : {summary['n_claims_with_non_expected_responds']}\n"
        ),
        (
            "- claims sans aucun document attendu jugé SUPPORTS ou REFUTES : "
            f"{summary['n_claims_without_expected_responds']}\n"
        ),
    ]


def _render_expected_crosstab_section(
    juge: str, judgments: list[dict], pairs_by_id: dict
) -> list[str]:
    lines = [f"### {juge}\n", "\n"]
    crosstab = expected_crosstab(judgments, pairs_by_id)
    lines.append("- tableau croisé (verdict / étiquette d'origine) :\n")
    for verdict in VALID_VERDICTS:
        for label in ORIGIN_LABELS:
            n = crosstab["confusion"].get((verdict, label), 0)
            lines.append(f"  - {verdict} / {label} : {n}\n")
    lines.append(f"- paires valides (dénominateur) : {crosstab['n_valid']}\n")
    lines.append(f"- part d'accord : {crosstab['agreement']:.4f}\n")
    return lines


def _render_refused_section(
    juge: str, judgments: list[dict] | None, pairs_by_id: dict
) -> list[str]:
    if judgments is None:
        return [f"### {juge}\n", "\n", "absent\n"]
    rows = refused_pairs(judgments, pairs_by_id)
    if not rows:
        return [f"### {juge}\n", "\n", "aucun refus\n"]
    lines = [f"### {juge}\n", "\n"]
    for row in rows:
        famille = ", ".join(row["famille"])
        etiquette = row["etiquette_origine"] or ""
        lines.append(f"- {row['pair_id']} : famille {famille}, étiquette {etiquette}\n")
    return lines


def _render_devant_section(
    juge: str, judgments: list[dict], pairs_by_id: dict
) -> list[str]:
    lines = [f"### {juge}\n", "\n"]
    summary = devant_summary(judgments, pairs_by_id)
    lines.append(f"- SUPPORTS : {summary['supports']}\n")
    lines.append(f"- REFUTES : {summary['refutes']}\n")
    lines.append(f"- NOT_ENOUGH_INFO : {summary['not_enough_info']}\n")
    lines.append(
        "- claims avec au moins un document devant jugé SUPPORTS ou REFUTES : "
        f"{len(summary['claims'])}\n"
    )
    lines.append(f"- liste de ces claims : {', '.join(summary['claims'])}\n")
    return lines


def _render_level_section(
    juge: str, judgments: list[dict], pairs_by_id: dict
) -> list[str]:
    lines = [f"### {juge}\n", "\n"]
    attendu_levels = level_distribution(judgments, pairs_by_id, "attendu")
    devant_levels = level_distribution(judgments, pairs_by_id, "devant")
    if attendu_levels is None:
        lines.append("sans niveau\n")
        return lines
    lines.append("- documents attendus (SUPPORT ou CONTRADICT) :\n")
    for level in LEVELS:
        lines.append(f"  - {level} : {attendu_levels[level]}\n")
    lines.append("- documents devant :\n")
    for level in LEVELS:
        lines.append(f"  - {level} : {devant_levels[level]}\n")
    return lines


def _render_pairwise_section(
    juge_a: str, juge_b: str, comparison: dict | None
) -> list[str]:
    lines = [f"### {juge_a} / {juge_b}\n", "\n"]
    if comparison is None:
        lines.append("aucune paire commune\n")
        return lines

    full = comparison["full"]
    binary = comparison["binary"]
    lines.append(f"- paires communes : {comparison['n_common']}\n")
    lines.append("- tableau croisé (3 × 3) :\n")
    for c_a in full["categories"]:
        for c_b in full["categories"]:
            n = full["confusion"].get((c_a, c_b), 0)
            lines.append(f"  - {c_a} / {c_b} : {n}\n")
    lines.append(f"- part d'accord : {full['agreement']:.4f}\n")
    lines.append(f"- kappa : {_kappa_str(full['kappa'])}\n")
    lines.append(
        f"- part d'accord (répond au claim ou non) : {binary['agreement']:.4f}\n"
    )
    lines.append(f"- kappa (répond au claim ou non) : {_kappa_str(binary['kappa'])}\n")
    return lines


def _render_cause_section(judgments: list[dict] | None) -> list[str]:
    if judgments is None:
        return ["absent\n"]
    causes = cause_summary(judgments)
    return [f"- {cause} : {n}\n" for cause, n in causes.items()]


def _render_annotation_section(
    rows: list[dict], juges_presents: list[str]
) -> list[str]:
    lines = [
        "| fichier | étiquette | claims |"
        + "".join(f" {j} (attendu / non attendu) |" for j in juges_presents)
        + "\n",
        "|---|---|---|" + "---|" * len(juges_presents) + "\n",
    ]
    for row in rows:
        cells = f"| {row['fichier']} | {row['etiquette']} | {row['n_claims']} |"
        for juge in juges_presents:
            par_juge = row["par_juge"][juge]
            cells += (
                f" {par_juge['expected_responds']} / "
                f"{par_juge['non_expected_responds']} |"
            )
        lines.append(cells + "\n")
    return lines


def build_report(campagne: str) -> str:
    pairs = load_pairs(_pairs_path(campagne))
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    judgments_by_juge = {juge: load_judgments(campagne, juge) for juge in JUDGE_NAMES}
    present = [juge for juge in JUDGE_NAMES if judgments_by_juge[juge] is not None]

    lines = [f"Campagne : {campagne}\n", "\n"]

    lines.append("## Juges\n\n")
    for juge in JUDGE_NAMES:
        lines.extend(_render_judge_section(juge, judgments_by_juge[juge]))
        lines.append("\n")

    lines.append("## Jeu de données\n\n")
    for juge in present:
        lines.extend(
            _render_dataset_section(juge, judgments_by_juge[juge], pairs_by_id)
        )
        lines.append("\n")

    lines.append("## Étiquette d'origine (documents attendus)\n\n")
    for juge in present:
        lines.extend(
            _render_expected_crosstab_section(
                juge, judgments_by_juge[juge], pairs_by_id
            )
        )
        lines.append("\n")

    lines.append("## Paires refusées par le juge\n\n")
    for juge in present:
        lines.extend(
            _render_refused_section(juge, judgments_by_juge[juge], pairs_by_id)
        )
        lines.append("\n")

    lines.append("## Documents classés devant\n\n")
    for juge in present:
        lines.extend(_render_devant_section(juge, judgments_by_juge[juge], pairs_by_id))
        lines.append("\n")

    lines.append("## Niveau de lecture\n\n")
    for juge in present:
        lines.extend(_render_level_section(juge, judgments_by_juge[juge], pairs_by_id))
        lines.append("\n")

    lines.append("## Accord entre juges\n\n")
    for juge_a, juge_b in itertools.combinations(present, 2):
        comparison = pairwise_comparison(
            judgments_by_juge[juge_a], judgments_by_juge[juge_b]
        )
        lines.extend(_render_pairwise_section(juge_a, juge_b, comparison))
        lines.append("\n")

    lines.append("## Juge par étapes — causes\n\n")
    lines.extend(_render_cause_section(judgments_by_juge["etapes"]))
    lines.append("\n")

    lines.append("## Étiquettes d'error analysis v1\n\n")
    present_judgments = {juge: judgments_by_juge[juge] for juge in present}
    rows = annotation_crosstab(pairs, present_judgments)
    lines.extend(_render_annotation_section(rows, present))

    return "".join(lines)


def write_report(campagne: str) -> Path:
    """Écrit `results/<campagne>/RAPPORT.md` et renvoie son chemin."""
    content = build_report(campagne)
    report_path = RESULTS_DIR / campagne / "RAPPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
