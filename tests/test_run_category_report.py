"""Tests du CLI du rapport par catégorie (EXE-142, critères 2 et 6).

Campagne fabriquée en dossier temporaire : aucun fichier réel n'est touché par
ces tests, sauf celui qui vérifie explicitement le critère 2 sur les fichiers
réels (porté par `test_category_report.py`, pas ici)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rag_eval_scifact import category_report, run_category_report
from rag_eval_scifact import report as report_module

RUN = {
    "campagne": "fab-campagne",
    "run_name": "strat-a-sans-reranker",
    "date": "2026-10-06T00:00:00",
    "dataset_hash": "sha256:fake",
    "config": {},
    "metrics": {},
    "extended_metrics": {},
    "queries": [],
}


def _write_gz(path: Path, run_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(run_data, f)


@pytest.fixture
def _isolated_results(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(report_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        category_report, "ORIGIN_LABELS_PATH", results_dir / "etiquettes-origine.json"
    )
    results_dir.mkdir(parents=True)
    (results_dir / "etiquettes-origine.json").write_text(
        json.dumps({"pairs": [], "claims": []}), encoding="utf-8"
    )
    return results_dir


def test_main_with_campagne_option_writes_under_that_campaign(_isolated_results):
    results_dir = _isolated_results
    (results_dir / "fab-campagne").mkdir()
    _write_gz(results_dir / "fab-campagne" / "strat-a-sans-reranker-1.json.gz", RUN)

    run_category_report.main(["--campagne", "fab-campagne"])

    assert (results_dir / "fab-campagne" / "RAPPORT-PAR-CATEGORIE.md").exists()


def test_main_reports_a_sentence_and_exits_cleanly_when_the_campaign_has_no_runs(
    _isolated_results, capsys
):
    results_dir = _isolated_results
    (results_dir / "vide").mkdir()

    with pytest.raises(SystemExit):
        run_category_report.main(["--campagne", "vide"])

    out = capsys.readouterr().out
    assert "vide" in out
    assert not (results_dir / "vide" / "RAPPORT-PAR-CATEGORIE.md").exists()


def test_main_ignores_a_non_run_shaped_json_file(_isolated_results):
    """EXE-156, critère 1 : un fichier .json qui n'a pas la forme d'un run
    (ex. hyde.json) rangé dans le dossier de la campagne n'empêche pas le
    rapport de s'écrire."""
    results_dir = _isolated_results
    (results_dir / "fab-campagne").mkdir()
    _write_gz(results_dir / "fab-campagne" / "strat-a-sans-reranker-1.json.gz", RUN)
    (results_dir / "fab-campagne" / "hyde.json").write_text(
        json.dumps([{"claim_id": "1", "hyde_text": "faux résumé"}]), encoding="utf-8"
    )

    run_category_report.main(["--campagne", "fab-campagne"])

    content = (results_dir / "fab-campagne" / "RAPPORT-PAR-CATEGORIE.md").read_text(
        encoding="utf-8"
    )
    assert "hyde" not in content


def test_main_reports_a_sentence_and_exits_cleanly_on_a_malformed_gz_file(
    _isolated_results, capsys
):
    """EXE-156, critère 4 : un fichier .json.gz qui n'a pas la forme d'un run
    produit une phrase qui le nomme, sans trace Python."""
    results_dir = _isolated_results
    bad_path = results_dir / "fab-campagne" / "pas-un-run-1.json.gz"
    _write_gz(bad_path, [{"not": "a run"}])

    with pytest.raises(SystemExit):
        run_category_report.main(["--campagne", "fab-campagne"])

    out = capsys.readouterr().out
    assert bad_path.name in out
    assert not (results_dir / "fab-campagne" / "RAPPORT-PAR-CATEGORIE.md").exists()
