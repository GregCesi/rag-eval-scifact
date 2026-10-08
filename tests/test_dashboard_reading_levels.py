"""Rendu de la page « Niveaux de lecture » (EXE-137).

Critère 13 : les critères 1, 4 et 11 sont portés par des tests au niveau du
rendu de la page, comme ceux d'EXE-126 (`test_dashboard_evidence_rendering.py`)
— on appelle les fonctions de rendu directement, en espionnant `dashboard.st`,
sans navigateur ni serveur Streamlit.
"""

from __future__ import annotations

import gzip
import inspect
import json
import shutil
from pathlib import Path

import pytest

import dashboard
from rag_eval_scifact import reading_levels
from rag_eval_scifact.compare import load_run

RESULTS_DIR = Path("results")
V2_GRID_DIR = Path("results/v2-grid")
QWEN_PASSAGES_SANS = (
    V2_GRID_DIR / "dense-qwen3-passages-sans-reranker-2026-10-03T09-56-14-983855"
    ".json.gz"
)

# Format réel de hyde.json (EXE-156) : une liste d'objets, jamais un run.
HYDE_ENTRY = {
    "claim_id": "1",
    "claim_text": "0-dimensional biomaterials show inductive properties.",
    "hyde_text": "faux résumé",
    "model": "llama3.1:8b",
    "duration_seconds": 12.3,
}


def _column_stub(monkeypatch):
    class _Column:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        dashboard.st, "columns", lambda n: [_Column() for _ in range(n)]
    )


def _stub_common_widgets(monkeypatch):
    monkeypatch.setattr(dashboard.st, "header", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "subheader", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "dataframe", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "plotly_chart", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "info", lambda *a, **kw: None)
    monkeypatch.setattr(
        dashboard.st,
        "radio",
        lambda label, options, index=0, **kw: list(options)[index],
    )
    _column_stub(monkeypatch)


def _fake_run(run_name: str) -> dict:
    return {
        "campagne": "_test-dashboard-reading-levels",
        "run_name": run_name,
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
def _real_fab_campaign():
    """Campagne fabriquée, écrite comme un vrai sous-dossier de `results/` pour
    coexister avec v2-grid (EXE-142, critères 7, 9, 10) — supprimée en fin de
    test, aucune trace laissée."""

    def _make(name: str, run_names: list[str]) -> Path:
        campaign_dir = RESULTS_DIR / name
        for run_name in run_names:
            _write_gz(
                campaign_dir / f"{run_name}-20261006.json.gz", _fake_run(run_name)
            )
        return campaign_dir

    created: list[Path] = []

    def _factory(name: str, run_names: list[str]) -> Path:
        path = _make(name, run_names)
        created.append(path)
        return path

    yield _factory

    for path in created:
        shutil.rmtree(path, ignore_errors=True)


def _spy(monkeypatch, name):
    calls: list[tuple] = []
    monkeypatch.setattr(dashboard.st, name, lambda *a, **kw: calls.append((a, kw)))
    return calls


# ---------------------------------------------------------------------------
# Critère 1 — la page existe, s'ouvre avant le test « aucun run choisi »
# ---------------------------------------------------------------------------


def test_navigation_lists_niveaux_de_lecture():
    source = inspect.getsource(dashboard.main)
    assert '"Niveaux de lecture"' in source


def test_main_routes_niveaux_de_lecture_before_requiring_a_selected_run():
    source = inspect.getsource(dashboard.main)
    route_pos = source.index('page == "Niveaux de lecture"')
    guard_pos = source.index("if selected_run is None:")
    assert route_pos < guard_pos


def test_page_reading_levels_renders_without_any_selected_run(monkeypatch):
    """`page_reading_levels` ne prend qu'un corpus — rien qui dépende d'un run
    choisi dans la barre de gauche (critère 1)."""
    monkeypatch.setattr(dashboard.st, "header", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "subheader", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "dataframe", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "plotly_chart", lambda *a, **kw: None)
    monkeypatch.setattr(
        dashboard.st,
        "radio",
        lambda label, options, index=0, **kw: list(options)[index],
    )
    monkeypatch.setattr(
        dashboard.st,
        "selectbox",
        lambda label, options, index=0, **kw: list(options)[index],
    )
    monkeypatch.setattr(
        dashboard.st,
        "multiselect",
        lambda label, options, default=None, **kw: default or [],
    )

    class _Column:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        dashboard.st, "columns", lambda n: [_Column() for _ in range(n)]
    )

    dashboard.page_reading_levels(corpus={})


# ---------------------------------------------------------------------------
# Critère 4 — seuil par défaut « 5 premiers », recalcul au changement
# ---------------------------------------------------------------------------


def test_threshold_widget_defaults_to_5_premiers(monkeypatch):
    radio_calls = []
    monkeypatch.setattr(dashboard.st, "header", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "subheader", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "dataframe", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "plotly_chart", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "multiselect", lambda *a, **kw: [])
    monkeypatch.setattr(
        dashboard.st, "selectbox", lambda label, options, index=0, **kw: options[index]
    )

    def fake_radio(label, options, index=0, **kw):
        radio_calls.append((label, list(options), index))
        return list(options)[index]

    monkeypatch.setattr(dashboard.st, "radio", fake_radio)

    class _Column:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        dashboard.st, "columns", lambda n: [_Column() for _ in range(n)]
    )

    dashboard.page_reading_levels(corpus={})

    seuil_calls = [c for c in radio_calls if c[0] == "Seuil"]
    assert len(seuil_calls) == 1
    _, options, index = seuil_calls[0]
    assert options[index] == "5 premiers"


def test_table_recomputes_when_threshold_changes(monkeypatch):
    goldens = reading_levels.load_goldens()
    runs = [load_run(QWEN_PASSAGES_SANS)]

    rows_at_1 = reading_levels.build_table_rows(goldens, runs, threshold=1)
    rows_at_10 = reading_levels.build_table_rows(goldens, runs, threshold=10)

    display_1, _ = dashboard.reading_levels_table_frames(rows_at_1)
    display_10, _ = dashboard.reading_levels_table_frames(rows_at_10)

    assert not display_1.equals(display_10)


# ---------------------------------------------------------------------------
# Critère 11 — détail d'un groupe : affirmation, document, rang, verdict,
# phrase citée, raison
# ---------------------------------------------------------------------------


def test_render_reading_levels_group_detail_shows_all_required_fields(monkeypatch):
    goldens = reading_levels.load_goldens()
    assert goldens is not None
    run_data = load_run(QWEN_PASSAGES_SANS)
    corpus = dashboard.load_corpus()

    rows = reading_levels.build_group_detail_rows(
        goldens, run_data, "sans_preuve_ne_repond_pas", corpus
    )
    rows = reading_levels.sort_rows_beyond_threshold_first(rows, threshold=5)

    markdown_calls = _spy(monkeypatch, "markdown")
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)

    dashboard.render_reading_levels_group_detail(rows)

    rendered = "\n".join(a[0] for a, _ in markdown_calls)
    sample = rows[0]
    assert sample["query_id"] in rendered
    assert sample["claim_text"] in rendered
    assert sample["doc_id"] in rendered
    assert sample["doc_title"] in rendered
    assert "hors des 100 premiers" in rendered  # au moins un golden non trouvé
    assert sample["verdict"] in rendered


def test_render_reading_levels_group_detail_reports_rank_when_found(monkeypatch):
    goldens = reading_levels.load_goldens()
    run_data = load_run(QWEN_PASSAGES_SANS)
    rows = reading_levels.build_group_detail_rows(goldens, run_data, "direct")
    found_row = next(r for r in rows if r["rank"] is not None)

    markdown_calls = _spy(monkeypatch, "markdown")
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)

    dashboard.render_reading_levels_group_detail([found_row])

    rendered = "\n".join(a[0] for a, _ in markdown_calls)
    assert f"rang {found_row['rank']}" in rendered


def test_render_reading_levels_group_detail_shows_cited_sentence_and_reason(
    monkeypatch,
):
    rows = [
        {
            "query_id": "1",
            "claim_text": "une affirmation",
            "doc_id": "d1",
            "doc_title": "un titre",
            "rank": 3,
            "verdict": "NOT_ENOUGH_INFO",
            "evidence": "la phrase citée par le juge",
            "reason": "la raison donnée par le juge",
        }
    ]
    markdown_calls = _spy(monkeypatch, "markdown")
    caption_calls = _spy(monkeypatch, "caption")
    monkeypatch.setattr(dashboard.st, "divider", lambda *a, **kw: None)

    dashboard.render_reading_levels_group_detail(rows)

    rendered_markdown = "\n".join(a[0] for a, _ in markdown_calls)
    rendered_captions = "\n".join(a[0] for a, _ in caption_calls)
    assert "une affirmation" in rendered_markdown
    assert "la phrase citée par le juge" in rendered_captions
    assert "la raison donnée par le juge" in rendered_captions


# ---------------------------------------------------------------------------
# EXE-142, critère 7 — choix de la campagne, v2-grid par défaut
# ---------------------------------------------------------------------------


def test_campaign_selectbox_lists_campaigns_with_runs_and_defaults_to_v2_grid(
    monkeypatch,
):
    _stub_common_widgets(monkeypatch)
    monkeypatch.setattr(dashboard.st, "multiselect", lambda *a, **kw: [])
    selectbox_calls = []

    def fake_selectbox(label, options, index=0, **kw):
        options = list(options)
        selectbox_calls.append((label, options, index))
        return options[index]

    monkeypatch.setattr(dashboard.st, "selectbox", fake_selectbox)

    dashboard.page_reading_levels(corpus={})

    campagne_calls = [c for c in selectbox_calls if c[0] == "Campagne"]
    assert len(campagne_calls) == 1
    _, options, index = campagne_calls[0]
    assert options == dashboard.list_campaign_dirs(RESULTS_DIR)
    assert options[index] == reading_levels.CAMPAGNE


def test_page_with_v2_grid_shows_the_same_known_numbers_as_before(monkeypatch):
    """Critère 8 : au seuil « 5 premiers », Qwen passages sans reranker rend
    les mêmes effectifs qu'avant ce ticket (relevé du 6 octobre 2026)."""
    _stub_common_widgets(monkeypatch)
    monkeypatch.setattr(dashboard.st, "multiselect", lambda *a, **kw: [])
    monkeypatch.setattr(
        dashboard.st,
        "selectbox",
        lambda label, options, index=0, **kw: list(options)[index],
    )
    captured_rows = {}
    real_render = dashboard.render_reading_levels_table

    def spy_render(rows):
        captured_rows["rows"] = rows
        return real_render(rows)

    monkeypatch.setattr(dashboard, "render_reading_levels_table", spy_render)

    dashboard.page_reading_levels(corpus={})

    rows = captured_rows["rows"]
    row = next(r for r in rows if r["run_name"] == "dense-qwen3-passages-sans-reranker")
    assert row["counts"]["direct"] == {"found": 60, "total": 60}
    assert row["counts"]["vocabulaire"] == {"found": 51, "total": 55}
    assert row["counts"]["raisonnement"] == {"found": 63, "total": 71}
    assert row["counts"]["avec_preuve_ne_repond_pas"] == {"found": 11, "total": 18}
    assert row["counts"]["sans_preuve_repond"] == {"found": 17, "total": 22}
    assert row["counts"]["sans_preuve_ne_repond_pas"] == {"found": 60, "total": 105}
    assert row["counts"]["non_juge"] == {"found": 8, "total": 8}


# ---------------------------------------------------------------------------
# EXE-142, critère 9 — pas de diagramme reranker sans run avec reranker
# ---------------------------------------------------------------------------


def test_reranker_chart_replaced_by_a_sentence_when_the_campaign_has_none(
    monkeypatch, _real_fab_campaign
):
    campaign_name = "_test-dashboard-no-reranker"
    _real_fab_campaign(campaign_name, ["strat-a-sans-reranker"])

    _stub_common_widgets(monkeypatch)
    monkeypatch.setattr(dashboard.st, "multiselect", lambda *a, **kw: [])
    chart_calls = []
    monkeypatch.setattr(
        dashboard,
        "render_reranker_effect_chart",
        lambda *a, **kw: chart_calls.append(a),
    )
    info_calls = []
    monkeypatch.setattr(dashboard.st, "info", lambda *a, **kw: info_calls.append(a))

    def fake_selectbox(label, options, index=0, **kw):
        options = list(options)
        if label == "Campagne":
            return campaign_name
        return options[index]

    monkeypatch.setattr(dashboard.st, "selectbox", fake_selectbox)

    dashboard.page_reading_levels(corpus={})

    assert chart_calls == []
    assert info_calls


# ---------------------------------------------------------------------------
# EXE-142, critère 10 — comparaison par défaut : les runs de la campagne, 6 au plus
# ---------------------------------------------------------------------------


def test_non_run_shaped_json_file_is_not_listed_as_a_run(
    monkeypatch, _real_fab_campaign
):
    """EXE-156, critère 2 : un fichier .json qui n'a pas la forme d'un run
    (ex. hyde.json) rangé dans le dossier d'une campagne n'apparaît pas comme
    run de la page « Niveaux de lecture » et ne la casse pas."""
    campaign_name = "_test-dashboard-non-run-json"
    campaign_dir = _real_fab_campaign(campaign_name, ["strat-a-sans-reranker"])
    (campaign_dir / "hyde.json").write_text(json.dumps([HYDE_ENTRY]), encoding="utf-8")

    _stub_common_widgets(monkeypatch)
    multiselect_calls = []

    def fake_multiselect(label, options, default=None, **kw):
        multiselect_calls.append((label, list(options), default))
        return default or []

    monkeypatch.setattr(dashboard.st, "multiselect", fake_multiselect)

    def fake_selectbox(label, options, index=0, **kw):
        options = list(options)
        if label == "Campagne":
            return campaign_name
        return options[index]

    monkeypatch.setattr(dashboard.st, "selectbox", fake_selectbox)

    dashboard.page_reading_levels(corpus={})

    _, options, _ = multiselect_calls[0]
    assert "hyde" not in options
    assert options == ["strat-a-sans-reranker"]


def test_comparison_default_is_the_chosen_campaigns_runs_up_to_six(
    monkeypatch, _real_fab_campaign
):
    campaign_name = "_test-dashboard-many-runs"
    run_names = [f"strat-{i}-sans-reranker" for i in range(8)]
    _real_fab_campaign(campaign_name, run_names)

    _stub_common_widgets(monkeypatch)
    multiselect_calls = []

    def fake_multiselect(label, options, default=None, **kw):
        multiselect_calls.append((label, list(options), default))
        return default or []

    monkeypatch.setattr(dashboard.st, "multiselect", fake_multiselect)

    def fake_selectbox(label, options, index=0, **kw):
        options = list(options)
        if label == "Campagne":
            return campaign_name
        return options[index]

    monkeypatch.setattr(dashboard.st, "selectbox", fake_selectbox)

    dashboard.page_reading_levels(corpus={})

    assert len(multiselect_calls) == 1
    _, options, default = multiselect_calls[0]
    assert set(options) == set(run_names)
    assert len(default) == 6
    assert set(default).issubset(set(run_names))
    assert default != list(reading_levels.DEFAULT_COMPARISON_RUN_NAMES)
