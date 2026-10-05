"""Étiquettes d'origine dans le dashboard (EXE-124, critères 10 à 12).

Logique testée sans navigateur ni serveur Streamlit (H9) : quelle étiquette,
quelles phrases, quels claims — jamais le rendu `st.*` lui-même.
"""

from __future__ import annotations

import dashboard

# ---------------------------------------------------------------------------
# Critère 11 — surlignage des phrases-preuve
# ---------------------------------------------------------------------------


def test_highlight_evidence_sentences_wraps_each_matching_sentence():
    text = "Phrase une. Phrase deux. Phrase trois."

    highlighted = dashboard.highlight_evidence_sentences(
        text, ["Phrase deux.", "Phrase trois."]
    )

    assert highlighted.count("<mark") == 2
    assert highlighted.startswith("Phrase une. <mark")


def test_highlight_evidence_sentences_leaves_unmatched_sentences_untouched():
    text = "Le texte BEIR ne reprend pas tout mot pour mot."

    highlighted = dashboard.highlight_evidence_sentences(
        text, ["une phrase absente du texte"]
    )

    assert highlighted == text


def test_highlight_evidence_sentences_tolerates_empty_list():
    text = "un texte quelconque"
    assert dashboard.highlight_evidence_sentences(text, []) == text


# ---------------------------------------------------------------------------
# Critère 10 — étiquette en clair par document attendu
# ---------------------------------------------------------------------------


def test_render_doc_text_applies_highlighting_after_truncation_split(monkeypatch):
    markdown_calls = []
    monkeypatch.setattr(
        dashboard.st, "markdown", lambda *a, **kw: markdown_calls.append((a, kw))
    )
    monkeypatch.setattr(
        dashboard, "split_at_truncation", lambda text, max_tokens: (text, "")
    )

    dashboard.render_doc_text(
        "une phrase preuve ici",
        key="k",
        max_tokens=256,
        evidence_sentences=["phrase preuve"],
    )

    (html,), _ = markdown_calls[0]
    assert "<mark" in html


def test_render_doc_text_without_evidence_sentences_has_no_highlighting(monkeypatch):
    markdown_calls = []
    monkeypatch.setattr(
        dashboard.st, "markdown", lambda *a, **kw: markdown_calls.append((a, kw))
    )
    monkeypatch.setattr(
        dashboard, "split_at_truncation", lambda text, max_tokens: (text, "")
    )

    dashboard.render_doc_text("un texte sans preuve", key="k", max_tokens=256)

    (html,), _ = markdown_calls[0]
    assert "<mark" not in html


# ---------------------------------------------------------------------------
# Critère 12 — filtre par catégorie, page Comparer deux runs
# ---------------------------------------------------------------------------


def test_filter_rows_by_categorie_is_reexported_for_the_compare_page():
    rows = [{"query_id": "1"}, {"query_id": "2"}]
    categorie_by_qid = {"1": "confirme", "2": "sans_preuve"}

    filtered = dashboard.filter_rows_by_categorie(rows, {"confirme"}, categorie_by_qid)

    assert [r["query_id"] for r in filtered] == ["1"]


def test_label_in_clear_is_reexported_for_the_dashboard():
    assert dashboard.label_in_clear("SUPPORT") == "confirme"
    assert dashboard.label_in_clear("CONTRADICT") == "contredit"
    assert dashboard.label_in_clear("SANS_PREUVE") == "sans preuve"


def test_load_origin_labels_returns_none_when_file_absent(tmp_path):
    assert dashboard.load_origin_labels(tmp_path / "absent.json") is None


def test_load_origin_labels_reads_the_committed_file():
    import json

    path = dashboard.RESULTS_DIR / "etiquettes-origine.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["pairs"]
    assert document["claims"]
    assert dashboard.load_origin_labels(path) == document
