"""Surlignage des phrases-preuve quel que soit le mode d'affichage (EXE-126).

Critère 10 : ces tests lisent les fichiers commités (`results/etiquettes-
origine.json`, `results/v2-grid`, `data/scifact/corpus.jsonl`) et appellent les
fonctions de rendu directement (pas de navigateur, pas de serveur Streamlit,
comme les autres tests `test_dashboard_*`), sur le document de contrôle du
ticket : claim 3, document 14717500, étiquette SUPPORT, 3 phrases-preuve,
toutes présentes mot pour mot dans le texte du document.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import dashboard

RESULTS_DIR = Path("results")
CORPUS_PATH = Path("data/scifact/corpus.jsonl")
CONTROL_PASSAGES_PATH = (
    RESULTS_DIR
    / "v2-grid"
    / "passages"
    / "dense-qwen3-passages-sans-reranker-2026-10-03T09-56-14-983855.passages.json.gz"
)


def _load_corpus_doc(doc_id: str) -> dict:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["_id"] == doc_id:
                return record
    raise AssertionError(f"doc {doc_id} absent du corpus commité")


def _load_origin_pair(query_id: str, doc_id: str) -> dict:
    document = json.loads(
        (RESULTS_DIR / "etiquettes-origine.json").read_text(encoding="utf-8")
    )
    for pair in document["pairs"]:
        if pair["query_id"] == query_id and pair["doc_id"] == doc_id:
            return pair
    raise AssertionError(
        f"paire ({query_id}, {doc_id}) absente d'etiquettes-origine.json"
    )


def _load_control_passages() -> list[dict]:
    with gzip.open(CONTROL_PASSAGES_PATH, "rt", encoding="utf-8") as f:
        detail = json.load(f)
    query = next(q for q in detail["queries"] if q["query_id"] == "3")
    return query["docs"]["14717500"]


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


CONTROL_DOC = _load_corpus_doc("14717500")
CONTROL_PAIR = _load_origin_pair("3", "14717500")
CONTROL_EVIDENCE = CONTROL_PAIR["evidence_sentences"]


def _spy_markdown(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(
        dashboard.st, "markdown", lambda *a, **kw: calls.append((a, kw))
    )
    return calls


# ---------------------------------------------------------------------------
# Fixtures de contrôle — garantissent que les faits supposés par H1/H3 tiennent
# toujours sur les données commitées, avant de tester le rendu lui-même.
# ---------------------------------------------------------------------------


def test_control_pair_is_support_with_three_evidence_sentences_all_verbatim():
    assert CONTROL_PAIR["label"] == "SUPPORT"
    assert len(CONTROL_EVIDENCE) == 3
    assert all(s in CONTROL_DOC["text"] for s in CONTROL_EVIDENCE)


# ---------------------------------------------------------------------------
# Critère 1 — mode entier (aucune troncature, aucun passage)
# ---------------------------------------------------------------------------


def test_render_doc_text_entire_mode_highlights_all_three_evidence_sentences(
    monkeypatch,
):
    calls = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)

    dashboard.render_doc_text(
        CONTROL_DOC["text"],
        key="k",
        max_tokens=None,
        evidence_sentences=CONTROL_EVIDENCE,
    )

    (html,), _ = calls[-1]
    assert html.count("<mark") == 3


def test_render_doc_text_entire_mode_readable_text_unchanged_by_highlighting(
    monkeypatch,
):
    """Critère 6 : une fois le balisage retiré, le texte est identique à
    celui produit sans phrases-preuve."""
    calls_with = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    dashboard.render_doc_text(
        CONTROL_DOC["text"],
        key="k",
        max_tokens=None,
        evidence_sentences=CONTROL_EVIDENCE,
    )
    (html_with,), _ = calls_with[-1]

    calls_without = _spy_markdown(monkeypatch)
    dashboard.render_doc_text(CONTROL_DOC["text"], key="k", max_tokens=None)
    (html_without,), _ = calls_without[-1]

    stripped_with = re.sub(r"</?mark[^>]*>", "", html_with)
    assert stripped_with == html_without


# ---------------------------------------------------------------------------
# Critère 3 — mode tronqué à 256 tokens (y compris la phrase coupée)
# ---------------------------------------------------------------------------


def test_render_doc_text_truncated_mode_highlights_all_three_evidence_sentences(
    monkeypatch,
):
    calls = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)

    dashboard.render_doc_text(
        CONTROL_DOC["text"],
        key="k",
        max_tokens=256,
        evidence_sentences=CONTROL_EVIDENCE,
    )

    (html,), _ = calls[-1]
    assert html.count("<mark") == 3


def test_render_doc_text_truncated_mode_keeps_the_cut_at_the_same_place(monkeypatch):
    """La phrase qui tombe dans la partie coupée ne doit pas déplacer la
    coupure (docstring de `render_doc_text`, H1 (a))."""
    calls_with = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
    dashboard.render_doc_text(
        CONTROL_DOC["text"],
        key="k",
        max_tokens=256,
        evidence_sentences=CONTROL_EVIDENCE,
    )
    (html_with,), _ = calls_with[-1]

    calls_without = _spy_markdown(monkeypatch)
    dashboard.render_doc_text(CONTROL_DOC["text"], key="k", max_tokens=256)
    (html_without,), _ = calls_without[-1]

    stripped_with = re.sub(r"</?mark[^>]*>", "", html_with)
    assert stripped_with == html_without
    assert "TRONCATURE" in html_with


# ---------------------------------------------------------------------------
# Critère 2 — mode passages, fichier dérivé présent
# ---------------------------------------------------------------------------


def test_render_doc_text_with_passages_highlights_evidence_and_keeps_best_marker(
    monkeypatch,
):
    calls = _spy_markdown(monkeypatch)

    doc_passages = _load_control_passages()
    full_text = f"{CONTROL_DOC['title']} {CONTROL_DOC['text']}"
    dashboard.render_doc_text_with_passages(
        full_text, doc_passages, key="k", evidence_sentences=CONTROL_EVIDENCE
    )

    (html,), _ = calls[-1]
    assert "<mark" in html
    assert "border-bottom:2px solid #2ecc71" in html  # critère 5, EXE-112


def test_render_doc_text_with_passages_readable_text_unchanged_by_highlighting(
    monkeypatch,
):
    """Critère 6, mode passages."""
    doc_passages = _load_control_passages()
    full_text = f"{CONTROL_DOC['title']} {CONTROL_DOC['text']}"

    calls_with = _spy_markdown(monkeypatch)
    dashboard.render_doc_text_with_passages(
        full_text, doc_passages, key="k", evidence_sentences=CONTROL_EVIDENCE
    )
    (html_with,), _ = calls_with[-1]

    calls_without = _spy_markdown(monkeypatch)
    dashboard.render_doc_text_with_passages(full_text, doc_passages, key="k")
    (html_without,), _ = calls_without[-1]

    assert _strip_tags(html_with) == _strip_tags(html_without)


def test_render_doc_text_with_passages_marks_exactly_one_non_overlapping_sentence(
    monkeypatch,
):
    """Cas simple, sans ambiguïté de découpe, pour vérifier le nombre exact
    de marqueurs (critère 10 : « nombre de phrases surlignées »)."""
    calls = _spy_markdown(monkeypatch)
    text = "AAAA BBBB CCCC DDDD"
    doc_passages = [
        {
            "passage_id": "d::0",
            "char_start": 0,
            "char_end": 10,
            "token_count": 2,
            "score": 0.1,
        },
        {
            "passage_id": "d::1",
            "char_start": 10,
            "char_end": 19,
            "token_count": 2,
            "score": 0.9,
        },
    ]

    dashboard.render_doc_text_with_passages(
        text, doc_passages, key="k", evidence_sentences=["CCCC"]
    )

    (html,), _ = calls[-1]
    assert html.count("<mark") == 1
    assert re.search(r"<mark[^>]*>CCCC</mark>", html)


# ---------------------------------------------------------------------------
# render_doc_passages_or_text — fil complet (critère 2, correction du bug 1a)
# ---------------------------------------------------------------------------


def test_render_doc_passages_or_text_threads_evidence_into_passages_mode(monkeypatch):
    calls = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)

    doc_passages = _load_control_passages()
    passage_detail = {
        "queries": [{"query_id": "3", "docs": {"14717500": doc_passages}}]
    }

    dashboard.render_doc_passages_or_text(
        "14717500",
        CONTROL_DOC,
        {"unit": "passages"},
        passage_detail,
        "3",
        key="k",
        evidence_sentences=CONTROL_EVIDENCE,
    )

    (html,), _ = calls[-1]
    assert "<mark" in html


# ---------------------------------------------------------------------------
# Critère 7 — document sans preuve : aucun surlignage
# ---------------------------------------------------------------------------


def test_render_doc_text_without_evidence_sentences_highlights_nothing_entire_mode(
    monkeypatch,
):
    calls = _spy_markdown(monkeypatch)
    monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)

    dashboard.render_doc_text(CONTROL_DOC["text"], key="k", max_tokens=None)

    (html,), _ = calls[-1]
    assert "<mark" not in html


def test_render_doc_text_with_passages_without_evidence_sentences_highlights_nothing(
    monkeypatch,
):
    calls = _spy_markdown(monkeypatch)
    doc_passages = _load_control_passages()
    full_text = f"{CONTROL_DOC['title']} {CONTROL_DOC['text']}"

    dashboard.render_doc_text_with_passages(full_text, doc_passages, key="k")

    (html,), _ = calls[-1]
    assert "<mark" not in html


# ---------------------------------------------------------------------------
# Critère 8 — phrase-preuve non localisée dans ce texte
# ---------------------------------------------------------------------------

MISSING_PAIR = _load_origin_pair("42", "18174210")
MISSING_DOC = _load_corpus_doc("18174210")


def test_count_unlocated_evidence_sentences_matches_the_committed_data():
    assert MISSING_PAIR["label"] == "CONTRADICT"
    assert len(MISSING_PAIR["evidence_sentences"]) == 3

    missing = dashboard.count_unlocated_evidence_sentences(
        MISSING_DOC["text"], MISSING_PAIR["evidence_sentences"]
    )

    assert missing == 1


def test_render_doc_text_announces_unlocated_evidence_sentences(monkeypatch):
    caption_calls = []
    monkeypatch.setattr(
        dashboard.st, "caption", lambda *a, **kw: caption_calls.append((a, kw))
    )
    monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)

    dashboard.render_doc_text(
        MISSING_DOC["text"],
        key="k",
        max_tokens=None,
        evidence_sentences=MISSING_PAIR["evidence_sentences"],
    )

    assert caption_calls == [
        (("1 phrase(s)-preuve non localisée(s) dans ce texte",), {})
    ]


def test_render_doc_text_says_nothing_when_all_evidence_sentences_are_located(
    monkeypatch,
):
    caption_calls = []
    monkeypatch.setattr(
        dashboard.st, "caption", lambda *a, **kw: caption_calls.append((a, kw))
    )
    monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)

    dashboard.render_doc_text(
        CONTROL_DOC["text"],
        key="k",
        max_tokens=None,
        evidence_sentences=CONTROL_EVIDENCE,
    )

    assert caption_calls == []


# ---------------------------------------------------------------------------
# Critère 9 — « Copier pour l'IA » donne l'étiquette en clair et les phrases
# ---------------------------------------------------------------------------


def test_export_query_markdown_includes_label_and_evidence_sentences():
    q = {
        "query_id": "3",
        "query_text": "une question",
        "per_query_metrics": {"best_rank": 1},
        "expected_docs": [{"doc_id": "14717500", "token_count": 10}],
        "retrieved_top100": [],
    }
    corpus = {"14717500": {"title": CONTROL_DOC["title"], "text": CONTROL_DOC["text"]}}
    label_by_pair = {("3", "14717500"): "SUPPORT"}
    evidence_by_pair = {("3", "14717500"): CONTROL_EVIDENCE}

    md = dashboard.export_query_markdown(
        q,
        corpus,
        {},
        {"name": "dense", "model": "m", "max_seq_length": 256, "unit": "document"},
        label_by_pair,
        evidence_by_pair,
    )

    assert "**Étiquette d'origine** : confirme" in md
    for sentence in CONTROL_EVIDENCE:
        assert f"- {sentence}" in md


def test_export_query_markdown_omits_label_section_when_no_label():
    q = {
        "query_id": "3",
        "query_text": "une question",
        "per_query_metrics": {"best_rank": 1},
        "expected_docs": [{"doc_id": "14717500", "token_count": 10}],
        "retrieved_top100": [],
    }
    corpus = {"14717500": {"title": CONTROL_DOC["title"], "text": CONTROL_DOC["text"]}}

    md = dashboard.export_query_markdown(
        q,
        corpus,
        {},
        {"name": "dense", "model": "m", "max_seq_length": 256, "unit": "document"},
    )

    assert "Étiquette d'origine" not in md
    assert "Phrases-preuve" not in md


def test_export_compare_markdown_includes_label_and_evidence_sentences():
    run = {
        "queries": [
            {
                "query_id": "3",
                "query_text": "une question",
                "expected_docs": [{"doc_id": "14717500", "token_count": 10}],
                "retrieved_top100": [],
            }
        ]
    }
    corpus = {"14717500": {"title": CONTROL_DOC["title"], "text": CONTROL_DOC["text"]}}
    label_by_pair = {("3", "14717500"): "SUPPORT"}
    evidence_by_pair = {("3", "14717500"): CONTROL_EVIDENCE}

    md = dashboard.export_compare_markdown(
        "3", run, run, "A", "B", corpus, label_by_pair, evidence_by_pair
    )

    assert "**Étiquette d'origine** : confirme" in md
    for sentence in CONTROL_EVIDENCE:
        assert f"- {sentence}" in md
