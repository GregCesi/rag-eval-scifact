"""Tests de l'affichage des passages dans le dashboard (EXE-112, critère 8).

Appelle directement les fonctions de rendu (pas de serveur Streamlit, pas de
navigateur — H9) ; les appels `st.*` sont remplacés par des enregistreurs,
comme `streamlit.testing` le ferait, sans dépendre de son runtime. Aucun
modèle d'embedding, aucun corpus réel.
"""

from __future__ import annotations

import dashboard


def _spy(monkeypatch, name: str) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(dashboard.st, name, lambda *a, **kw: calls.append((a, kw)))
    return calls


# ---------------------------------------------------------------------------
# Critère 8 — run en passages sans fichier dérivé
# ---------------------------------------------------------------------------


def test_missing_passage_detail_shows_the_unavailability_caption(monkeypatch):
    caption_calls = _spy(monkeypatch, "caption")
    _spy(monkeypatch, "write")

    dashboard.render_doc_passages_or_text(
        "d1",
        {"title": "T", "text": "un texte complet, jamais coupe"},
        {"unit": "passages"},
        None,
        "q1",
        key="k",
    )

    assert caption_calls == [(("passages non disponibles pour ce run",), {})]


def test_missing_passage_detail_shows_the_full_uncut_text(monkeypatch):
    _spy(monkeypatch, "caption")
    write_calls = _spy(monkeypatch, "write")

    full_text = "un texte complet, jamais coupe"
    dashboard.render_doc_passages_or_text(
        "d1",
        {"title": "T", "text": full_text},
        {"unit": "passages"},
        None,
        "q1",
        key="k",
    )

    assert write_calls == [((full_text,), {})]


def test_passage_detail_present_but_doc_absent_also_shows_the_unavailability_caption(
    monkeypatch,
):
    """Un fichier dérivé existe pour la requête, mais pas pour ce doc précis
    (ne devrait pas arriver — `needed_doc_ids` couvre toujours les docs
    affichés — mais la page ne doit jamais planter)."""
    caption_calls = _spy(monkeypatch, "caption")
    _spy(monkeypatch, "write")

    passage_detail = {"queries": [{"query_id": "q1", "docs": {}}]}
    dashboard.render_doc_passages_or_text(
        "d1",
        {"title": "T", "text": "texte"},
        {"unit": "passages"},
        passage_detail,
        "q1",
        key="k",
    )

    assert caption_calls == [(("passages non disponibles pour ce run",), {})]


# ---------------------------------------------------------------------------
# Critère 4, 5, 6 — un fichier dérivé existe : pas de message d'indisponibilité,
# les passages sont rendus (lisibilité à l'écran laissée à la relecture humaine)
# ---------------------------------------------------------------------------


def test_passage_detail_present_skips_the_unavailability_caption(monkeypatch):
    caption_calls = _spy(monkeypatch, "caption")
    markdown_calls = _spy(monkeypatch, "markdown")

    passage_detail = {
        "queries": [
            {
                "query_id": "q1",
                "docs": {
                    "d1": [
                        {
                            "passage_id": "d1::0",
                            "char_start": 0,
                            "char_end": 4,
                            "token_count": 4,
                            "score": 0.9,
                        }
                    ]
                },
            }
        ]
    }

    dashboard.render_doc_passages_or_text(
        "d1",
        {"title": "", "text": "text"},
        {"unit": "passages"},
        passage_detail,
        "q1",
        key="k",
    )

    assert caption_calls == []
    assert markdown_calls  # le texte annoté est rendu en HTML


# ---------------------------------------------------------------------------
# Hors de l'unité passages : comportement inchangé
# ---------------------------------------------------------------------------


def test_document_unit_writes_the_raw_text_without_touching_passage_detail(monkeypatch):
    caption_calls = _spy(monkeypatch, "caption")
    write_calls = _spy(monkeypatch, "write")

    dashboard.render_doc_passages_or_text(
        "d1",
        {"title": "T", "text": "texte brut"},
        {"unit": "document"},
        None,
        "q1",
        key="k",
    )

    assert caption_calls == []
    assert write_calls == [(("texte brut",), {})]
