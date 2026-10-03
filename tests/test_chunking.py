"""Tests du découpage en passages (EXE-92, critères 1 et 2).

`offsets_fn` est fabriqué (un token = un caractère) : aucun tokenizer réel
n'est chargé. Ce choix rend les décalages de caractères triviaux à prédire à
la main, sans rien dire sur le tokenizer de production (chargé paresseusement
dans `_default_offsets_fn`, jamais exercé ici).
"""

from __future__ import annotations

from rag_eval_scifact.chunking import chunk_text


def _char_offsets(text: str) -> list[tuple[int, int]]:
    """Un token = un caractère : décalages triviaux à vérifier à la main."""
    return [(i, i + 1) for i in range(len(text))]


# ---------------------------------------------------------------------------
# Critère 1 — fenêtres de `chunk_size` tokens avec `chunk_overlap` de chevauchement
# ---------------------------------------------------------------------------


def test_text_shorter_than_chunk_size_produces_a_single_passage():
    passages = chunk_text(
        "d1", "courte", chunk_size=128, chunk_overlap=32, offsets_fn=_char_offsets
    )

    assert len(passages) == 1
    assert passages[0].text == "courte"
    assert passages[0].doc_id == "d1"


def test_windows_advance_by_chunk_size_minus_overlap():
    # 25 caractères, fenêtres de 10 tokens, chevauchement de 3 -> pas de 7.
    text = "abcdefghijklmnopqrstuvwxy"
    passages = chunk_text(
        "d1", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert [p.text for p in passages] == [
        text[0:10],
        text[7:17],
        text[14:24],
        text[21:25],
    ]


def test_passage_ids_are_doc_id_qualified_and_ordered():
    text = "abcdefghijklmnopqrstuvwxy"
    passages = chunk_text(
        "docX", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert [p.passage_id for p in passages] == [
        "docX::0",
        "docX::1",
        "docX::2",
        "docX::3",
    ]
    assert all(p.doc_id == "docX" for p in passages)


# ---------------------------------------------------------------------------
# Critère 2 — au moins un passage par document, aucun ne dépasse chunk_size tokens
# ---------------------------------------------------------------------------


def test_every_document_produces_at_least_one_passage_even_when_empty():
    passages = chunk_text(
        "d1", "", chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert len(passages) == 1


def test_no_passage_exceeds_chunk_size_tokens():
    text = "x" * 137  # pas un multiple du pas (10 - 3 = 7)
    passages = chunk_text(
        "d1", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert all(p.token_count <= 10 for p in passages)
    assert passages[-1].token_count > 0


def test_token_count_matches_the_number_of_offsets_in_the_window():
    text = "abcdefghijklmnopqrstuvwxy"
    passages = chunk_text(
        "d1", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert [p.token_count for p in passages] == [10, 10, 10, 4]


# ---------------------------------------------------------------------------
# EXE-112 — char_start/char_end : position du passage dans le texte d'origine
# ---------------------------------------------------------------------------


def test_passage_char_offsets_match_its_text_slice():
    text = "abcdefghijklmnopqrstuvwxy"
    passages = chunk_text(
        "d1", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    for p in passages:
        assert text[p.char_start : p.char_end] == p.text


def test_passage_char_offsets_advance_with_the_window():
    text = "abcdefghijklmnopqrstuvwxy"
    passages = chunk_text(
        "d1", text, chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert [(p.char_start, p.char_end) for p in passages] == [
        (0, 10),
        (7, 17),
        (14, 24),
        (21, 25),
    ]


def test_empty_text_passage_has_zero_width_offsets():
    passages = chunk_text(
        "d1", "", chunk_size=10, chunk_overlap=3, offsets_fn=_char_offsets
    )

    assert (passages[0].char_start, passages[0].char_end) == (0, 0)
