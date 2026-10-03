"""Découpage d'un document en passages (fenêtres de tokens avec chevauchement).

Le découpage se fait sur les décalages de caractères de chaque token
(`return_offsets_mapping`), jamais par décodage des ids : la passage retrouve
exactement la sous-chaîne d'origine, sans artefact de détokénisation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Passage:
    passage_id: str
    doc_id: str
    text: str
    token_count: int
    char_start: int
    char_end: int


def chunk_text(
    doc_id: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    offsets_fn: Callable[[str], list[tuple[int, int]]],
) -> list[Passage]:
    """Découpe `text` en passages de `chunk_size` tokens, chevauchement `chunk_overlap`.

    `offsets_fn` retourne le décalage (début, fin) en caractères de chaque
    token du texte (tokenizer réel en production, fabriqué dans les tests —
    jamais chargé tant qu'il n'est pas fourni par l'appelant). Toujours au
    moins un passage, même pour un texte vide ou plus court que `chunk_size`.
    """
    offsets = offsets_fn(text)
    if not offsets:
        return [
            Passage(
                passage_id=f"{doc_id}::0",
                doc_id=doc_id,
                text=text,
                token_count=0,
                char_start=0,
                char_end=0,
            )
        ]

    step = chunk_size - chunk_overlap
    n = len(offsets)
    passages: list[Passage] = []
    start = 0
    index = 0
    while start < n:
        end = min(start + chunk_size, n)
        span_start = offsets[start][0]
        span_end = offsets[end - 1][1]
        passages.append(
            Passage(
                passage_id=f"{doc_id}::{index}",
                doc_id=doc_id,
                text=text[span_start:span_end],
                token_count=end - start,
                char_start=span_start,
                char_end=span_end,
            )
        )
        index += 1
        if end == n:
            break
        start += step
    return passages


def default_offsets_fn(model_name: str) -> Callable[[str], list[tuple[int, int]]]:
    """Décalages de caractères par token, avec le tokenizer du modèle, chargé
    paresseusement, au plus une fois.

    Une campagne de test injecte sa propre fonction (fabriquée) pour ne
    jamais charger de tokenizer réel — voir `retrieve_campaign`. Sans tokens
    spéciaux : leurs décalages (0, 0) casseraient le découpage par tranche de
    caractères.
    """
    tokenizer_box: dict[str, Any] = {}

    def _offsets(text: str) -> list[tuple[int, int]]:
        if "tokenizer" not in tokenizer_box:
            from transformers import AutoTokenizer

            tokenizer_box["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        tokenizer = tokenizer_box["tokenizer"]
        encoding = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        return [tuple(span) for span in encoding["offset_mapping"]]

    return _offsets
