"""Prompt du juge LLM (EXE-119, hypothèse H1).

Partagé par tous les juges à prompt unique (local, puis Claude en EXE-120) :
le critère 3 d'EXE-120 exige que le juge Claude reçoive, octet pour octet, le
même prompt système et le même prompt utilisateur que le juge local pour une
même paire — d'où son isolement dans ce module plutôt que dans `judge_local.py`.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a scientific fact-checking annotator. You are given a CLAIM and a "
    "DOCUMENT (title and abstract of a scientific paper). Decide whether the "
    "document contains evidence about the claim. SUPPORTS: the document contains "
    "evidence that the claim is true. REFUTES: the document contains evidence "
    "that the claim is false. NOT_ENOUGH_INFO: the document does not provide "
    "evidence either way, even if it is on the same topic. Quote the single most "
    "decisive sentence, copied verbatim from the document; use an empty string "
    "for NOT_ENOUGH_INFO. Answer with a JSON object only: "
    '{"verdict": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO", "evidence": "...", '
    '"reason": "one sentence"}'
)


def build_user_prompt(claim_text: str, doc_title: str, doc_text: str) -> str:
    """Trois blocs séparés par une ligne vide, dans cet ordre (H1)."""
    return (
        f"CLAIM: {claim_text}\n\n"
        f"DOCUMENT TITLE: {doc_title}\n\n"
        f"DOCUMENT ABSTRACT: {doc_text}"
    )
