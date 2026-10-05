"""Prompt du juge LLM (EXE-119, EXE-125 hypothèse H1).

Partagé par tous les juges à prompt unique (local, puis Claude en EXE-120) :
le critère 3 d'EXE-120 exige que le juge Claude reçoive, octet pour octet, le
même prompt système et le même prompt utilisateur que le juge local pour une
même paire — d'où son isolement dans ce module plutôt que dans `judge_local.py`.

Le prompt système d'EXE-125 (H1) remplace celui d'EXE-119 : il demande en plus
le niveau de lecture (DIRECT, VOCABULARY, REASONING, NONE) nécessaire pour
rattacher le document au claim. Le prompt utilisateur ne change pas.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a scientific fact-checking annotator. You are given a CLAIM and a "
    "DOCUMENT (title and abstract of a scientific paper). Decide whether the "
    "document answers the claim. SUPPORTS: the document shows that the claim is "
    "true. REFUTES: the document shows that the claim is false. NOT_ENOUGH_INFO: "
    "the document does not settle the claim, even if it is on the same topic. "
    "When the verdict is SUPPORTS or REFUTES, also give the level of reading "
    "needed. DIRECT: the document states it in nearly the same words as the "
    "claim. VOCABULARY: the document states it with different terms, and the "
    "reader must know that these terms mean the same thing. REASONING: the "
    "reader must combine or infer beyond what the sentences state. Use NONE for "
    "NOT_ENOUGH_INFO. Quote the single most decisive sentence, copied verbatim "
    "from the document; use an empty string for NOT_ENOUGH_INFO. Answer with a "
    'JSON object only: {"verdict": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO", '
    '"level": "DIRECT" | "VOCABULARY" | "REASONING" | "NONE", "evidence": "...", '
    '"reason": "one sentence"}'
)


def build_user_prompt(claim_text: str, doc_title: str, doc_text: str) -> str:
    """Trois blocs séparés par une ligne vide, dans cet ordre (H1)."""
    return (
        f"CLAIM: {claim_text}\n\n"
        f"DOCUMENT TITLE: {doc_title}\n\n"
        f"DOCUMENT ABSTRACT: {doc_text}"
    )
