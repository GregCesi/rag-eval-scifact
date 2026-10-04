"""Juge par étapes : quatre questions courtes à un petit modèle plutôt qu'une
seule difficile (EXE-121, H1).

Graphe LangGraph à quatre nœuds, sans boucle : claim (étape 1, sur le claim
seul) -> document (étape 2, cite au plus 3 phrases du document, vérifiées mot
pour mot) -> verdict (étape 3, sur le claim et les seules phrases retenues) ->
cause (étape 4, seulement quand l'étape 3 répond « ni l'un ni l'autre »).
Quand l'étape 2 ne retient aucune phrase, le jugement est NOT_ENOUGH_INFO /
« hors sujet » sans appeler les étapes 3 et 4 (critère 4).

`call_fn` est injectable (tests sans aucun modèle réel, même signature que
`judge_local.call_ollama`) ; en production c'est `call_ollama`, importé sans
modifier `judge_local.py` (le juge local ne change pas).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag_eval_scifact.judge_local import MAX_RETRIES, SEED, citation_found

DEFAULT_MODEL = "llama3.1:8b"
VALID_STEP3_VERDICTS = {"SUPPORTS", "REFUTES", "NEITHER"}
VALID_CAUSES = {"vocabulaire", "inférence", "sujet voisin"}
MAX_SENTENCES = 3

CallFn = Callable[[str, str, str, int], str]

CLAIM_SYSTEM_PROMPT = (
    "You analyze a single scientific CLAIM in isolation, with no document. "
    "Identify, in three short fields: `topic` (the entity, phenomenon or "
    "mechanism the claim is about), `effect` (the property or effect the "
    "claim attributes to it), and `direction` (the sense of that "
    "attribution, e.g. increase, decrease, presence, absence, equivalence). "
    "Answer with a JSON object only: "
    '{"topic": "...", "effect": "...", "direction": "..."}'
)

DOCUMENT_SYSTEM_PROMPT = (
    "You are given a CLAIM's topic and effect, and a DOCUMENT (title and "
    "abstract of a scientific paper). Quote at most 3 sentences from the "
    "document, copied verbatim, that discuss the claim's topic or its "
    "effect. If no sentence in the document discusses them, answer with an "
    "empty list. Answer with a JSON object only: "
    '{"sentences": ["...", ...]}'
)

VERDICT_SYSTEM_PROMPT = (
    "You are given a CLAIM and a short list of SENTENCES, each copied "
    "verbatim from a document. Decide whether the sentences, taken "
    "together, support the claim, refute it, or neither. Use NEITHER when "
    "the sentences do not settle the claim either way. Point to the single "
    "most decisive sentence, copied verbatim from the list given. Answer "
    "with a JSON object only: "
    '{"verdict": "SUPPORTS" | "REFUTES" | "NEITHER", "decisive_sentence": "..."}'
)

CAUSE_SYSTEM_PROMPT = (
    "The sentences given neither supported nor refuted the claim. Choose "
    "the single most likely cause, among: `vocabulaire` (the document may "
    "say the same thing with different words), `inférence` (concluding "
    "would require reasoning the sentences do not make explicit), "
    "`sujet voisin` (same general theme, but a different question). Answer "
    "with a JSON object only: "
    '{"cause": "vocabulaire" | "inférence" | "sujet voisin"}'
)


def _build_claim_user_prompt(claim_text: str) -> str:
    return f"CLAIM: {claim_text}"


def _build_document_user_prompt(
    claim_profile: dict, doc_title: str, doc_text: str
) -> str:
    return (
        f"CLAIM TOPIC: {claim_profile['topic']}\n\n"
        f"CLAIM EFFECT: {claim_profile['effect']}\n\n"
        f"DOCUMENT TITLE: {doc_title}\n\n"
        f"DOCUMENT ABSTRACT: {doc_text}"
    )


def _build_sentences_user_prompt(claim_text: str, sentences: list[str]) -> str:
    bullets = "\n".join(f"- {s}" for s in sentences)
    return f"CLAIM: {claim_text}\n\nSENTENCES:\n{bullets}"


def _parse_json_object(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_claim_response(raw: str) -> dict | None:
    data = _parse_json_object(raw)
    if data is None:
        return None
    fields = ("topic", "effect", "direction")
    if not all(isinstance(data.get(f), str) and data.get(f) for f in fields):
        return None
    return {f: data[f] for f in fields}


def _parse_document_response(raw: str, doc_text: str) -> dict | None:
    data = _parse_json_object(raw)
    if data is None:
        return None
    candidates = data.get("sentences")
    if not isinstance(candidates, list) or len(candidates) > MAX_SENTENCES:
        return None
    if not all(isinstance(s, str) for s in candidates):
        return None
    retained = [s for s in candidates if s != "" and citation_found(s, doc_text)]
    discarded = len(candidates) - len(retained)
    return {"sentences": retained, "discarded": discarded}


def _parse_verdict_response(raw: str, retained_sentences: list[str]) -> dict | None:
    data = _parse_json_object(raw)
    if data is None:
        return None
    verdict = data.get("verdict")
    decisive = data.get("decisive_sentence")
    if verdict not in VALID_STEP3_VERDICTS:
        return None
    if not isinstance(decisive, str) or decisive not in retained_sentences:
        return None
    return {"verdict": verdict, "decisive_sentence": decisive}


def _parse_cause_response(raw: str) -> dict | None:
    data = _parse_json_object(raw)
    if data is None:
        return None
    cause = data.get("cause")
    if cause not in VALID_CAUSES:
        return None
    return {"cause": cause}


class GraphState(TypedDict):
    claim_text: str
    doc_title: str
    doc_text: str
    claim_profile: dict | None
    sentences: list[str]
    discarded_sentences: int
    final_verdict: str | None
    evidence: str
    cause: str | None
    illisible: bool
    steps: dict
    model_calls: int


def _call_with_retries(
    call_fn: CallFn,
    model: str,
    seed: int,
    max_retries: int,
    system_prompt: str,
    user_prompt: str,
    parse_fn: Callable[[str], dict | None],
) -> tuple[dict | None, int]:
    """Un appel, puis jusqu'à `max_retries` nouvelles tentatives tant que la
    réponse est illisible (critère 11)."""
    attempts = 0
    for _ in range(max_retries + 1):
        attempts += 1
        raw = call_fn(model, system_prompt, user_prompt, seed)
        parsed = parse_fn(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def _make_claim_node(call_fn: CallFn, model: str, seed: int, max_retries: int):
    def node(state: GraphState) -> dict:
        user_prompt = _build_claim_user_prompt(state["claim_text"])
        parsed, attempts = _call_with_retries(
            call_fn,
            model,
            seed,
            max_retries,
            CLAIM_SYSTEM_PROMPT,
            user_prompt,
            _parse_claim_response,
        )
        step_record = {"input": {"claim_text": state["claim_text"]}, "output": parsed}
        update = {
            "steps": {**state["steps"], "claim": step_record},
            "model_calls": state["model_calls"] + attempts,
        }
        if parsed is None:
            update["illisible"] = True
        else:
            update["claim_profile"] = parsed
        return update

    return node


def _make_document_node(call_fn: CallFn, model: str, seed: int, max_retries: int):
    def node(state: GraphState) -> dict:
        step_input = {
            "claim_profile": state["claim_profile"],
            "doc_title": state["doc_title"],
            "doc_text": state["doc_text"],
        }
        user_prompt = _build_document_user_prompt(
            state["claim_profile"], state["doc_title"], state["doc_text"]
        )
        parsed, attempts = _call_with_retries(
            call_fn,
            model,
            seed,
            max_retries,
            DOCUMENT_SYSTEM_PROMPT,
            user_prompt,
            lambda raw: _parse_document_response(raw, state["doc_text"]),
        )
        update = {
            "steps": {
                **state["steps"],
                "document": {"input": step_input, "output": parsed},
            },
            "model_calls": state["model_calls"] + attempts,
        }
        if parsed is None:
            update["illisible"] = True
            return update

        update["sentences"] = parsed["sentences"]
        update["discarded_sentences"] = parsed["discarded"]
        if not parsed["sentences"]:
            update["final_verdict"] = "NOT_ENOUGH_INFO"
            update["cause"] = "hors sujet"
            update["evidence"] = ""
        return update

    return node


def _make_verdict_node(call_fn: CallFn, model: str, seed: int, max_retries: int):
    def node(state: GraphState) -> dict:
        step_input = {
            "claim_text": state["claim_text"],
            "sentences": state["sentences"],
        }
        user_prompt = _build_sentences_user_prompt(
            state["claim_text"], state["sentences"]
        )
        parsed, attempts = _call_with_retries(
            call_fn,
            model,
            seed,
            max_retries,
            VERDICT_SYSTEM_PROMPT,
            user_prompt,
            lambda raw: _parse_verdict_response(raw, state["sentences"]),
        )
        update = {
            "steps": {
                **state["steps"],
                "verdict": {"input": step_input, "output": parsed},
            },
            "model_calls": state["model_calls"] + attempts,
        }
        if parsed is None:
            update["illisible"] = True
            return update

        update["evidence"] = parsed["decisive_sentence"]
        if parsed["verdict"] in ("SUPPORTS", "REFUTES"):
            update["final_verdict"] = parsed["verdict"]
        return update

    return node


def _make_cause_node(call_fn: CallFn, model: str, seed: int, max_retries: int):
    def node(state: GraphState) -> dict:
        step_input = {
            "claim_text": state["claim_text"],
            "sentences": state["sentences"],
        }
        user_prompt = _build_sentences_user_prompt(
            state["claim_text"], state["sentences"]
        )
        parsed, attempts = _call_with_retries(
            call_fn,
            model,
            seed,
            max_retries,
            CAUSE_SYSTEM_PROMPT,
            user_prompt,
            _parse_cause_response,
        )
        update = {
            "steps": {
                **state["steps"],
                "cause": {"input": step_input, "output": parsed},
            },
            "model_calls": state["model_calls"] + attempts,
        }
        if parsed is None:
            update["illisible"] = True
            return update

        update["final_verdict"] = "NOT_ENOUGH_INFO"
        update["cause"] = parsed["cause"]
        return update

    return node


def _route_after_claim(state: GraphState) -> str:
    return END if state["illisible"] else "document"


def _route_after_document(state: GraphState) -> str:
    if state["illisible"] or state["final_verdict"] is not None:
        return END
    return "verdict"


def _route_after_verdict(state: GraphState) -> str:
    if state["illisible"] or state["final_verdict"] is not None:
        return END
    return "cause"


def _build_graph(call_fn: CallFn, model: str, seed: int, max_retries: int):
    graph = StateGraph(GraphState)
    graph.add_node("claim", _make_claim_node(call_fn, model, seed, max_retries))
    graph.add_node("document", _make_document_node(call_fn, model, seed, max_retries))
    graph.add_node("verdict", _make_verdict_node(call_fn, model, seed, max_retries))
    graph.add_node("cause", _make_cause_node(call_fn, model, seed, max_retries))
    graph.set_entry_point("claim")
    graph.add_conditional_edges(
        "claim", _route_after_claim, {"document": "document", END: END}
    )
    graph.add_conditional_edges(
        "document", _route_after_document, {"verdict": "verdict", END: END}
    )
    graph.add_conditional_edges(
        "verdict", _route_after_verdict, {"cause": "cause", END: END}
    )
    graph.add_edge("cause", END)
    return graph.compile()


def judge_pair(
    pair: dict,
    call_fn: CallFn,
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Juge une paire par le graphe à quatre étapes. Ne transmet jamais `pair["rank"]`
    ni `pair["document_attendu"]` au modèle (critère 11, règle commune)."""
    graph = _build_graph(call_fn, model, seed, max_retries)

    start = time.monotonic()
    result = graph.invoke(
        {
            "claim_text": pair["claim_text"],
            "doc_title": pair["doc_title"],
            "doc_text": pair["doc_text"],
            "claim_profile": None,
            "sentences": [],
            "discarded_sentences": 0,
            "final_verdict": None,
            "evidence": "",
            "cause": None,
            "illisible": False,
            "steps": {},
            "model_calls": 0,
        }
    )
    duration = time.monotonic() - start

    judgment = {
        "pair_id": pair["pair_id"],
        "claim_id": pair["claim_id"],
        "doc_id": pair["doc_id"],
        "model": model,
        "duration_seconds": duration,
        "model_calls": result["model_calls"],
        "discarded_sentences": result["discarded_sentences"],
        "steps": result["steps"],
    }

    if result["illisible"]:
        return {**judgment, "verdict": "illisible", "evidence": "", "cause": ""}

    return {
        **judgment,
        "verdict": result["final_verdict"],
        "evidence": result["evidence"],
        "cause": result["cause"] or "",
    }


def judge_pairs(
    pairs: list[dict],
    call_fn: CallFn,
    already_judged_ids: set[str],
    model: str = DEFAULT_MODEL,
    seed: int = SEED,
    limit: int | None = None,
) -> list[dict]:
    """Juge les paires non encore jugées, dans l'ordre, jusqu'à `limit` (règle commune,
    comme les deux autres juges)."""
    judgments: list[dict] = []
    for pair in pairs:
        if pair["pair_id"] in already_judged_ids:
            print(f"{pair['pair_id']} : déjà jugée, non rejugée")
            continue
        if limit is not None and len(judgments) >= limit:
            break
        judgments.append(judge_pair(pair, call_fn, model=model, seed=seed))
    return judgments
