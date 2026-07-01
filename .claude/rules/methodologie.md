---
description: Décisions négatives méthodologiques — ce que le projet interdit volontairement
---

# Méthodologie — décisions négatives

Ces règles sont des **interdictions explicites**. Elles protègent l'objectif pédagogique et la rigueur méthodologique du projet. Un agent qui les viole livrera un résultat techniquement correct mais qui **tue l'objectif**.

## Éval fait-main — INTERDICTION de libs d'éval

**NE PAS** importer ou utiliser :
- `pytrec_eval` / `pytrec-eval`
- `beir.retrieval.evaluation`
- `sentence_transformers.evaluation`
- toute lib qui calcule Recall/nDCG/MRR à ta place

Les 6 métriques (Recall@{1,5,10,100} + nDCG@10 + MRR) doivent être **implémentées manuellement**.
C'est le coeur pédagogique du projet (cadrage §2.6).
Ref: `rag-eval-scifact-etape2-decisions.md:82`

## Dense-seul v1 — measure before levers

v1 est **volontairement minimale** : embedding dense + cosinus, c'est tout.

**NE PAS ajouter en v1 :**
- BM25 ou retrieval lexical
- Fusion RRF (multi-retriever)
- Reranking (cross-encoder ou autre)
- LLM de génération (mode RAG complet)
- Chunking sous-document

Tout levier est **hors scope v1** et gated sur l'error analysis des résultats v1.
On mesure d'abord, on améliore ensuite — jamais l'inverse.
Ref: `rag-eval-scifact-etape2-decisions.md:90-91`

## Ollama uniquement — zéro API payante

**NE PAS** appeler d'API LLM payante :
- OpenAI (GPT-*)
- Anthropic (Claude)
- Google (Gemini)
- Cohere, Mistral API, etc.

Si un LLM est nécessaire (post-v1 : génération, LLM-as-judge), utiliser **Ollama** (local uniquement).
Invariant transversal absolu du projet, non négociable.
