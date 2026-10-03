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
- `ranx` (y compris `ranx.fuse`), `ir_measures`, `trectools`
- toute lib qui calcule Recall/nDCG/MRR à ta place

Les 6 métriques (Recall@{1,5,10,100} + nDCG@10 + MRR) doivent être **implémentées manuellement**.
En phase labo, même règle pour la **fusion RRF** et pour le **test statistique apparié** entre deux runs : codés à la main et testés. `numpy` reste autorisé pour le calcul.
C'est le coeur pédagogique du projet (cadrage §2.6).
Ref: `rag-eval-scifact-etape2-decisions.md:82`

## Leviers — par campagne, jamais par intuition

v1 (dense seul) est close et sert de référence. Depuis le 2026-10-01, les leviers pré-enregistrés
(`rag-eval-scifact-etape2-decisions.md` §5 : BM25, fusion RRF, reranking, fenêtre longue / autre modèle,
chunking) **entrent par une campagne**, pas un par un.

**Règles d'une campagne :**
- **Prédiction avant lancement — OBLIGATOIRE.** Avant le premier run, `results/{campagne}/PREDICTION.md`
  est écrit et commité : levier attendu gagnant, buckets v1 qu'il doit corriger (near_miss / deep_miss /
  miss_100), ordre de grandeur attendu sur nDCG@10. Il s'appuie sur l'error analysis v1. Pas de prédiction
  = pas de campagne. La campagne vérifie la prédiction, elle ne la remplace pas.
- **Un levier = une dimension de config Hydra**, jamais un comportement codé en dur dans le pipeline.
- **Fusion** : si des poids sont optimisés, ils le sont sur le split `train`, jamais sur `test`.
- **Chunking sous-document** : toujours suivi d'un regroupement chunk → document avant le calcul des
  métriques (qrels au niveau document).
- **Lecture** : avec plusieurs dizaines de runs, des écarts « significatifs » apparaissent par hasard.
  On lit les gros écarts, pas les 0,01.

**NE PAS ajouter dans la campagne `v2-grid` :**
- LLM de génération (mode RAG complet), LLM-as-judge
- réécriture de requête, HyDE

Ref: décision du 2026-10-01, entrée Journal Notion « rag-eval-scifact sort de pause : le banc devient un labo
multi-stratégies tracé dans MLflow ».

## Ollama uniquement — zéro API payante

**NE PAS** appeler d'API LLM payante :
- OpenAI (GPT-*)
- Anthropic (Claude)
- Google (Gemini)
- Cohere, Mistral API, etc.

Si un LLM est nécessaire (post-v1 : génération, LLM-as-judge), utiliser **Ollama** (local uniquement).
Attention : les juges LLM de MLflow (`mlflow.genai`) appellent OpenAI par défaut — les configurer sur Ollama.
Invariant transversal absolu du projet, non négociable.

Exception unique : le juge de référence de la campagne v3-juge appelle Claude par le mode non interactif de Claude Code, sur l'abonnement, sans clé d'API. Ses jugements sont étiquetés comme tels. Aucun autre code du dépôt n'appelle un modèle distant.
