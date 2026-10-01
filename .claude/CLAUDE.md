# RAG-Eval SciFact

Banc d'évaluation du retrieval sur le dataset BEIR SciFact (5183 docs scientifiques). v1 : pipeline dense mono-passage (ingestion → retrieval cosinus → éval fait-main : Recall@k + nDCG@10 + MRR). Phase labo : stratégies déclarées en configs Hydra, comparées en grille, runs tracés dans MLflow. L'éval reste fait-main.

## État

v1-dense close (run du 2026-07-01). Error analysis v1 faite (4 buckets, annotations manuelles et auto via `/annotate`) ; aucun levier désigné.

Phase en cours (depuis le 2026-10-01) : **labo multi-stratégies**. Première campagne `v2-grid` : 34 runs (17 retrievers × rerank oui/non). Pilotage dans Notion : TCK-258 à TCK-264 (socle + briques), TCK-229 (lancement de la grille).

## Stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`, 384d, 256 tokens) · Ollama (LLM local uniquement)

Labo (à installer par le socle) : Hydra (configs de stratégie) · MLflow (suivi des runs, local) · `Qwen/Qwen3-Embedding-0.6B` (modèle long contexte)

## Structure du projet

Voir `.claude/docs/CODEMAP.md` pour la carte détaillée.
Source de vérité décisions : `rag-eval-scifact-etape2-decisions.md`

## Commandes courantes

Voir le tableau « Commandes » du `README.md` (ingestion, run d'évaluation, tests, dashboard). Les commandes de campagne arrivent avec le socle (TCK-258).

## Conventions non-standard

- **Run versioning strict** : tout run doit être commité (une ligne `RESULTS.md` append-only + `results/{campagne}/{run}.json.gz`), un tag git par campagne. Run non commité = n'existe pas. Voir `rules/versioning.md`.
- **Éval fait-main** : métriques, fusion RRF et tests statistiques codés soi-même, interdiction d'importer des libs d'éval (`pytrec_eval`, `beir.retrieval.evaluation`, `ranx`, etc.).
- **Leviers par campagne** : BM25, fusion, rerank, chunking et modèle long n'entrent que comme dimensions de config d'une campagne, avec une prédiction écrite avant le lancement. Pas de génération LLM dans la campagne `v2-grid`. Voir `rules/methodologie.md`.
- **Ollama uniquement** : zéro API payante (OpenAI, Anthropic, etc.). LLM local seulement.
