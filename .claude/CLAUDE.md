# RAG-Eval SciFact

Benchmark d'évaluation pour retrieval dense sur le dataset BEIR SciFact (5183 docs scientifiques). Pipeline mono-passage : ingestion → retrieval cosinus → éval fait-main (Recall@k + nDCG@10 + MRR).

## État

Étape 2 close (décisions dataset gelées). Étape 3 : implémentation.

## Stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`, 384d, 256 tokens) · Ollama (LLM local uniquement)

## Structure du projet

Voir `.claude/docs/CODEMAP.md` pour la carte détaillée.
Source de vérité décisions : `rag-eval-scifact-etape2-decisions.md`

## Commandes courantes

*(à compléter lors de l'implémentation)*

## Conventions non-standard

- **Run versioning strict** : tout run doit être commité (`RESULTS.md` append-only + `results/*.json` taggés). Run non commité = n'existe pas.
- **Éval fait-main** : métriques calculées soi-même, interdiction d'importer des libs d'éval (`pytrec_eval`, `beir.retrieval.evaluation`, etc.).
- **Dense-seul v1** : pas de BM25/RRF/rerank/génération. Leviers post-v1 gated sur error analysis.
- **Ollama uniquement** : zéro API payante (OpenAI, Anthropic, etc.). LLM local seulement.
