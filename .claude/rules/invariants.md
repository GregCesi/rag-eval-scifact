---
description: Invariants config/dataset du pipeline RAG-Eval SciFact v1
paths: ["**/*.py", "RESULTS.md"]
---

# Invariants config/dataset

Ces invariants sont des **gates** : vérifier AVANT chaque run/commit.

## Checklist pré-run

- [ ] **Espace de similarité = cosinus, pas L2**
  ChromaDB utilise L2 par défaut. Configurer `hnsw:space = cosine` OU normaliser les vecteurs avant indexation.
  MiniLM est entraîné pour le cosinus — L2 fausse silencieusement la baseline.
  Ref: `rag-eval-scifact-etape2-decisions.md:88` (Noeud C)

- [ ] **Corpus indexé = 5183 docs entiers**
  Toujours indexer les 5183 documents du corpus SciFact, même en évaluant sur split `test` (300 requêtes).
  Protocole BEIR : les distracteurs font partie de la tâche. Jamais de filtrage par qrels.
  Ref: `rag-eval-scifact-etape2-decisions.md:87` (Noeud B)

- [ ] **Troncature = dette explicite dans RESULTS.md**
  MiniLM tronque silencieusement à 256 tokens. 71 % des docs dépassent ce seuil.
  Cette troncature est ACCEPTÉE pour v1 (baseline BEIR comparable), mais doit être
  **notée explicitement comme dette** dans `RESULTS.md` à chaque run. Pas silencieuse.
  En phase labo : chaque run note le `max_seq_length` effectif et la part de documents (ou de chunks) tronqués.
  Ref: `rag-eval-scifact-etape2-decisions.md:32-33`
