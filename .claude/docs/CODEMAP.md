# CODEMAP — RAG-Eval SciFact
> Cartographie décisionnelle. État : fin d'étape 2 (pré-implémentation).
> Source : `rag-eval-scifact-etape2-decisions.md`.
> Générée le 2026-07-01.
>
> NOTE : Ce projet est en phase **pré-code** (étape 2 close, étape 3 à démarrer).
> La carte pointe vers les **décisions gelées** qui structureront l'implémentation.
> Le code source n'existe pas encore.

---

## Architecture générale

**Pipeline de bout en bout** :
- **Ingestion** : corpus SciFact (5183 docs) → embedding → indexation ChromaDB
- **Retrieval** : requête → embedding → similarité cosinus → top-100 classés
- **Évaluation** : Recall@k + nDCG + MRR vs qrels (harness fait-main)
- **Artefacts** : `RESULTS.md` (append-only) + `results/*.json` (par run)

Voir `rag-eval-scifact-etape2-decisions.md` (§1–5).

---

## Dataset / Corpus

**Source & faits mesurés** : `rag-eval-scifact-etape2-decisions.md` (§1)

- **Taille corpus** : 5183 documents (BEIR SciFact complet)
  - Longueur médiane : 316 tokens (MiniLM tokenizer)
  - Longueur p90 : 502 tokens, p95 : 567, max : 1939
  - **71 % des docs > 256 tokens** (seuil MiniLM par défaut)

- **Qrels (docs pertinents par requête)** :
  - Split `test` : 300 requêtes, 92,3 % avec 1 seul doc pertinent
  - Split `train` : 809 requêtes, 90,5 % avec 1 seul doc pertinent
  - Tâche : retrieval du 1 unique doc pour ~9 requêtes sur 10

**Décision d'unité** : Le document entier (`title + text`), pas de chunking sous-document (`rag-eval-scifact-etape2-decisions.md:31–36`).
- Rationale : Qrels définis au niveau document ; chunking cassant l'alignement + la comparabilité leaderboard.
- Troncature 256 tokens = **dette explicite v1** (baseline BEIR), notée comme levier à débloquer post-v1.

---

## Embedding & Indexation

**Modèle & config v1** : `rag-eval-scifact-etape2-decisions.md` (§2, §6)

- **Modèle** : `all-MiniLM-L6-v2`
  - Dimension : 384
  - Fenêtre max : 256 tokens (tronqué, pas échappé, **dette acceptée**)
  - Entraîné pour **similarité cosinus** (critique)

- **Base de données vectorielle** : ChromaDB
  - **Espace de similarité : cosinus (✋ VERIFY avant tout run)** — pas L2 par défaut
  - Indexation : tous les 5183 docs (pas filtré par qrels) — ✋ VERIFY

**Rationale de v1** : Standard BEIR pour sentence-transformers ; baseline comparable, pas cassée d'emblée. Levier troncature (512 / autre modèle / chunking) gated sur error analysis v1 de retrieval (pas supposé).

---

## Retrieval

**Similitude & ranking** : `rag-eval-scifact-etape2-decisions.md` (§5, §6)

- **Similarité** : cosinus entre vecteur requête et corpus indexé
  - Top-100 ramenés = profondeur de log (+ récupération scores de similarité)
  - Uniquement top-100 : au-delà, rien n'est mesuré → rien n'est stocké

- **Protocole** : zero-shot retrieval sur split `test` (300 requêtes, ~300 qrels) + sanity check optionnel sur `train` (809 requêtes).
  - Corpus indexé : 5183 docs (TOUJOURS, même pour `test`) — protocole BEIR, distracteurs = tâche

---

## Métriques & Évaluation

**Harness d'éval (fait-main)** : `rag-eval-scifact-etape2-decisions.md` (§3)

**Jeu de sonde (k-de-mesure, pluriel)** :
- `Recall@1` : juste du premier coup ? (sensé pour 92 % des requêtes à 1 doc)
- `Recall@5` : dans le budget RAG réel ?
- `Recall@10` : filet + comparable (standard)
- `Recall@100` : juge de la dette de troncature (docs >256 vs ≤256)

**Métriques agrégées (pour leaderboard / reporting)** :
- `nDCG@10` : bien classé ? (0–1) — **métrique BEIR leaderboard** (audience externe)
- `MRR` : rang moyen — **lisible pour error analysis** (audience interne)

**Calcul** : À implémenter soi-même (pas de lib clé-en-main type `pytrec_eval`) — ✋ VERIFY avant commit. Voir `rag-eval-scifact-etape2-decisions.md:82`.

---

## Artefacts & Versioning

**Format de résultats** : `rag-eval-scifact-etape2-decisions.md` (§4)

**Niveau run** (une fois, carte d'identité) :
- `version` : tag git (ex. `v1-dense`)
- `date` : horodatage run
- `dataset_hash` : garantit que deux runs portent sur les mêmes données
- **Config embedding** :
  - `model` : `all-MiniLM-L6-v2`
  - `max_seq_length` : 256 (cible v1) → 512 (levier post-v1)
  - `dim` : 384
- Métriques agrégées (Recall@k, nDCG@10, MRR)

**Niveau requête** (un objet par requête, cœur error analysis) :
- `query_id` + texte de la requête
- **Docs attendus** : `_id` des qrels **avec leur longueur en tokens** ← corrèle échec ↔ troncature
- **Top-100 ramenés** : `_id` + rang + score de similarité
- Métriques de cette requête (found@k, rang du bon doc)

**Versioning** :
- `RESULTS.md` : append-only (log historique des runs, une ligne = un run)
- `results/*.json` : un fichier JSON par run (détail complet pour replay/debug)
- Tous les deux commités avec le même tag git (ex. `v1-dense`)
- **✋ Run non commité = run inexistant** (critère de vérification §6)

---

## Décisions gelées (fin étape 2)

Voir `rag-eval-scifact-etape2-decisions.md` pour détail complet.

**Nœuds tranchés (passent à l'implémentation)** :
- A. Split d'éval : `test` (300 requêtes), corpus complet 5183 docs
- B. Corpus : 5183 docs entiers, pas de chunking v1
- C. Similarité : **cosinus explicite** (✋ Vérifier ChromaDB + normalisation vecteurs)
- D. Troncature 256 : **dette acceptée** (MiniLM baseline, comparable BEIR)
- E. Harness : 6 métriques custom (Recall@{1,5,10,100} + nDCG@10 + MRR)
- F. Artefacts : `RESULTS.md` + `results/*.json`, tagués+commités

**Leviers pré-enregistrés (hors v1, activés si error analysis les désigne)** :
- BM25 (reranking léger)
- Fusion RRF (multi-retriever)
- Reranking dense
- LLM de génération (mode RAG full)
- Sortie de dette troncature (512 / autre modèle long-context / chunking for embedding)
- Frontend / Inspector (visualisation)

Backlog troncature (ordre croissant coût) :
1. `max_seq_length` 256 → 512 (une ligne ; 71 % → 8,8 % troncature)
2. Autre modèle embedding (long-context, ex. Jina embeddings)
3. Chunk-for-embedding + remap chunk→doc (plus lourd)

---

## Critères ✋ VERIFY pour l'implémentation

Avant tout commit/run :
- ✋ **Espace de similarité = cosinus**, pas L2 (ChromaDB `hnsw:space = cosine` ou normalisation vecteurs MiniLM)
- ✋ **Corpus indexé = 5183 docs**, pas sous-ensemble filtré par qrels
- ✋ **Troncature 256 notée comme dette** dans `RESULTS.md` (pas silencieuse)
- ✋ **Métriques calculées à la main**, pas déléguées à lib clé-en-main
- ✋ **Run non commité = run inexistant** : `RESULTS.md` + `results/*.json` produits et tagués ensemble

Voir `rag-eval-scifact-etape2-decisions.md:100–106`.

---

## Entrées de l'étape 3

- `rag-eval-scifact-etape2-decisions.md` (ce document : faits + décisions figées)
- Cadrage invariants (non sur disque, mémoriser ou récupérer oral)
- Chat neuf pour l'implémentation (pas d'historique étape 2)

**Nœuds d'implémentation restants** (hors dataset) :
- Un seul `IMPLEMENTATION.md` ou séparé (éval / embed) ? (décision de séquençage)
- Contenu exact du `.claude/` encodant les 5 critères ✋ (structure rules)

---

## Absence de code source

Ce projet est actuellement **sans implémentation** (étape 2 close, code lancé à l'étape 3).
- Pas de `src/`, `lib/`, `tests/`
- Pas de `requirements.txt`, `setup.py`, `pyproject.toml`
- Pas de `main.py` ou point d'entrée

Tous les fichiers `.py` seront créés à l'étape 3.
