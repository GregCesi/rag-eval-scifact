# CODEMAP — RAG-Eval SciFact

> Carte de retrieval. Générée le 2026-08-04.
> Pointeurs vers le code réel. Ne pas recopier le code. Régénérable — ne pas éditer à la main.

---

## Architecture générale

**Pipeline mono-passage, trois étapes :**
1. **Ingestion** (`rag_eval_scifact/ingest.py:54`) : corpus SciFact (5183 docs) → embeddings MiniLM (256 tokens) → ChromaDB (espace cosinus)
2. **Retrieval** (`rag_eval_scifact/retrieve.py:80`) : requêtes test (300) → embeddings → top-100 brute-force numpy (exact, pas HNSW approximé)
3. **Évaluation** (`rag_eval_scifact/run_eval.py:22`) : 6 métriques fait-main (Recall@{1,5,10,100} + nDCG@10 + MRR) → JSON + RESULTS.md

**Stack :** Python 3.11+ · ChromaDB (PersistentClient) · sentence-transformers (all-MiniLM-L6-v2, 384d) · numpy (similarité cosinus brute-force)

---

## Ingestion & Stockage

### Charger et embedder le corpus

- `ingest.py:54–138` — Fonction `ingest()` — Pipeline complet :
  - Charge `corpus.jsonl` (5183 docs) depuis `data/scifact/`
  - Concatène `title + " " + text` par doc (ligne 62)
  - Calcule `token_count` avec `AutoTokenizer.encode(..., add_special_tokens=True)` (ligne 70), rapporte taux troncature (71 % au-delà de 256 tokens)
  - Embedde avec `SentenceTransformer.encode()` (ligne 80), batch_size=64, max_seq_length=256
  - Calcule `dataset_hash` (SHA-256 du fichier corpus.jsonl brut, ligne 85)
  - Crée/indexe ChromaDB avec espace cosinus (ligne 90–124)

- **Constantes config :** `ingest.py:24–29`
  - `CORPUS_PATH = "data/scifact/corpus.jsonl"`
  - `MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`
  - `MAX_SEQ_LENGTH = 256` (→ 71 % des docs tronqués)
  - `CHROMA_DIR = Path("chroma_data")`
  - `COLLECTION_NAME = "scifact_v1"`
  - `EXPECTED_COUNT = 5183` (assertion de compte)

- **Fonction `compute_dataset_hash(corpus_path)`** (ligne 32) — SHA-256 du fichier brut (format `"sha256:{hexdigest}"`), immuable entre runs, guarantit reproducibilité

- **Fonction `load_corpus(corpus_path)`** (ligne 41) — Parse corpus.jsonl ligne par ligne (pas via datasets HF), retourne liste de `{"_id": str, "title": str, "text": str}`

### ChromaDB indexation (critiques)

- **Collection création avec espace cosinus :** `ingest.py:88–109`
  - Supprime collection existante avant création (ligne 95) — **CRUCIAL** : `hnsw:space` n'est configuré qu'**à la création**, non modifiable après
  - Crée avec `metadata={"hnsw:space": "cosine", "dataset_hash": dataset_hash}` (ligne 98–104)
  - **Vérifie espace = cosinus** avant indexation (ligne 107–109) — **gate critique** pour baseline correcte (MiniLM entraîné cosinus, L2 par défaut de ChromaDB classerait mal silencieusement)

- **Batch indexing :** `ingest.py:111–124` — BATCH_SIZE=1000, avec metadatas incluant `title` et `token_count` (crucial pour error analysis troncature)

- **Invariants vérifiés :**
  - Corpus complet 5183 docs (ligne 128)
  - Espace cosinus (ligne 107–109)
  - Token count avec `add_special_tokens=True` (ligne 68–72, identique mesure étape 2)

---

## Retrieval Dense

### Charger requêtes & qrels

- `retrieve.py:45–77` — Fonction `load_test_queries(queries_path, qrels_path)` — Charge requêtes depuis `queries.jsonl`, filtrées par split test (via qrels), charge qrels depuis `test.tsv` (colonnes `query-id`, `corpus-id`). Retourne `(queries_list, {query_id: {pertinent_doc_ids}})`

### Retrieval brute-force exact

- `retrieve.py:80–152` — Fonction `retrieve()` — Pipeline complet :
  - Charge requêtes test (300) + qrels (ligne 89–91)
  - Récupère tous les embeddings docs depuis ChromaDB avec `.get(include=["embeddings"])` (ligne 101–104)
  - Relu `dataset_hash` depuis `collection.metadata` (ligne 98)
  - Embedde requêtes test avec SentenceTransformer (ligne 108–113)
  - **Calcul cosinus exact (pas HNSW approximé)** (ligne 115–135) :
    - L2-normalise requêtes ET docs : `embeddings / norm` (ligne 117–121)
    - Dot-product entre normalized queries et docs (ligne 124) → matrice similarité (300 × 5183)
    - Pour chaque requête : `argpartition` top-100 puis tri décroissant (ligne 134)
  - Retourne `(results: list[RetrievalResult], qrels, dataset_hash)` (ligne 145–152)

- **Constantes :** `retrieve.py:24–30`
  - `QUERIES_PATH = "data/scifact/queries.jsonl"`
  - `QRELS_PATH = "data/scifact/qrels/test.tsv"` (split test, 300 requêtes)
  - `TOP_K = 100` (profondeur de retrieval)
  - `MAX_SEQ_LENGTH = 256` (identique à ingest)

### Data model

- **Type `RetrievalResult` (dataclass)** (ligne 34–42) — Contrat figé avec harness d'éval :
  - `query_id: str`
  - `query_text: str`
  - `retrieved: list[dict]` — Top-100 trié score décroissant (max en premier), chaque dict = `{"doc_id": str, "rank": int, "score": float}`

---

## Évaluation fait-main

### Métriques (implémentation manuelle, aucune lib d'éval)

- **`metrics.py:13–22` — `recall_at_k(retrieved_ids, relevant_ids, k) → float`**
  - Formule : |{pertinents ∩ top-k}| / |{pertinents}|
  - Edge case : relevant_ids vide → 0.0

- **`metrics.py:25–51` — `ndcg_at_k(retrieved_ids, relevant_ids, k) → float`**
  - DCG@k = Σᵢ₌₁^k (1 si doc[i] pertinent, 0 sinon) / log₂(i+1)
  - IDCG@k = DCG du classement parfait (min(|relevant|, k) docs pertinents aux premiers rangs)
  - nDCG@k = DCG / IDCG (0.0 si IDCG = 0)
  - Relevance binaire (pas de gradation)

- **`metrics.py:54–66` — `mrr(retrieved_ids, relevant_ids) → float`**
  - 1 / rang du premier doc pertinent
  - Retourne 0.0 si aucun pertinent dans le top-100 fourni

**Interdiction méthodologique :** `aucune import de pytrec_eval, beir.retrieval.evaluation, sentence_transformers.evaluation` — 6 métriques implémentées from scratch

### Tests métriques

- **`tests/test_metrics.py:21–155`** — Suite pytest complète :
  - Fixtures : 5 cas-tests (trivial rang-1, rang-2, rang-3, absent hors top-100, multi-docs 2 docs)
  - Tests Recall@k → `test_metrics.py:67–88` (4 valeurs de k testées)
  - Tests nDCG@10 → `test_metrics.py:95–119` (avec oracles calculés à la main)
  - Tests MRR → `test_metrics.py:126–142`
  - Edge case IDCG=0 → `test_metrics.py:150–154` (pas de division par zéro)
  - Tolérance : 1e-4 sur comparaisons flottantes (TOL ligne 18)

### Orchestration run complet

- **`run_eval.py:22–77` — Fonction `run_eval()`** — Point d'entrée officiel : `python -m rag_eval_scifact.run_eval`
  - Appelle `retrieve()` → récupère 300 RetrievalResult + qrels + dataset_hash (ligne 27)
  - Boucle sur 300 requêtes, agrège 6 métriques (macro-average) (ligne 41–50)
  - Crée dict agg (somme par métrique) puis normalise par n=300 (ligne 52)
  - Génère artefacts JSON (ligne 56) + append RESULTS.md (ligne 58)
  - Affiche résumé console (ligne 61–72)
  - **Reminder de commit :** ligne 73

### Artefacts (JSON + RESULTS.md)

- **`run_output.py:28–49` — `compute_per_query_metrics(retrieved_ids, relevant_ids) → dict`**
  - Trouve `best_rank` = rang du premier doc pertinent (None si absent)
  - Retourne `{"found@1", "found@5", "found@10", "found@100" (bools), "best_rank" (int|None)}`

- **`run_output.py:52–60` — `get_token_counts(doc_ids) → dict[str, int]`**
  - Récupère depuis ChromaDB metadatas (clé `token_count`) — crucial pour correler échecs et troncature dans error analysis

- **`run_output.py:63–109` — `generate_run_json(results, qrels, metrics, dataset_hash, run_date) → dict`** — Assemble JSON complet :
  - **Niveau run (racine) :**
    - `version` = "v1-dense" (constante ligne 24)
    - `date` = ISO format (ligne 100)
    - `dataset_hash` = relu depuis retrieve, garantit reproducibilité (ligne 101)
    - `config` = `{model: "all-MiniLM-L6-v2", max_seq_length: 256, dim: 384}` (ligne 102–106)
    - `metrics` = dict des 6 métriques macro-average (ligne 107)

  - **Niveau requête (array `queries`, 300 objets) :**
    - `query_id`, `query_text` (ligne 91–92)
    - `expected_docs` = array de `{doc_id, token_count}` pour tous docs pertinents (qrels), trié par _id (ligne 85–88)
    - `retrieved_top100` = le champ `.retrieved` de RetrievalResult (100 dicts avec doc_id, rank, score)
    - `per_query_metrics` = résultat de `compute_per_query_metrics()` (found@k, best_rank)

- **`run_output.py:112–120` — `write_run_json(run_data) → Path`**
  - Écrit `results/v1-dense-{date ISO avec tirets}.json` (format ISO date remplace ":" par "-")
  - indent=2, UTF-8, ensure_ascii=False

- **`run_output.py:123–150` — `append_results_md(metrics) → None`** — **Append-only strict**
  - Crée header si fichier n'existe pas (ligne 125–127)
  - Ajoute ligne au tableau : version | date (YYYY-MM-DD) | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR | dette (max_seq=256) | note d'analyse
  - Mode append strict (`.a`, ligne 148–149), jamais réécriture

### Versioning des runs

**Invariant :** `.claude/rules/versioning.md` — **Run non commité = run inexistant**. Chaque run doit :
1. Ajouter ligne dans `RESULTS.md` (append-only, jamais réécrire)
2. Écrire fichier `results/v1-dense-{date ISO}.json` avec détail complet
3. Commiter **les deux** sous même tag git (ex. `v1-dense`)

**Historique actuel :** `RESULTS.md:3–4` — Deux runs v1-dense exécutés 2026-07-01 : R@1=0.4823, R@5=0.7379, R@10=0.7833, R@100=0.925, nDCG@10=0.6451, MRR=0.6110

---

## Dataset & Données

### Sources brutes BEIR SciFact

- **Corpus :** `data/scifact/corpus.jsonl` — 5183 documents (title + text)
- **Requêtes :** `data/scifact/queries.jsonl` — 1109 requêtes total, ID + text
- **Qrels test :** `data/scifact/qrels/test.tsv` — 300 requêtes × 1.13 docs pertinents moyenne (92.3 % à 1 seul doc)

### Invariants dataset (`.claude/rules/invariants.md`)

- **Corpus indexé = 5183 docs entiers toujours**, quel que soit le split (protocole BEIR ; les distracteurs font partie de la tâche)
- **Split d'éval = test** (300 requêtes, 92.3 % à 1 doc pertinent) — retrieval zero-shot, comparable leaderboard
- **Troncature MiniLM = 256 tokens**, 71 % des docs dépassent ce seuil → **dette acceptée pour v1, notée explicite dans RESULTS.md** (ligne 144 run_output.py)

### Résultats (versionnés, append-only)

- **JSON courant :** `results/v1-dense-2026-07-01T17-19-59.json` et `results/v1-dense-2026-07-01T17-35-18.json` — Runs complets horodatés, 300 requêtes, top-100 par requête avec scores cosinus brute-force
- **Historique :** `RESULTS.md` — Append-only, une ligne par run tagué
- **Bucketisation erreurs :** `results/v1-buckets.json` — Classification queries par profondeur d'échec (perfect, near_miss, deep_miss, miss_100)
- **Annotations :** `results/v1-annotations.json`, `results/v1-near_miss-auto-annotations.json`, etc. — Annotations manuelles/automatiques des erreurs

---

## Scripts post-run (error analysis et dashboard)

### `scripts/extract_error_analysis.py` — Bucketisation & annotation

- **Usage :** `python scripts/extract_error_analysis.py [results/v1-dense-*.json]` (défaut : dernier JSON)
- **Constantes :** mêmes que run (CORPUS_PATH, MODEL_NAME, MAX_TOKENS=256)
- **Fonction `classify_query(q: dict) → tuple[str, int|None, float|None]`** (ligne 31) — Classe requête par profondeur d'échec :
  - `"perfect"` si best_rank == 1
  - `"near_miss"` si 1 < best_rank ≤ 10
  - `"deep_miss"` si best_rank > 10
  - `"miss_100"` si best_rank est None (doc pertinent hors top-100)
- **Sorties :**
  - `results/v1-buckets.json` — `{query_id: {bucket, best_rank, score_at_best_rank}}`
  - `results/v1-error-analysis-miss100.md` — Fiche d'annotation markdown par requête miss_100

### `dashboard.py` — Exploration interactive Streamlit

- **Démarrage :** `streamlit run dashboard.py`
- **Cache functions :** load_run (JSON), load_corpus (corpus.jsonl), load_tokenizer, load_buckets (v1-buckets.json), load_annotations
- **Pages :**
  - Overview (ligne 185) : métriques agrégées, graphiques Plotly
  - Explore (ligne 259) : filtrage par query, affichage retrieval results
  - Errors (ligne 421) : bucketisation erreurs, zoom sur miss_100
- **Fonctions support :** split_at_truncation (troncature 256 tokens), export_query_markdown, render_query_detail (ligne 580), render_doc_text (ligne 622)

---

## Configuration & Invariants

### Stack & Dépendances

- **`pyproject.toml:5–21`** :
  - Dependencies : `sentence-transformers`, `chromadb`, `numpy`
  - Dev : `pytest`, `ruff>=0.11.0`
  - Optional dashboard : `streamlit`, `plotly`
  - Python ≥ 3.11

### Constantes critiques

- **Embedding :** `ingest.py:27–28`, `retrieve.py:28–29`
  - Modèle : `sentence-transformers/all-MiniLM-L6-v2` (384d)
  - max_seq_length : 256 tokens (→ 71 % troncature, dette v1)
  - Batch size : 64

- **Corpus & paths :** `ingest.py:24–26`, `retrieve.py:24–26`
  - Chemins : `data/scifact/{corpus,queries}.jsonl`, `data/scifact/qrels/test.tsv`
  - ChromaDB : `chroma_data/` (PersistentClient)
  - Collection : `scifact_v1`

- **Retrieval :** `retrieve.py:30`
  - TOP_K : 100 (profondeur de log)
  - Similarité : cosinus **exact** (numpy brute-force, pas HNSW)

- **Output :** `run_output.py:24–25`
  - VERSION : "v1-dense"
  - MAX_SEQ_LENGTH : 256 (noté dette dans RESULTS.md)

### Checklist pré-run (gates critiques)

→ `.claude/rules/invariants.md` :

1. **Espace cosinus vérifié** : `collection.metadata["hnsw:space"] == "cosine"` (pas L2 par défaut ChromaDB) → `ingest.py:107–109`
   - MiniLM entraîné cosinus ; L2 fausserait silencieusement la baseline
   - Non négociable

2. **Corpus complet 5183** : assertion → `ingest.py:128`

3. **Troncature notée dette** : `max_seq=256` enregistré explicitement dans RESULTS.md → `run_output.py:144`
   - 71 % des docs dépassent ce seuil
   - Acceptée pour v1 (baseline comparable BEIR), non silencieuse

---

## Points d'entrée

| Commande | Fichier | Rôle |
|----------|---------|------|
| `python -m rag_eval_scifact.ingest` | `ingest.py:137–138` | Embedding corpus + indexation ChromaDB (à faire 1 fois) |
| `python -m rag_eval_scifact.retrieve` | `retrieve.py:155–168` | Retrieval test seul (affiche résumé rapide, pas les métriques) |
| `python -m rag_eval_scifact.run_eval` | `run_eval.py:76–77` | **Point d'entrée officiel** — Run d'éval complet (retrieval + métriques + artefacts) |
| `pytest` | `tests/test_metrics.py` | Tests unitaires des 6 métriques |
| `python scripts/extract_error_analysis.py [JSON]` | `scripts/extract_error_analysis.py:137` | Post-traitement : bucketisation + error analysis |
| `streamlit run dashboard.py` | `dashboard.py:141` | Dashboard exploration interactive (charts + détail per-query) |

---

## Conventions & Méthodologie

### Éval fait-main (interdiction libs d'éval)

**Règle :** `.claude/rules/methodologie.md` — Aucune lib d'éval autorisée :
- Interdit : `pytrec_eval`, `pytrec-eval`, `beir.retrieval.evaluation`, `sentence_transformers.evaluation`
- Les 6 métriques calculées soi-même, implémentées dans `metrics.py` (recall, ndcg, mrr)
- C'est le **coeur pédagogique** du projet

### Dense-seul v1 (measure before levers)

**Règle :** `.claude/rules/methodologie.md` — v1 volontairement minimale :
- **DANS v1 :** embedding dense (MiniLM) + cosinus exact + 6 métriques made-in
- **HORS v1 :** BM25, RRF, reranking, LLM génération, chunk-for-embedding, sortie de dette troncature
- Tous leviers **pré-enregistrés, gated sur error analysis** v1
- On mesure d'abord, on améliore ensuite — jamais l'inverse

### Ollama uniquement (zéro API payante)

**Règle :** `.claude/rules/methodologie.md` — Aucune API LLM payante :
- Interdit : OpenAI, Anthropic, Google Gemini, Cohere, Mistral API
- Si LLM nécessaire (post-v1 : génération, LLM-as-judge) → **Ollama local** uniquement
- Invariant transversal absolu, non négociable

---

## Invariants détectés

- **Pas de tests intégration** : `tests/test_metrics.py` couvre métriques unitaires seulement. Pas de test end-to-end (run complet + JSON output).
- **Pas de CLI structurée** : entrées par modification de constantes dans les fichiers source (chemins, model, max_seq_length). Convention non-standard pour projet de cette taille.
- **Pas de logging persistant** : affichage console via `print()` uniquement. Pas d'audit trail persistant des exécutions (seulement les JSON artefacts).
- **Retrieval exact brute-force** : numpy @ numpy.T pour chaque run. Pas de cache embeddings, pas d'optimisation HNSW approximée (délibéré pour v1 : correctness première).
- **ChromaDB espace cosinus critique** : dépend de la config `hnsw:space="cosine"` **à la création**. Non modifiable après. Vérification explicite avant chaque indexation.
- **Token count stocké par doc** : crucial pour error analysis troncature. Métadonnée ChromaDB, passée en expected_docs du JSON (ligne 86 run_output.py).

