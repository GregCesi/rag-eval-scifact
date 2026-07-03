# CODEMAP — RAG-Eval SciFact

> Carte de retrieval. Générée par codemap-builder le 2026-07-01.
> Pointeurs vers le code réel. Ne pas recopier le code. Régénérable — ne pas éditer à la main.

---

## Architecture générale

**Pipeline mono-passage, trois étapes :**
1. **Ingestion** (`rag_eval_scifact/ingest.py:54`) : corpus SciFact (5183 docs) → embeddings MiniLM (256 tokens) → ChromaDB (espace cosinus)
2. **Retrieval** (`rag_eval_scifact/retrieve.py:80`) : requêtes test (300) → embeddings → top-100 brute-force numpy (exact, pas HNSW approximé)
3. **Évaluation** (`rag_eval_scifact/run_eval.py:22`) : 6 métriques fait-main (Recall@{1,5,10,100} + nDCG@10 + MRR) → JSON + RESULTS.md

---

## Ingestion & Stockage

### Charger et embedder le corpus
- `ingest.py:54–138` — Fonction `ingest()` — Pipeline complet : charge `corpus.jsonl` (5183 docs), concatène `title + " " + text` par doc, calcule token_count (MiniLM exact, avec spéciaux), embedde avec `all-MiniLM-L6-v2` (max_seq_length=256, batch_size=64), calcule `dataset_hash` (SHA-256 des fichiers sources)
  - Constantes config : `CORPUS_PATH`, `CHROMA_DIR`, `COLLECTION_NAME`, `MODEL_NAME`, `MAX_SEQ_LENGTH=256`, `EXPECTED_COUNT=5183` → `ingest.py:24–29`
  - Fonction support `load_corpus(corpus_path)` → `ingest.py:41–51` — Charge depuis JSONL brut (pas via `datasets` HF)
  - Fonction support `compute_dataset_hash(corpus_path)` → `ingest.py:32–38` — SHA-256 déterministe des fichiers sources, stocké en metadata collection

### ChromaDB indexation
- Collection fraîche création avec `hnsw:space="cosine"` → `ingest.py:88–104` — **CRITIQUE** : `delete_collection` avant création si existe (hnsw:space n'est pas modifiable après création). Assertion de vérification de l'espace → `ingest.py:107–109`
- Batch indexing par 1000 docs → `ingest.py:111–124`
- Metadata par doc : `title`, `token_count` (pour error analysis troncature) → `ingest.py:119–122`

### Invariants vérifiés
- Corpus complet 5183 docs → `ingest.py:128`
- Espace cosinus (pas L2 par défaut) → `ingest.py:107`
- Token count avec `add_special_tokens=True` (identique inspection étape 2) → `ingest.py:68–72`

---

## Retrieval Dense

### Charger requêtes & qrels
- `retrieve.py:45–77` — Fonction `load_test_queries(queries_path, qrels_path)` — Charge requêtes depuis `queries.jsonl` (filtrées par test split via qrels), charge qrels depuis `test.tsv` (tab-delimited, colonnes `query-id`, `corpus-id`). Retourne lista queries + dict qrels indexé par query_id

### Retrieval brute-force exact
- `retrieve.py:80–152` — Fonction `retrieve()` — Pipeline complet :
  - Charge requêtes test (300) et qrels → `retrieve.py:89–91`
  - Récupère tous les embeddings docs depuis ChromaDB → `retrieve.py:94–104`
  - Embedde requêtes (MiniLM, max_seq_length=256) → `retrieve.py:107–113`
  - **Calcul cosinus exact** : L2-normalise requêtes + docs → dot product → top-100 par requête (numpy, pas HNSW) → `retrieve.py:115–135`
  - Retourne `results` (list[RetrievalResult]), `qrels`, `dataset_hash` → `retrieve.py:145–152`

### Data model
- `RetrievalResult` (dataclass) → `retrieve.py:34–42` — Contrat figé avec harness d'éval :
  - `query_id` : str
  - `query_text` : str
  - `retrieved` : list[dict] — Top-100 trié score décroissant, chaque dict = `{"doc_id": str, "rank": int, "score": float}`

---

## Évaluation fait-main

### Métriques (implémentation manuelle)
- `metrics.py:13–22` — `recall_at_k(retrieved_ids, relevant_ids, k)` — |{pertinents ∩ top-k}| / |{pertinents}|
- `metrics.py:25–51` — `ndcg_at_k(retrieved_ids, relevant_ids, k)` — DCG@k / IDCG@k (relevance binaire, log₂(i+1) pour rang i)
- `metrics.py:54–66` — `mrr(retrieved_ids, relevant_ids)` — 1/rang du premier doc pertinent (0.0 si absent)

**Invariant méthodologique** → `.claude/rules/methodologie.md` : **Aucune lib d'éval autorisée** (pytrec_eval, beir.retrieval.evaluation, etc.). Les 6 métriques sont implémentées soi-même.

### Tests métriques
- `tests/test_metrics.py:21–155` — Suite complète pytest :
  - Fixtures cas-tests : trivial (rang 1), rang 2, rang 3, absent (hors top-100), multi (2 docs)
  - Tests Recall@k → `test_metrics.py:67–88`
  - Tests nDCG@10 avec oracles calculés à la main → `test_metrics.py:95–119`
  - Tests MRR → `test_metrics.py:126–142`
  - Edge case : IDCG=0 (pas de division par zéro) → `test_metrics.py:150–154`

### Orchestration run complet
- `run_eval.py:22–77` — Fonction `run_eval()` — Point d'entrée unique (commande : `python -m rag_eval_scifact.run_eval`) :
  - Appelle `retrieve()` → récupère résultats, qrels, dataset_hash
  - Boucle sur 300 requêtes, agrège 6 métriques (macro-average) → `run_eval.py:29–52`
  - Génère artefacts JSON + append RESULTS.md → `run_eval.py:55–58`
  - Affiche résumé console → `run_eval.py:60–73`

### Artefacts (JSON + RESULTS.md)
- `run_output.py:28–49` — `compute_per_query_metrics(retrieved_ids, relevant_ids)` — Calcule `found@{1,5,10,100}` et `best_rank` par requête
- `run_output.py:52–60` — `get_token_counts(doc_ids)` — Récupère token_count depuis metadata ChromaDB (pour correler échecs et troncature)
- `run_output.py:63–109` — `generate_run_json(results, qrels, metrics, dataset_hash, run_date)` — Assemble JSON complet :
  - Niveau run : version, date ISO, dataset_hash, config (model, max_seq_length, dim), métriques agrégées
  - Niveau requête (300 objets) : query_id, query_text, expected_docs (avec token_count), retrieved_top100, per_query_metrics
- `run_output.py:112–120` — `write_run_json(run_data)` — Écrit `results/v1-dense-{date ISO}.json`
- `run_output.py:123–150` — `append_results_md(metrics)` — Ajoute ligne RESULTS.md (append-only strict), format : version | date | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR | dette (max_seq=256) | note d'analyse

### Versioning des runs
- Invariant → `.claude/rules/versioning.md` : **Run non commité = run inexistant**. Chaque run doit produire + commiter :
  1. Une ligne dans `RESULTS.md` (append-only, jamais réécrire)
  2. Un fichier `results/{tag}-{date}.json` avec détail complet
  3. Les deux sous le même tag git (ex. `v1-dense`)
- Ligne historique → `RESULTS.md:3` — Premier run v1-dense exécuté 2026-07-01 : R@1=0.4823, R@5=0.7379, R@10=0.7833, R@100=0.925, nDCG@10=0.6451, MRR=0.6110

---

## Dataset & Données

### Sources brutes BEIR SciFact
- Corpus : `data/scifact/corpus.jsonl` — 5183 documents (title + text)
- Requêtes : `data/scifact/queries.jsonl` — ID + text pour 1109 requêtes (dont 300 test)
- Qrels test : `data/scifact/qrels/test.tsv` — 300 requêtes × 1.13 docs pertinents en moyenne (92.3 % à 1 seul doc)

**Invariant dataset** → `.claude/rules/invariants.md` :
- Corpus indexé = 5183 docs entiers toujours (jamais filtré par qrels)
- Split d'éval = test (300 requêtes, 92.3 % à 1 doc pertinent)
- Troncature MiniLM = 256 tokens, 71 % des docs dépassent ce seuil → dette acceptée pour v1, notée explicite dans RESULTS.md

### Résultats
- JSON courant → `results/v1-dense-2026-07-01T17-19-59.json` — Run complet horodaté, 300 requêtes, top-100 par requête avec scores cosinus
- Historique → `RESULTS.md` — Append-only, une ligne par run tagué

---

## Configuration & Invariants

### Stack & Dépendances
- `pyproject.toml:5–21` — Dependencies : `sentence-transformers` (MiniLM), `chromadb` (storage cosinus), `numpy` (calcul cosinus brute-force). Dev : `pytest`
- Python ≥ 3.11

### Constantes critiques
- **Embedding** → `ingest.py:27–28`, `retrieve.py:28–29` :
  - Modèle : `sentence-transformers/all-MiniLM-L6-v2` (384d)
  - max_seq_length : 256 tokens (→ 71 % troncature, dette v1)
- **Corpus** → `ingest.py:24–26`, `retrieve.py:24–26` :
  - Chemins : `data/scifact/corpus.jsonl`, `data/scifact/queries.jsonl`, `data/scifact/qrels/test.tsv`
  - ChromaDB dir : `chroma_data/`
  - Collection name : `scifact_v1`
- **Retrieval** → `retrieve.py:30` :
  - TOP_K : 100 (profondeur de log)
- **Métriques** → `run_eval.py:32–39` :
  - 6 métriques clés : recall@{1,5,10,100}, nDCG@10, MRR

### Checklist pré-run (gates)
→ `.claude/rules/invariants.md` :
1. **Espace cosinus vérifié** : `collection.metadata["hnsw:space"] == "cosine"` (pas L2 par défaut) → `ingest.py:107–109`
2. **Corpus complet 5183** : vérification count → `ingest.py:128`
3. **Troncature notée dette** : `max_seq=256` enregistré dans RESULTS.md → `run_output.py:144`

---

## Points d'entrée

- **Ingestion** : `python -m rag_eval_scifact.ingest` ou `ingest.py:137–138`
- **Retrieval test** : `python -m rag_eval_scifact.retrieve` ou `retrieve.py:155–168` (affiche résumé rapide)
- **Run complet** : `python -m rag_eval_scifact.run_eval` (point d'entrée officiel, à commiter après)

---

## Invariants détectés

- **Pas de tests intégration** : `tests/test_metrics.py` couvre métriques unitaires seulement. Pas de test end-to-end (run complet + JSON output).
- **Pas de CLI structurée** : entrées par modification de constantes dans les fichiers (chemins, model, max_seq_length). Convention non-standard pour project de cette taille.
- **Pas de logging** : affichage console via `print()` uniquement. Pas d'audit trail persistant des exécutions.
- **Dense-seul v1** : volontairement minimale (embedding + cosinus). Aucun BM25/RRF/reranking/LLM de génération. Leviers hors scope v1, gated sur error analysis.
- **Éval fait-main strict** : Aucune dépendance à des libs d'éval (pytrec_eval, beir.retrieval.evaluation) — interdiction méthodologique explicite.
- **Ollama-only** : LLM local seulement (post-v1, si nécessaire). Zéro API payante.
