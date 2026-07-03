# RAG-Eval SciFact

Harness d'évaluation de retrieval dense sur le dataset BEIR SciFact (5183 docs scientifiques). L'éval est le livrable : métriques fait-main, zéro lib d'éval.

## Objectif

Mesurer la qualité d'un retrieval dense mono-passage (embedding cosinus) sur SciFact, avec 6 métriques implémentées manuellement : Recall@{1,5,10,100}, nDCG@10, MRR.

## Stack

- Python 3.11+
- `sentence-transformers` (modèle `all-MiniLM-L6-v2`, 384d, 256 tokens)
- ChromaDB (store vectoriel, espace cosinus)
- numpy (similarité cosinus exacte)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Les données SciFact doivent se trouver dans `data/scifact/` (fichiers BEIR bruts : `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`).

## Commandes

### 1. Ingestion (embedding + indexation ChromaDB)

```bash
python -m rag_eval_scifact.ingest
```

Charge les 5183 docs, les embedde avec MiniLM, et les indexe dans une collection ChromaDB en espace cosinus. A faire une seule fois (ou pour recréer la collection).

Produit le dossier `chroma_data/`.

### 2. Run d'évaluation complet

```bash
python -m rag_eval_scifact.run_eval
```

Enchaîne : retrieval dense (cosinus exact sur les 300 requêtes test) -> calcul des 6 métriques -> génération des artefacts.

Produit :
- `results/{tag}-{date}.json` (détail complet par requête)
- Une ligne dans `RESULTS.md` (append-only)

### 3. Tests

```bash
pytest
```

Tests unitaires des métriques (recall, nDCG, MRR).

## Structure

```
rag_eval_scifact/
  ingest.py       # Ingestion corpus -> ChromaDB
  retrieve.py     # Retrieval dense cosinus exact (top-100)
  metrics.py      # Recall@k, nDCG@10, MRR (fait-main)
  run_output.py   # Génération artefacts (JSON + RESULTS.md)
  run_eval.py     # Orchestrateur : retrieval -> métriques -> artefacts
tests/
  test_metrics.py # Tests unitaires métriques
data/scifact/     # Données BEIR brutes (non versionnées)
chroma_data/      # Index ChromaDB (non versionné)
results/          # JSON détaillé par run (versionné)
RESULTS.md        # Historique des runs (append-only, versionné)
```

## Conventions

- **Run non commité = run inexistant** : `RESULTS.md` + `results/*.json` doivent etre commités ensemble.
- **Troncature 256 tokens** : dette acceptee pour v1 (71% des docs depassent), notee explicitement dans chaque run.
- **Dense-seul v1** : pas de BM25, reranking, RRF, ou generation. Leviers gates sur error analysis.
- **Ollama uniquement** : zero API payante.
