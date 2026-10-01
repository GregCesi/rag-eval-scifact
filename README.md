# RAG-Eval SciFact

Harness d'évaluation de retrieval dense sur le dataset BEIR SciFact (5183 docs scientifiques). L'éval est le livrable : métriques fait-main, zéro lib d'éval.

## Résultats — baseline v1-dense (2026-07-01)

| R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR |
|-----|-----|------|-------|---------|-----|
| 0.482 | 0.738 | 0.783 | 0.925 | 0.645 | 0.611 |

Modèle all-MiniLM-L6-v2, troncature à 256 tokens (71 % des documents la dépassent). Historique complet et notes d'analyse : [RESULTS.md](RESULTS.md).

## Objectif

Mesurer la qualité d'un retrieval dense mono-passage (embedding cosinus) sur SciFact, avec 6 métriques implémentées manuellement : Recall@{1,5,10,100}, nDCG@10, MRR.

## Stack

- Python 3.11+
- `sentence-transformers` (modèle `all-MiniLM-L6-v2`, 384d, 256 tokens)
- ChromaDB (store vectoriel, espace cosinus)
- numpy (similarité cosinus exacte)

## Quickstart

```bash
# 1. Environnement Python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Données BEIR SciFact
mkdir -p data/scifact
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip -d data/scifact_tmp
mv data/scifact_tmp/scifact/corpus.jsonl data/scifact/
mv data/scifact_tmp/scifact/queries.jsonl data/scifact/
mv data/scifact_tmp/scifact/qrels data/scifact/
rm -rf data/scifact_tmp scifact.zip

# 3. Ingestion (embedding + indexation ChromaDB) — ~2 min
python -m rag_eval_scifact.ingest

# 4. Run d'évaluation
python -m rag_eval_scifact.run_eval

# 5. Tests
pytest
```

Le dossier `data/scifact/` doit contenir `corpus.jsonl`, `queries.jsonl`, et `qrels/test.tsv`.

## Commandes

| Commande | Description |
|----------|-------------|
| `python -m rag_eval_scifact.ingest` | Embedde les 5183 docs et indexe dans ChromaDB (cosinus). A faire une seule fois. Produit `chroma_data/`. |
| `python -m rag_eval_scifact.run_eval` | Retrieval dense (300 requêtes test) + calcul des 6 métriques + artefacts (`results/*.json` + `RESULTS.md`). |
| `pytest` | Tests unitaires des métriques (recall, nDCG, MRR). |
| `pip install -e ".[dashboard]"` | Installe les dépendances dashboard (streamlit, plotly). |
| `streamlit run dashboard.py` | Lance le dashboard d'exploration des résultats. |

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
