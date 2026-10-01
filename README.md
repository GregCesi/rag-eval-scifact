# RAG-Eval SciFact

Banc d'évaluation qui compare des stratégies de retrieval sur BEIR SciFact (5183 docs scientifiques), avec des métriques codées à la main.

## Résultats de référence — v1-dense (2026-07-01)

| R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR |
|-----|-----|------|-------|---------|-----|
| 0.482 | 0.738 | 0.783 | 0.925 | 0.645 | 0.611 |

Modèle `all-MiniLM-L6-v2`, troncature à 256 tokens (71 % des documents la dépassent). Historique complet et notes d'analyse : [RESULTS.md](RESULTS.md).

## Error analysis v1

Les 300 requêtes du split `test` se répartissent en 4 buckets (comptes de [results/v1-buckets.json](results/v1-buckets.json), relevés le 1er octobre 2026) :

- **151** trouvées au rang 1
- **87** trouvées dans le top 10, hors rang 1
- **40** trouvées dans le top 100, hors top 10
- **22** hors top 100

Aucun levier n'est désigné comme correctif à ce jour : cette répartition cadre les prédictions de la campagne en cours, elle ne conclut rien. Fiches annotées par bucket : [results/export-annotations-v1.md](results/export-annotations-v1.md).

## Labo en cours

Depuis le 2026-10-01, le banc devient un labo multi-stratégies, tracé dans MLflow :

- Une stratégie se déclare dans une configuration Hydra ([conf/config.yaml](conf/config.yaml)), jamais en dur dans le pipeline.
- Chaque run est archivé dans `results/` et suivi dans MLflow (tracking local, `mlruns/`).
- Deux runs se comparent par un test de randomisation apparié, codé à la main.
- Une campagne ne se lance qu'après une prédiction écrite et commitée (`results/{campagne}/PREDICTION.md`).

## Leviers de la première campagne

La campagne `v2-grid` teste cinq leviers, chacun une dimension de configuration — **prévus, pas mesurés** : la grille n'a pas encore tourné, aucun résultat n'est à annoncer.

- BM25
- Fusion RRF
- Reranking
- Découpage en passages (chunking)
- Modèle long contexte `Qwen/Qwen3-Embedding-0.6B`

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
| `python -m rag_eval_scifact.run_campaign` | Lance un run de campagne depuis `conf/config.yaml` (Hydra). Sans argument : reproduit v1 à l'identique. Surcharge CLI : `python -m rag_eval_scifact.run_campaign top_k=10`. Artefacts : `results/<campagne>/<run>.json.gz` + `RESULTS.md`, et le run est journalisé dans MLflow (expérience = campagne). |
| `MLFLOW_ALLOW_FILE_STORE=true mlflow ui` | Ouvre l'interface MLflow (http://127.0.0.1:5000) sur le tracking local `mlruns/`. La variable d'environnement garde le backend fichier disponible (mode maintenance depuis MLflow 3.x). |
| `python -m rag_eval_scifact.compare_runs <run_a> <run_b>` | Compare deux runs de campagne (`.json` ou `.json.gz`) par test de randomisation apparié (nDCG@10 et MRR, codé à la main). Options : `--metric`, `--n-permutations`, `--seed`. |
| `pytest` | Tests unitaires des métriques (recall, nDCG, MRR), des artefacts de campagne, du suivi MLflow et de la comparaison de runs. |
| `pip install -e ".[dashboard]"` | Installe les dépendances dashboard (streamlit, plotly). |
| `streamlit run dashboard.py` | Lance le dashboard d'exploration des résultats. |

## Structure

```
conf/
  config.yaml     # Config Hydra par défaut (reproduit v1)
rag_eval_scifact/
  ingest.py        # Ingestion corpus -> ChromaDB
  retrieve.py      # Retrieval dense cosinus exact (top-k paramétrable)
  metrics.py       # Recall@k, nDCG@10, MRR (fait-main)
  run_output.py    # Génération artefacts du run v1 historique (JSON + RESULTS.md)
  run_eval.py      # Orchestrateur v1 : retrieval -> métriques -> artefacts
  campaign.py      # Artefacts d'un run de campagne (JSON gzip + RESULTS.md)
  run_campaign.py  # Point d'entrée CLI de campagne (config Hydra résolue)
  mlflow_tracking.py # Suivi MLflow d'un run de campagne (local, sans serveur)
  stats.py         # Test de randomisation apparié (fait-main)
  compare.py       # Charge deux runs et les compare via stats.py
  compare_runs.py  # Point d'entrée CLI de comparaison de deux runs
tests/
  test_metrics.py  # Tests unitaires métriques
  test_campaign.py # Tests des artefacts de campagne
  test_mlflow_tracking.py # Tests du suivi MLflow
  test_stats.py    # Tests du test de randomisation apparié
  test_compare.py  # Tests de la comparaison de deux runs
data/scifact/      # Données BEIR brutes (non versionnées)
chroma_data/       # Index ChromaDB (non versionné)
results/           # results/v1-*.json (run v1 historique) + results/<campagne>/<run>.json.gz
RESULTS.md         # Historique des runs (append-only, versionné)
```

## Méthode

- **Éval fait-main** : métriques, fusion RRF et test statistique apparié sont codés à la main, sans lib d'évaluation (`pytrec_eval`, `beir`, `ranx`, ...). Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Ollama uniquement** : zéro API LLM payante, un LLM local passe par Ollama. Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Run non commité = run inexistant** : `RESULTS.md` + `results/{campagne}/{run}.json.gz` doivent être commités ensemble, avec un tag git par campagne. Voir [.claude/rules/versioning.md](.claude/rules/versioning.md).
