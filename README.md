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

## Campagne v2-grid — résultats (2026-10-03)

34 runs (17 retrievers × rerank oui/non), lancés le 2026-10-03. Rapport complet : [results/v2-grid/RAPPORT.md](results/v2-grid/RAPPORT.md). Prédiction écrite avant le premier run : [results/v2-grid/PREDICTION.md](results/v2-grid/PREDICTION.md).

| run | nDCG@10 | Δ nDCG@10 | p (nDCG@10) | MRR |
|---|---|---|---|---|
| dense-qwen3-passages-sans-reranker | 0.732 | +0.087 | 0.0001 | 0.701 |
| hybrid-qwen3-passages-union-sans-reranker | 0.731 | +0.086 | 0.0001 | 0.703 |
| hybrid-qwen3-256-rrf-sans-reranker | 0.712 | +0.067 | 0.0003 | 0.690 |
| dense-qwen3-abstract-entier-sans-reranker | 0.700 | +0.055 | 0.0006 | 0.669 |
| dense-qwen3-256-sans-reranker | 0.697 | +0.052 | 0.0004 | 0.663 |
| dense-minilm-256-avec-reranker | 0.689 | +0.044 | 0.0070 | 0.663 |
| dense-minilm-passages-sans-reranker | 0.676 | +0.031 | 0.0259 | 0.641 |
| bm25-document-sans-reranker | 0.655 | +0.010 | 0.6200 | 0.626 |
| dense-minilm-256-sans-reranker (référence v1) | 0.645 | +0.000 | 1.0000 | 0.611 |

Écart et p-value face à la référence `dense-minilm-256-sans-reranker` (v1 rejouée), test de randomisation apparié codé à la main.

### Effet de chaque levier isolé, en nDCG@10

- **Modèle, à 256 tokens** : MiniLM 0,645, Qwen 0,697.
- **Passages contre troncature** : MiniLM 0,676 contre 0,645 ; Qwen 0,732 contre 0,697 ; BM25 0,625 contre 0,655.
- **Abstract entier contre troncature, avec Qwen** : 0,700 contre 0,697.

### Reranker

Sur les 17 retrievers, 9 runs avec reranker sont au-dessus de leur jumeau sans reranker (ceux à base de MiniLM seul, de BM25 seul ou des deux, plus Qwen abstract entier à +0,001) et 8 en dessous (tous à base de Qwen) ; les 17 runs avec reranker tiennent entre 0,676 et 0,700.

### Effet sur les buckets v1

Au mieux, 8 des 22 claims `miss_100` reviennent dans le top 10 (part 0,364), et au mieux 26 des 40 claims `deep_miss` reviennent dans le top 10 (part 0,650). Buckets : [results/v1-buckets.json](results/v1-buckets.json).

### Ma prédiction face à la mesure

Prédiction complète : [results/v2-grid/PREDICTION.md](results/v2-grid/PREDICTION.md).

- **Confirmé** : le penchant pour le dense sur passages, et un meilleur nDCG@10 sous 0,80.
- **Contredit** : « un reranker est toujours mieux » ; « les miss_100 tombent à 0 ».

### Limites

- Un seul jeu de données (BEIR SciFact).
- Un seul reranker testé (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Les p-values comparent chaque run à la référence v1, pas les runs entre eux.
- Les jugements de pertinence de SciFact sont incomplets ; l'effet de ce manque sur ces chiffres n'a pas été mesuré.

## Labo en cours

Depuis le 2026-10-01, le banc est un labo multi-stratégies, tracé dans MLflow. Première campagne, `v2-grid` : 34 runs, lancée et rapportée le 2026-10-03 — résultats ci-dessus.

- Une stratégie se déclare dans une configuration Hydra ([conf/config.yaml](conf/config.yaml)), jamais en dur dans le pipeline, et se lance par le lanceur Hydra standard (`--multirun` pour une grille).
- Chaque run est archivé dans `results/` et suivi dans MLflow (tracking local, SQLite `mlflow.db`).
- Deux runs se comparent par un test de randomisation apparié, codé à la main.
- Une campagne ne se lance qu'après une prédiction écrite et commitée (`results/{campagne}/PREDICTION.md`).

## Stack

- Python 3.13 (requis par le lanceur Hydra, voir `scripts/preflight.sh`)
- `sentence-transformers` (modèle `all-MiniLM-L6-v2`, 384d, 256 tokens)
- ChromaDB (store vectoriel, espace cosinus) — index de `run_eval.py` (v1)
- Hydra (déclaration de stratégie, `--multirun`) + MLflow (tracking SQLite, `mlflow.db`)
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
| `python -m rag_eval_scifact.run_campaign` | Lance un run de campagne depuis `conf/config.yaml` (lanceur Hydra standard). Sans argument : reproduit v1 à l'identique. Surcharge CLI : `python -m rag_eval_scifact.run_campaign top_k=10`. Plusieurs stratégies en une commande : `python -m rag_eval_scifact.run_campaign --multirun top_k=10,20`. Aide et valeurs surchargeables : `--help`. Artefacts : `results/<campagne>/<run>.json.gz` + `RESULTS.md`, et le run est journalisé dans MLflow (expérience = campagne). Les embeddings de documents et le classement de premier étage sont mis en cache sous `cache_dir` (défaut `~/.cache/rag-eval-scifact`) : un run identique ne recalcule ni l'un ni l'autre. |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` | Ouvre l'interface MLflow (http://127.0.0.1:5001) sur le tracking SQLite local `mlflow.db`. Pour relire une requête d'un run, ouvrir ce run puis son onglet **Traces** : une trace par requête (claim, top-10 remonté, docs attendus, rang du premier attendu dans le top 100 ou son absence) ; désactivable via `tracing=false`. |
| `python -m rag_eval_scifact.compare_runs <run_a> <run_b>` | Compare deux runs de campagne (`.json` ou `.json.gz`) par test de randomisation apparié (nDCG@10 et MRR, codé à la main). Options : `--metric`, `--n-permutations`, `--seed`. |
| `python -m rag_eval_scifact.run_grid --list` | Affiche les combinaisons déclarées dans `conf/grid/v2-grid.yaml` (une par ligne, nom de run lisible) puis leur nombre, sans rien lancer. |
| `python -m rag_eval_scifact.run_grid` | Lance chaque combinaison de `conf/grid/v2-grid.yaml` comme un run de la campagne `v2-grid` (`--campagne <nom>` pour cibler une autre campagne, ex. `dev`). Refuse de lancer `v2-grid` tant que `results/v2-grid/PREDICTION.md` n'est pas dans le dernier commit. Relancée sur une campagne interrompue, saute les combinaisons dont le fichier de résultats existe déjà et le dit. |
| `python -m rag_eval_scifact.run_report <campagne>` | Écrit `results/<campagne>/RAPPORT.md` : tableau des runs de la campagne trié par nDCG@10 décroissant, avec les 6 métriques, l'écart et la p-value du test apparié face à `results/v1-rejeu/baseline`, la part par bucket v1, la part tronquée, la latence et la durée d'indexation. |
| `pytest` | Tests unitaires des métriques (recall, nDCG, MRR), des artefacts de campagne, du suivi MLflow, de la comparaison de runs, de la grille v2-grid et du rapport de campagne. |
| `pip install -e ".[dashboard]"` | Installe les dépendances dashboard (streamlit, plotly). |
| `streamlit run dashboard.py` | Lance le dashboard d'exploration des résultats. Page **Comparer deux runs** : choisit une campagne (dossier de `results/` avec des runs au format campagne) puis deux de ses runs, et affiche leurs 6 métriques côte à côte, le nombre de claims où le meilleur rang d'un document attendu s'améliore / se dégrade / ne change pas entre les deux runs, la liste filtrable des claims dont le rang change, et pour un claim choisi le détail — documents attendus et top 10 de chaque run, côte à côte. |

## Structure

```
conf/
  config.yaml     # Config Hydra par défaut (reproduit v1)
  grid/
    v2-grid.yaml   # Grille déclarative : une entrée = une combinaison déjà réalisable
rag_eval_scifact/
  ingest.py        # Ingestion corpus -> ChromaDB
  retrieve.py      # Retrieval dense cosinus exact (top-k paramétrable)
  metrics.py       # Recall@k, nDCG@10, MRR (fait-main)
  run_output.py    # Génération artefacts du run v1 historique (JSON + RESULTS.md)
  run_eval.py      # Orchestrateur v1 : retrieval -> métriques -> artefacts
  campaign.py      # Artefacts d'un run de campagne (JSON gzip + RESULTS.md)
  run_campaign.py  # Point d'entrée CLI de campagne (lanceur Hydra standard, --multirun)
  cache.py         # Cache disque (embeddings de documents, classement de premier étage)
  mlflow_tracking.py # Suivi MLflow d'un run de campagne (local, SQLite mlflow.db)
  stats.py         # Test de randomisation apparié (fait-main)
  compare.py       # Charge deux runs et les compare via stats.py
  compare_runs.py  # Point d'entrée CLI de comparaison de deux runs
  grid.py          # Lecture de la grille déclarative + détection des runs déjà faits
  run_grid.py      # Point d'entrée CLI de la grille (mode liste / lancement)
  report.py        # Tableau d'une campagne face à la référence v1-rejeu/baseline
  run_report.py    # Point d'entrée CLI du rapport de campagne
tests/
  test_metrics.py  # Tests unitaires métriques
  test_campaign.py # Tests des artefacts de campagne
  test_mlflow_tracking.py # Tests du suivi MLflow
  test_stats.py    # Tests du test de randomisation apparié
  test_compare.py  # Tests de la comparaison de deux runs
  test_cache.py    # Tests du cache (embeddings, classement de premier étage)
  test_retrieve_campaign.py # Tests du retrieval de campagne (cache embarqué)
  test_run_campaign_cli.py  # Tests du lanceur Hydra (--help, --multirun)
  test_grid.py     # Tests de la grille déclarative et de la détection de runs faits
  test_run_grid.py # Tests du CLI de grille (mode liste, lancement, garde, reprise)
  test_report.py   # Tests du rapport de campagne (tri, écart, p-value, régénération)
data/scifact/      # Données BEIR brutes (non versionnées)
chroma_data/       # Index ChromaDB (non versionné)
results/           # results/v1-*.json (run v1 historique) + results/<campagne>/<run>.json.gz
RESULTS.md         # Historique des runs (append-only, versionné)
```

## Méthode

- **Éval fait-main** : métriques, fusion RRF et test statistique apparié sont codés à la main, sans lib d'évaluation (`pytrec_eval`, `beir`, `ranx`, ...). Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Ollama uniquement** : zéro API LLM payante, un LLM local passe par Ollama. Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Run non commité = run inexistant** : `RESULTS.md` + `results/{campagne}/{run}.json.gz` doivent être commités ensemble, avec un tag git par campagne. Voir [.claude/rules/versioning.md](.claude/rules/versioning.md).
