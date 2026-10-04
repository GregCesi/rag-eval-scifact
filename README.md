# RAG-Eval SciFact

Banc d'évaluation qui compare des stratégies de retrieval sur BEIR SciFact (5183 docs scientifiques), avec des métriques codées à la main.

## Résultats

| Stratégie | nDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| Dense MiniLM, document tronqué à 256 tokens (référence) | 0,645 | 0,783 | 0,611 |
| BM25, document entier | 0,655 | 0,779 | 0,626 |
| Dense MiniLM, passages | 0,676 | 0,817 | 0,641 |
| Dense Qwen3, abstract entier, avec reranker | 0,700 | 0,836 | 0,671 |
| Hybride Qwen3 passages + BM25, union | 0,731 | 0,857 | 0,703 |
| Dense Qwen3, passages | 0,732 | 0,854 | 0,701 |

Détail des 34 runs et prédiction écrite avant la campagne : [results/v2-grid/RAPPORT.md](results/v2-grid/RAPPORT.md), [results/v2-grid/PREDICTION.md](results/v2-grid/PREDICTION.md).

## Ce qui est testé

- Modèle d'embedding : `all-MiniLM-L6-v2` ou `Qwen3-Embedding-0.6B`.
- Retrieval lexical : BM25.
- Lecture du document : tronqué à 256 tokens, abstract entier, ou passages.
- Fusion dense + lexical : union ou RRF.
- Reranker : avec ou sans.

34 combinaisons.

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
| `python -m rag_eval_scifact.generate_passage_detail <run.json.gz>` | Pour un run dense ou BM25 en unité passages, sans reranker : fichier dérivé `<run>.passages.json.gz` listant, pour chaque requête et chaque document attendu ou du top 10, la liste de ses passages avec le score de chacun. Lit le cache d'embeddings du run (dense) sans jamais le recalculer ; s'arrête sans rien écrire si ce cache est absent, si le `dataset_hash` ne correspond plus au corpus, ou si le meilleur score de passage ne retrouve pas le score du run. Option : `--cache-dir` (défaut `~/.cache/rag-eval-scifact`). |
| `pytest` | Tests unitaires des métriques (recall, nDCG, MRR), des artefacts de campagne, du suivi MLflow, de la comparaison de runs, de la grille v2-grid et du rapport de campagne. |
| `pip install -e ".[dashboard]"` | Installe les dépendances dashboard (streamlit, plotly). |
| `streamlit run dashboard.py` | Lance le dashboard d'exploration des résultats. Page **Comparer deux runs** : choisit une campagne (dossier de `results/` avec des runs au format campagne) puis deux de ses runs, et affiche leurs 6 métriques côte à côte, le nombre de claims où le meilleur rang d'un document attendu s'améliore / se dégrade / ne change pas entre les deux runs, la liste filtrable des claims dont le rang change, et pour un claim choisi le détail — documents attendus et top 10 de chaque run, côte à côte. Pour un run en unité passages dont le fichier dérivé existe (`generate_passage_detail`), le texte d'un document attendu ou du top 10 montre le score de chaque passage et surligne celui qui a fait remonter le document ; sans fichier dérivé, la page l'indique (« passages non disponibles pour ce run ») et affiche le texte sans coupure. |

## Méthode

- **Éval fait-main** : métriques, fusion RRF et test statistique apparié sont codés à la main, sans lib d'évaluation (`pytrec_eval`, `beir`, `ranx`, ...). Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Ollama uniquement** : zéro API LLM payante, un LLM local passe par Ollama. Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Run non commité = run inexistant** : `RESULTS.md` + `results/{campagne}/{run}.json.gz` doivent être commités ensemble, avec un tag git par campagne. Voir [.claude/rules/versioning.md](.claude/rules/versioning.md).
