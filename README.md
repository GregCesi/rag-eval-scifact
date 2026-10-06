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
| `python -m rag_eval_scifact.run_grid --list --campagne <nom>` | Affiche les combinaisons déclarées dans `conf/grid/<nom>.yaml` (une par ligne, nom de run lisible) puis leur nombre, sans rien lancer. Sans `--campagne` : `v2-grid` (34 runs), comme avant. Campagne sans fichier de grille : message nommant le fichier attendu, sans trace Python. |
| `python -m rag_eval_scifact.run_grid --campagne <nom>` | Lance chaque combinaison de `conf/grid/<nom>.yaml` comme un run de cette campagne (défaut `v2-grid`). `--run <run_name>` ne lance que ce run nommé (nom inconnu : liste les noms connus de la campagne). Refuse de lancer toute campagne sauf `dev` tant que `results/<nom>/PREDICTION.md` n'est pas dans le dernier commit. Relancée sur une campagne interrompue, saute les combinaisons dont le fichier de résultats existe déjà et le dit. |
| `python -m rag_eval_scifact.run_grid --campagne <nom> --essai <run_name>` | Essai rapide d'un run nommé : encode les passages des 50 premiers documents du corpus, affiche leur nombre, la durée et son extrapolation au corpus entier en minutes. N'écrit aucun fichier de résultats ni run MLflow. |
| `python -m rag_eval_scifact.run_hyde` | Rédige, pour chacune des 300 affirmations du jeu de test, un faux résumé d'article avec le modèle local llama3.1:8b par Ollama (température 0, graine fixe) — méthode HyDE. Écrit `results/v4-leviers/hyde.json` (identifiant, texte de l'affirmation, texte rédigé, modèle, durée). Affiche la progression toutes les 10 affirmations. Reprise sur une relance interrompue : les affirmations déjà rédigées ne le sont pas à nouveau. S'arrête sans trace Python, sans enregistrer l'affirmation en cours, si Ollama ne répond pas ou rend un texte vide. `--limit` borne le nombre d'affirmations rédigées. Le run `qwen3-passages-hyde` de la grille `v4-leviers` (`retriever.query_source=hyde`) cherche avec ce texte à la place de l'affirmation, sans instruction de requête ; refuse de se lancer si `hyde.json` est absent ou contient moins de 300 textes. |
| `python -m rag_eval_scifact.run_report <campagne>` | Écrit `results/<campagne>/RAPPORT.md` : tableau des runs de la campagne trié par nDCG@10 décroissant, avec les 6 métriques, l'écart et la p-value du test apparié face à `results/v1-rejeu/baseline`, la part par bucket v1, la part tronquée, la latence et la durée d'indexation. |
| `python -m rag_eval_scifact.run_origin_labels` | Écrit `results/etiquettes-origine.json` : pour chacune des 339 paires (claim, document attendu) du split test, l'étiquette des annotateurs d'origine SciFact (SUPPORT, CONTRADICT ou SANS_PREUVE) et, pour les deux premières, le texte de ses phrases-preuve, plus la catégorie de chaque claim (confirme, contredit, sans preuve). S'arrête sans rien écrire si ces paires ne correspondent pas exactement à `data/scifact/qrels/test.tsv`. |
| `python -m rag_eval_scifact.run_category_report` | Écrit `results/v2-grid/RAPPORT-PAR-CATEGORIE.md` à partir de `results/etiquettes-origine.json` et des 34 runs commités de `v2-grid` : pour chaque run et chaque catégorie de claim (confirme, contredit, sans preuve), l'effectif et les taux rang 1 / top 10 / top 100 ; le nDCG@10 sur les 300 claims, les 188 claims avec preuve et les 112 claims sans preuve ; et, par stratégie de base, l'écart de nDCG@10 avec et sans reranker sur ces deux derniers ensembles, avec la p-value du test apparié déjà codé à la main. |
| `python -m rag_eval_scifact.generate_passage_detail <run.json.gz>` | Pour un run dense ou BM25 en unité passages, sans reranker : fichier dérivé `<run>.passages.json.gz` listant, pour chaque requête et chaque document attendu ou du top 10, la liste de ses passages avec le score de chacun. Lit le cache d'embeddings du run (dense) sans jamais le recalculer ; s'arrête sans rien écrire si ce cache est absent, si le `dataset_hash` ne correspond plus au corpus, ou si le meilleur score de passage ne retrouve pas le score du run. Option : `--cache-dir` (défaut `~/.cache/rag-eval-scifact`). |
| `pytest` | Tests unitaires des métriques (recall, nDCG, MRR), des artefacts de campagne, du suivi MLflow, de la comparaison de runs, de la grille v2-grid et du rapport de campagne. |
| `python -m rag_eval_scifact.run_judge_pairs` | Écrit `results/<campagne>/paires.json` (défaut : campagne `v3-juge`), lu dans le run `dense-qwen3-passages-sans-reranker` et dans `results/etiquettes-origine.json` : deux familles de paires, chacune pouvant porter les deux — « attendu » (chaque paire claim/document attendu, 339, avec son étiquette d'origine) et « devant » (pour chaque claim avec preuve, les documents classés avant le premier document SUPPORT ou CONTRADICT, 5 au plus). Chaque paire porte son champ `famille` (`attendu`, `devant`, ou les deux). |
| `python -m rag_eval_scifact.run_judge --juge local` | Juge les paires d'une campagne (défaut : `v3-juge`) avec le juge local : un appel Ollama par paire (`--model`, défaut `llama3.1:8b`), température 0, graine fixe. Le prompt demande en plus, pour SUPPORTS/REFUTES, le niveau de lecture (DIRECT, VOCABULARY, REASONING ; NONE pour NOT_ENOUGH_INFO) — un niveau incohérent avec le verdict compte comme illisible. Écrit `results/<campagne>/jugements-local.json`, journalise un run MLflow (expérience = campagne) avec une trace par jugement. Reprise sur une campagne interrompue : les paires déjà jugées ne sont pas rejugées. `--limit` borne le nombre de paires jugées. Refuse de juger `v3-juge` tant que `results/v3-juge/PREDICTION.md` n'est pas dans le dernier commit. |
| `python -m rag_eval_scifact.run_judge --juge claude` | Même commande, avec le juge de référence Claude (`--model`, défaut `sonnet`) : un appel Claude Code non interactif par paire, sans outils, même prompt octet pour octet que le juge local (exception documentée de `.claude/rules/methodologie.md`). Écrit `results/<campagne>/jugements-claude.json`. |
| `python -m rag_eval_scifact.run_judge --juge etapes` | Même commande, avec le juge par étapes : un graphe à quatre nœuds par paire (claim, document, verdict, cause), un appel Ollama par nœud réellement traversé. Ne rend aucun niveau de lecture (prompts inchangés depuis EXE-121). Écrit `results/<campagne>/jugements-etapes.json`, trace MLflow avec une étape enfant par nœud appelé. |
| `python -m rag_eval_scifact.run_judge_report` | Écrit `results/<campagne>/RAPPORT.md` (défaut : campagne `v3-juge`) à partir des jugements présents : effectifs de verdicts et durée moyenne par juge, ce que chaque juge dit du jeu de données (documents attendus et intrus classés devant), le tableau croisé verdict / étiquette d'origine et la part d'accord sur les documents attendus, les verdicts sur les documents classés devant et les claims où l'un d'eux répond, la répartition des niveaux de lecture (« sans niveau » pour le juge par étapes), le tableau croisé entre juges, la part d'accord et le kappa de Cohen (codé à la main) entre chaque paire de juges, les causes du juge par étapes, et le croisement avec les étiquettes de `results/v1-annotations.json` et `results/v1-deep-miss-annotations.json`. Fonctionne avec un, deux ou trois juges présents et dit lesquels manquent. |
| `pip install -e ".[dashboard]"` | Installe les dépendances dashboard (streamlit, plotly). |
| `streamlit run dashboard.py` | Lance le dashboard d'exploration des résultats. Page **Comparer deux runs** : choisit une campagne (dossier de `results/` avec des runs au format campagne) puis deux de ses runs, et affiche leurs 6 métriques côte à côte, le nombre de claims où le meilleur rang d'un document attendu s'améliore / se dégrade / ne change pas entre les deux runs, la liste filtrable des claims dont le rang change, et pour un claim choisi le détail — documents attendus et top 10 de chaque run, côte à côte. Pour un run en unité passages dont le fichier dérivé existe (`generate_passage_detail`), le texte d'un document attendu ou du top 10 montre le score de chaque passage et surligne celui qui a fait remonter le document ; sans fichier dérivé, la page l'indique (« passages non disponibles pour ce run ») et affiche le texte sans coupure. |

## Méthode

- **Éval fait-main** : métriques, fusion RRF et test statistique apparié sont codés à la main, sans lib d'évaluation (`pytrec_eval`, `beir`, `ranx`, ...). Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Ollama uniquement** : zéro API LLM payante, un LLM local passe par Ollama. Voir [.claude/rules/methodologie.md](.claude/rules/methodologie.md).
- **Run non commité = run inexistant** : `RESULTS.md` + `results/{campagne}/{run}.json.gz` doivent être commités ensemble, avec un tag git par campagne. Voir [.claude/rules/versioning.md](.claude/rules/versioning.md).
