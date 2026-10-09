# RAG-Eval SciFact

Banc d'évaluation du retrieval sur BEIR SciFact : 5 183 résumés d'articles scientifiques, 300 affirmations à vérifier, et des métriques codées à la main. Le dépôt est tenu comme un cahier de labo. Chaque campagne commence par une prédiction écrite et commitée, se termine par un rapport, et tous ses runs sont archivés dans `results/`.

## Résultats

nDCG@10 du meilleur système de chaque étape.

| Système | 188 affirmations avec preuve | 300 affirmations | Campagne |
|---|---|---|---|
| Dense MiniLM, document tronqué à 256 tokens (référence) | 0,774 | 0,645 | v1, v2-grid |
| Dense Qwen3-Embedding-0.6B, passages de 128 tokens | 0,873 | 0,732 | v2-grid |
| Dense Qwen3-Embedding-4B, passages de 128 tokens | 0,907 | 0,791 | v4-leviers |
| Qwen3-Embedding-0.6B, passages, puis Qwen3-Reranker-0.6B sur les 20 premiers documents | 0,910 | 0,769 | v5-rerankers |

La colonne « 188 affirmations avec preuve » est la mesure de référence du labo. Pour ces affirmations, les annotateurs de SciFact ont désigné dans le document attendu les phrases qui confirment ou contredisent l'affirmation. Pour les 112 autres, le document attendu est un article cité par l'affirmation, sans aucune phrase de preuve annotée.

Ce que les campagnes ont mesuré :

1. **Le jeu de données mélange deux tâches.** Sur les 339 paires (affirmation, document attendu), 138 confirment, 71 contredisent et 130 n'ont pas de preuve. Le même système (Qwen3-Embedding-0.6B, passages) obtient 0,873 sur les 188 affirmations avec preuve et 0,486 sur les 112 sans preuve. Sur les 188, le document attendu est dans les 100 premiers pour toutes les affirmations.
2. **Un reranker se juge sur les affirmations avec preuve.** Dans la grille v2-grid, le reranker MiniLM monte le score des 17 stratégies sur les 188 affirmations avec preuve (de +0,002 à +0,100) et le baisse pour les 17 sur les 112 sans preuve. Sur les 300 affirmations réunies, 9 stratégies montent et 8 descendent.
3. **Parmi quatre leviers testés sur le meilleur système de la grille, un seul change le score.** Le modèle d'embedding plus gros (Qwen3-Embedding-4B) gagne +0,034 (p = 0,002). Retirer l'instruction de requête, chercher avec un faux résumé rédigé par un LLM (HyDE) ou passer à un modèle biomédical (MedCPT) donnent des écarts de 0,011 au plus, avec p > 0,5.
4. **Un petit reranker sur le petit modèle fait jeu égal avec le gros modèle seul.** Qwen3-Reranker-0.6B posé sur Qwen3-Embedding-0.6B atteint 0,910, contre 0,907 pour Qwen3-Embedding-4B sans reranker. Le reranker MiniLM de la grille, sur la même base, apporte +0,007 (p = 0,67).

Tous les écarts ci-dessus sont des nDCG@10 sur les 188 affirmations avec preuve, sauf mention contraire. Les p-values viennent d'un test de randomisation apparié.

## Protocole

**Données.** BEIR SciFact, split test : 5 183 documents, 300 affirmations, 339 paires (affirmation, document attendu). Le corpus entier est toujours indexé, distracteurs compris. Les étiquettes par paire viennent de la publication d'origine de SciFact ([allenai/scifact](https://github.com/allenai/scifact)) : ses paires correspondent une à une aux qrels BEIR, et la commande qui les rapproche s'arrête si ce n'est plus le cas.

**Trois catégories d'affirmations.** Chaque affirmation est rangée selon l'étiquette de son document attendu.

| Catégorie | Affirmations | Ce que dit le document attendu |
|---|---|---|
| confirme | 124 | Des phrases annotées confirment l'affirmation |
| contredit | 64 | Des phrases annotées la contredisent |
| sans preuve | 112 | L'article est cité, aucune phrase n'est annotée |

**Métriques.** Recall@1, @5, @10, @100, nDCG@10 et MRR, implémentés dans [`metrics.py`](rag_eval_scifact/metrics.py) et testés. Aucune bibliothèque d'évaluation n'est importée (`pytrec_eval`, `beir`, `ranx`). La fusion RRF et le kappa de Cohen sont codés à la main eux aussi.

**Comparer deux runs.** Test de randomisation apparié sur les scores par affirmation, 10 000 permutations ([`stats.py`](rag_eval_scifact/stats.py)). Avec plusieurs dizaines de runs, des écarts « significatifs » apparaissent par hasard : le labo lit les gros écarts, pas les 0,01.

**Prédiction avant mesure.** Aucune campagne ne se lance tant que son fichier `PREDICTION.md` n'est pas dans le dernier commit. Le lanceur de grille le vérifie et refuse sinon.

**Traçabilité.** Une stratégie est une configuration Hydra ([`conf/`](conf/)), jamais du code modifié. Chaque run écrit un fichier `results/<campagne>/<run>.json.gz` avec sa configuration résolue, l'empreinte du jeu de données, les 100 premiers documents et les métriques de chaque affirmation. Il ajoute une ligne à [`RESULTS.md`](RESULTS.md) et un run MLflow avec une trace par affirmation. Une campagne terminée porte un tag git. Un run qui n'est pas commité n'existe pas.

**Machine.** Les runs tournent en local sur un Mac de 16 Go. Les LLM locaux passent par Ollama.

## Campagnes

| Campagne | Question | Runs | Prédiction | Rapport |
|---|---|---|---|---|
| `v1-dense` | Que vaut un retrieval dense simple ? | 1 | | [RESULTS.md](RESULTS.md) |
| `v2-grid` | Quelle combinaison de modèle, d'unité de lecture, de fusion et de reranker ? | 34 | [prédiction](results/v2-grid/PREDICTION.md) | [rapport](results/v2-grid/RAPPORT.md), [par catégorie](results/v2-grid/RAPPORT-PAR-CATEGORIE.md) |
| `v3-juge` | Un juge LLM lit-il les documents comme les annotateurs ? | 438 paires jugées, 2 juges | [prédiction](results/v3-juge/PREDICTION.md) | [rapport](results/v3-juge/RAPPORT.md) |
| `v4-leviers` | Qu'est-ce qui fait monter le score sur les affirmations avec preuve ? | 5 | [prédiction](results/v4-leviers/PREDICTION.md) | [par catégorie](results/v4-leviers/RAPPORT-PAR-CATEGORIE.md) |
| `v5-rerankers` | Un meilleur reranker rattrape-t-il un plus gros modèle ? | 5 sur 6 | [prédiction](results/v5-rerankers/PREDICTION.md) | [par catégorie](results/v5-rerankers/RAPPORT-PAR-CATEGORIE.md) |

### v1-dense : la référence

`all-MiniLM-L6-v2`, un vecteur par document, texte tronqué à 256 tokens (71 % des documents dépassent cette longueur).

| R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| 0,482 | 0,738 | 0,783 | 0,925 | 0,645 | 0,611 |

### v2-grid : 34 stratégies

Deux modèles d'embedding (`all-MiniLM-L6-v2`, `Qwen3-Embedding-0.6B`), BM25, trois unités de lecture (document tronqué à 256 tokens, résumé entier, passages de 128 tokens regroupés par document), deux fusions dense + BM25 (union, RRF), avec ou sans reranker `ms-marco-MiniLM-L-6-v2` sur les 100 premiers documents.

| Stratégie | nDCG@10, 188 avec preuve | nDCG@10, 300 | Recall@10, 300 | MRR, 300 |
|---|---|---|---|---|
| Dense MiniLM, document tronqué à 256 tokens (référence) | 0,774 | 0,645 | 0,783 | 0,611 |
| BM25, document entier | 0,815 | 0,655 | 0,779 | 0,626 |
| Dense MiniLM, passages | 0,809 | 0,676 | 0,817 | 0,641 |
| Dense Qwen3, résumé entier, avec reranker | 0,876 | 0,700 | 0,836 | 0,671 |
| Hybride Qwen3 passages + BM25, union | 0,865 | 0,731 | 0,857 | 0,703 |
| Dense Qwen3, passages | 0,873 | 0,732 | 0,854 | 0,701 |

### v3-juge : pourquoi un document est attendu

Deux juges LLM lisent chaque paire (affirmation, document) et rendent un verdict : confirme, contredit, ou information insuffisante. Le juge de référence est Claude, le juge local est `llama3.1:8b` par Ollama, avec le même prompt.

| | Juge Claude | Juge local |
|---|---|---|
| Accord avec les annotateurs de SciFact sur les documents attendus | 85,8 % | 63,3 % |
| Accord entre les deux juges | 63,6 % (kappa 0,47) | |

Le juge Claude donne aussi un niveau de lecture à chaque document attendu qui porte une preuve : la preuve se lit mot pour mot (direct), elle demande de connaître un terme scientifique équivalent (vocabulaire), ou elle demande un raisonnement. Nombre de documents attendus classés dans les 5 premiers, par niveau :

| Système | Direct (60) | Vocabulaire (55) | Raisonnement (71) |
|---|---|---|---|
| Qwen3-Embedding-0.6B, passages | 60 | 51 | 63 |
| Qwen3-Embedding-4B, passages | 59 | 55 | 65 |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 58 | 54 | 66 |

Le juge a aussi lu les documents classés devant le document attendu : il estime que 38 d'entre eux sur 104 répondent à l'affirmation, sur 28 affirmations.

### v4-leviers : un levier à la fois

Référence : Qwen3-Embedding-0.6B, passages, dense seul, sans reranker. Chaque run ne change qu'un levier.

| Levier | nDCG@10, 188 avec preuve | Écart | p | nDCG@10, 300 |
|---|---|---|---|---|
| Référence | 0,873 | | | 0,732 |
| Sans instruction de requête | 0,867 | -0,006 | 0,62 | 0,703 |
| HyDE (faux résumé rédigé par `llama3.1:8b`) | 0,875 | +0,002 | 0,87 | 0,740 |
| MedCPT (modèle biomédical) | 0,862 | -0,011 | 0,57 | 0,728 |
| Qwen3-Embedding-4B | 0,907 | +0,034 | 0,002 | 0,791 |

### v5-rerankers : le deuxième étage

Même premier étage que la référence de v4-leviers. Chaque reranker relit les 20 premiers documents.

| Reranker | nDCG@10, 188 avec preuve | Écart | p | nDCG@10, 300 |
|---|---|---|---|---|
| Aucun | 0,873 | | | 0,732 |
| `ms-marco-MiniLM-L-6-v2` | 0,880 | +0,007 | 0,67 | 0,711 |
| `MedCPT-Cross-Encoder` | 0,900 | +0,027 | 0,058 | 0,774 |
| `bge-reranker-v2-m3` | 0,904 | +0,032 | 0,016 | 0,744 |
| `Qwen3-Reranker-0.6B` | 0,910 | +0,038 | 0,009 | 0,769 |

Le sixième run, `Qwen3-Reranker-4B`, n'a pas abouti : le modèle ne tient pas en mémoire sur la machine du labo.

## Prédictions face aux mesures

Les prédictions sont écrites à l'instinct avant chaque campagne et ne sont jamais corrigées après coup.

| Campagne | Prédiction | Mesure |
|---|---|---|
| v2-grid | Le meilleur nDCG@10 sera autour de 0,75, et 0,80 est hors de portée | 0,732 sur les 300 affirmations |
| v2-grid | Un reranker améliore les 17 stratégies | Sur les 300 affirmations, 9 montent et 8 descendent |
| v2-grid | Avec la meilleure stratégie, plus aucun document attendu hors des 100 premiers | 14 affirmations restent hors des 100 premiers, toutes sans preuve |
| v3-juge | Accord du juge Claude avec les annotateurs : plus de 85 % espéré, 70 % redouté | 85,8 % |
| v3-juge | Plus de 50 des documents classés devant le document attendu répondent à l'affirmation | 38 sur 104 selon le juge Claude |
| v4-leviers | HyDE fera moins bien, au mieux aussi bien | +0,002 (p = 0,87) |
| v4-leviers | Qwen3-Embedding-4B fera un peu mieux, +0,02 serait déjà excellent | +0,034 (p = 0,002) |
| v4-leviers | MedCPT sera en dessous de la référence | -0,011 (p = 0,57) |
| v5-rerankers | Aucun reranker posé sur le petit modèle n'atteint le gros modèle seul (0,907) | Qwen3-Reranker-0.6B : 0,910 |
| v5-rerankers | Le meilleur reranker sera Qwen3-Reranker-4B | Non mesuré |

## Limites

- Un seul jeu de données et 300 affirmations. Tous les choix sont faits sur le jeu qui sert à mesurer ; le split d'entraînement de SciFact n'a pas encore servi à les vérifier.
- Chaque p-value compare un run à sa référence. Aucune correction pour comparaisons multiples n'est appliquée.
- Les niveaux de lecture viennent d'un juge LLM, pas d'annotateurs humains. Ce juge est d'accord avec les annotateurs de SciFact dans 85,8 % des cas.
- `Qwen3-Reranker-4B` n'est pas mesuré, et la combinaison de Qwen3-Embedding-4B avec un reranker non plus.
- Le banc mesure le retrieval seul. Aucune génération de réponse n'est évaluée.

## Reproduire

```bash
# 1. Environnement (Python 3.11 ou plus)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Données BEIR SciFact
mkdir -p data/scifact
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip -d data/scifact_tmp
mv data/scifact_tmp/scifact/corpus.jsonl data/scifact/
mv data/scifact_tmp/scifact/queries.jsonl data/scifact/
mv data/scifact_tmp/scifact/qrels data/scifact/
rm -rf data/scifact_tmp scifact.zip

# 3. Tests
pytest

# 4. Voir les runs d'une campagne, puis la lancer
python -m rag_eval_scifact.run_grid --list --campagne v5-rerankers
python -m rag_eval_scifact.run_grid --campagne v5-rerankers

# 5. Rapport par catégorie, avec écart et p-value face à une référence
python -m rag_eval_scifact.run_category_report --campagne v5-rerankers --reference qwen3-passages-sans-reranker
```

Une campagne relancée saute les runs dont le fichier de résultats existe déjà. Les embeddings et les classements de premier étage sont mis en cache hors du dépôt.

Pour explorer les résultats :

```bash
streamlit run dashboard.py                                      # runs commités : comparaison de deux runs, niveaux de lecture
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # runs lancés sur la machine, une trace par affirmation
```

Les étiquettes par catégorie (`results/etiquettes-origine.json`) sont commitées. Pour les régénérer, il faut l'archive `data.tar.gz` de [allenai/scifact](https://github.com/allenai/scifact) décompressée dans `data/scifact/origine/`.

Toutes les commandes du dépôt sont décrites dans [docs/COMMANDES.md](docs/COMMANDES.md).

## Structure du dépôt

| Chemin | Contenu |
|---|---|
| [`rag_eval_scifact/`](rag_eval_scifact/) | Pipeline : découpage, BM25, retrieval, fusion, reranker, métriques, test statistique, juges, rapports |
| [`conf/`](conf/) | Configuration Hydra par défaut et une grille par campagne |
| [`results/`](results/) | Un dossier par campagne : prédiction, runs compressés, rapports |
| [`RESULTS.md`](RESULTS.md) | Journal de tous les runs, une ligne par run |
| [`tests/`](tests/) | Tests unitaires du pipeline, des rapports et du dashboard |
| [`dashboard.py`](dashboard.py) | Dashboard Streamlit |
| [`.claude/rules/`](.claude/rules/) | Règles du projet, lues par l'agent de code à chaque tour |

## Règles du projet

- **Éval codée à la main** : métriques, fusion RRF et test statistique sans bibliothèque d'évaluation. Voir [methodologie.md](.claude/rules/methodologie.md).
- **LLM locaux** : aucun appel à une API payante, les LLM passent par Ollama. Seule exception, le juge de référence de v3-juge, qui appelle Claude sans clé d'API. Voir [methodologie.md](.claude/rules/methodologie.md).
- **Run non commité = run inexistant** : `RESULTS.md` et `results/<campagne>/<run>.json.gz` sont commités ensemble. Voir [versioning.md](.claude/rules/versioning.md).
- **Invariants de mesure** : similarité cosinus, corpus entier indexé, troncature notée à chaque run. Voir [invariants.md](.claude/rules/invariants.md).

## Stack

Python, sentence-transformers, ChromaDB, rank_bm25, Hydra, MLflow, Streamlit, Ollama, LangGraph.
