---
description: Convention de versioning des runs d'évaluation — RESULTS.md append-only + JSON taggés
paths: ["results/**", "RESULTS.md", "**/*.py"]
---

# Versioning des runs

## Run non commité = run inexistant

Chaque run d'évaluation **doit** produire ET commiter :
1. Une ligne dans `RESULTS.md` (append-only, jamais réécrire l'historique)
2. Un fichier `results/{campagne}/{run}.json.gz` avec le détail complet du run (format ci-dessous, compressé gzip)
3. **Un tag git par campagne** (ex. `v2-grid`), posé sur le commit qui contient tous ses runs

Le run v1 garde son format historique (`results/v1-dense-*.json`, non compressé, tag `v1-dense`).
MLflow lit ces JSON comme artefacts. Son stockage local (`mlflow.db`, et `mlruns/` ou `mlartifacts/`
pour les artefacts) n'est **pas** la source de vérité et n'est pas commité : un run présent dans MLflow
mais absent de `RESULTS.md` et de `results/` n'existe pas.

**Campagne `dev` — runs de vérification d'un ticket.** Un run lancé pour vérifier un critère pendant un
tour va dans la campagne `dev` : `results/dev/` est ignoré par git, et la ligne que ce run ajoute à
`RESULTS.md` est retirée avant le commit. Un run `dev` n'est jamais un résultat. Décidé le 2 octobre 2026.

Un run dont les artefacts ne sont pas commités **n'existe pas**.
Ref: `rag-eval-scifact-etape2-decisions.md:106`

## Format JSON obligatoire

**Niveau run** (carte d'identité) :
- `version` (tag git), `date`, `dataset_hash`
- `campagne`, `run_name`, `config` (config Hydra résolue : retriever, unité/chunking, modèle, fusion, rerank, top-k)
- Config embedding : `model`, `max_seq_length`, `dim`
- Les 6 métriques agrégées

**Niveau requête** (un objet par requête) :
- `query_id` + texte
- Docs attendus (qrels) **avec longueur en tokens** — corrèle échec et troncature
- Top-100 ramenés : `_id` + rang + score de similarité
- Métriques de cette requête (found@k, rang du bon doc)

`dataset_hash` garantit que deux runs portent sur les mêmes données.
Ref: `rag-eval-scifact-etape2-decisions.md:58-76`
