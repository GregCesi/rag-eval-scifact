---
description: Convention de versioning des runs d'évaluation — RESULTS.md append-only + JSON taggés
paths: ["results/**", "RESULTS.md", "**/*.py"]
---

# Versioning des runs

## Run non commité = run inexistant

Chaque run d'évaluation **doit** produire ET commiter :
1. Une ligne dans `RESULTS.md` (append-only, jamais réécrire l'historique)
2. Un fichier `results/{tag}-{date}.json` avec le détail complet du run
3. Les deux sous le même tag git (ex. `v1-dense`)

Un run dont les artefacts ne sont pas commités **n'existe pas**.
Ref: `rag-eval-scifact-etape2-decisions.md:106`

## Format JSON obligatoire

**Niveau run** (carte d'identité) :
- `version` (tag git), `date`, `dataset_hash`
- Config embedding : `model`, `max_seq_length`, `dim`
- Les 6 métriques agrégées

**Niveau requête** (un objet par requête) :
- `query_id` + texte
- Docs attendus (qrels) **avec longueur en tokens** — corrèle échec et troncature
- Top-100 ramenés : `_id` + rang + score de similarité
- Métriques de cette requête (found@k, rang du bon doc)

`dataset_hash` garantit que deux runs portent sur les mêmes données.
Ref: `rag-eval-scifact-etape2-decisions.md:58-76`
