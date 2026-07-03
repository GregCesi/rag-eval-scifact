# Journal — Outil d'analyse d'erreurs v1-dense

**Date** : 2026-07-02

## Contexte

Run v1-dense termine (R@1 0.482, R@10 0.783, R@100 0.925, nDCG@10 0.645).
Objectif : outiller l'analyse d'erreurs a la Husain avant de choisir le levier v2.

## Livrables

### 1. `scripts/extract_error_analysis.py`

Bucketise les 300 queries par profondeur d'echec :
- **perfect** (hit rang 1) : 151
- **near_miss** (top 10, pas rang 1) : 87
- **deep_miss** (top 100, hors top 10) : 40
- **miss_100** (hors top 100) : 22

Sorties :
- `results/v1-buckets.json` : mapping query_id -> bucket + rang + score
- `results/v1-error-analysis-miss100.md` : fiche d'annotation par query miss_100
  - Marqueur `✂ coupe 256 tokens` insere par le tokenizer HF du modele
  - Champs `categorie:` et `note:` vides a remplir a la main

### 2. Page Streamlit "Analyse d'erreurs" (refonte)

Remplace l'ancienne vue tabs par :
- Selecteur de bucket avec comptes
- Navigation query par query (prev/next + selecteur direct)
- Docs attendus avec visualisation troncature (vert/rouge)
- Top-5 retrouve avec rangs et scores
- Lecture des annotations depuis `v1-error-analysis-miss100.md` (tolerant)

## Decisions

- L'annotation vit dans le .md versionne, pas dans Streamlit (read-only)
- Le parsing des annotations est tolerant : champ vide = pas d'affichage, pas d'erreur
- Les sections .md sont identifiees par `## query_id: <id>` pour un parsing regex simple
