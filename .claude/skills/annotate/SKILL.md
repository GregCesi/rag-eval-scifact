---
name: annotate
description: Annotation automatique des erreurs de retrieval par sous-agents. Prend un bucket (near_miss, deep_miss, miss_100), génère les fiches markdown, lance les sous-agents annotateurs par vagues de 8, compile les résultats en JSON.
---

# /annotate

Annotation automatique des erreurs de retrieval dense, exécutée par sous-agents en parallèle.

## Contrat

```
INPUT REQUIS
  - un bucket : near_miss | deep_miss | miss_100
  - optionnel : liste de query IDs (ex: "13,70,128") pour restreindre le périmètre
  - prérequis fichiers : results/v1-buckets.json + results/v1-dense-*.json

OUTPUT GARANTI
  - artefact produit : results/v1-{bucket}-auto-annotations.json
  - format : {"query_id": {"categorie": "...", "rang_golden": N, "note": "..."}}
  - résumé console : distribution des catégories + nombre d'ALARM
  - ✋ incomplet si : le bucket "perfect" est demandé (rien à annoter)
```

## Procédure

### Étape 0 — Valider les prérequis

1. Vérifier que `results/v1-buckets.json` existe.
2. Vérifier qu'au moins un `results/v1-dense-*.json` existe.
3. Vérifier que le bucket demandé est `near_miss`, `deep_miss` ou `miss_100`. Refuser `perfect`.
4. Si `results/v1-{bucket}-auto-annotations.json` existe déjà, prévenir l'utilisateur et demander confirmation avant écrasement.

### Étape 1 — Générer les fiches de cas

Exécuter :
```bash
python scripts/generate_case_files.py <bucket> [--query-ids <ids>]
```

Vérifier que le nombre de fiches produites dans `results/{bucket}_cases/` correspond au nombre attendu (cf. `v1-buckets.json` counts).

### Étape 2 — Lancer les sous-agents annotateurs

1. Lister les fiches `results/{bucket}_cases/query_*.md`.
2. Diviser en vagues de 8 maximum.
3. Pour chaque vague : lancer 8 sous-agents en parallèle avec l'outil Agent.
   - Chaque sous-agent reçoit comme prompt :
     a. Le contenu de la fiche markdown (lu avec Read)
     b. Le protocole d'annotation ci-dessous (copié verbatim)
   - Le prompt du sous-agent doit commencer par le protocole, suivi de `---\n` puis du contenu de la fiche.
4. Attendre la complétion de la vague avant de lancer la suivante.
5. Pour chaque réponse de sous-agent : extraire le JSON `{categorie, rang_golden, note}`.
   - Si le sous-agent a enveloppé le JSON dans un bloc markdown, le nettoyer.
   - Si le JSON est invalide, noter l'erreur et passer à la suite (ne pas retenter).

### Étape 3 — Compiler les résultats

1. Fusionner toutes les annotations dans un dict `{query_id: {categorie, rang_golden, note}}`.
2. Écrire `results/v1-{bucket}-auto-annotations.json` (indent=2, ensure_ascii=False).
3. Afficher la distribution des catégories (tableau).
4. Si >20% des annotations contiennent "ALARM" dans la note → alerter l'utilisateur.

---

## Protocole d'annotation (prompt sous-agent)

Le texte ci-dessous est transmis TEL QUEL à chaque sous-agent, précédé de la fiche markdown du cas.

```
Tu es un annotateur d'erreurs de retrieval RAG. Analyse ce cas.

## PROTOCOLE
### Règle d'or : Le golden RÉPOND au claim ≠ le golden CONTIENT les mots du claim.

### Étape 0 — Polarité
Détermine si le golden SUPPORTS ou REFUTES le claim.
En SciFact, le qrel marque le doc comme pertinent même s'il contredit le claim.

### Étape 1 — Le golden statue-t-il sur le claim ?
Lis le golden ENTIÈREMENT (y compris après le marqueur ✂ coupe).
- OUI : le golden contient une réponse directe au claim (même par la négative/réfutation)
- NON : le golden ne répond pas réellement au claim (tangentiel, sujet différent, inférence requise)

### Étape 2 (si OUI) — Cause de l'échec
Le golden répond mais le retriever ne l'a pas trouvé. Pourquoi ?

- **T** (Troncature) : le passage-clé est APRÈS le marqueur ✂. L'embedding n'a jamais vu l'evidence.
  Critère : le passage répondant au claim est physiquement situé après "--- ✂ coupe 256 tokens ---".

- **C-lexical** : gap de vocabulaire pur. Le golden utilise des termes différents du claim pour dire
  la même chose, et un BM25 avec les bons termes l'aurait retrouvé.
  Critère : on peut identifier un terme précis du claim absent du golden (ou vice-versa).

- **C-sémantique** : gap plus profond. Le golden reformule au niveau conceptuel/d'abstraction.
  Un reranker cross-encoder ou NLI serait nécessaire.
  Critère : le gap n'est pas un simple swap lexical mais un changement de niveau d'abstraction.

- **combo** : plusieurs causes combinées (T+C, C-lexical+C-sémantique, etc.). Préciser lesquelles
  dans la note.

### Étape 3-Q (si NON) — Le golden ne répond pas au claim

- **Q-inférence** : le lien claim→golden requiert une inférence multi-sauts hors de portée du retriever.
- **Q-inférence (gold-faible)** : le golden ne répond VRAIMENT pas au claim — c'est probablement
  une erreur d'annotation du dataset.
- **Q-saturation** : le corpus contient N documents aussi proches du claim que le golden — le ranking
  est arbitraire.

### Garde-fou
Si le bucket est deep_miss ou miss_100 ET que le golden ne statue pas (Étape 1 = NON) :
→ Commence la note par "ALARM". C'est un signal de problème dans le dataset.

### Output
Réponds UNIQUEMENT avec ce bloc, sans prose avant ni après :

```
Query <ID> — [claim court]
Rang golden : <rang>
Bucket : <catégorie>
Justif 1 ligne : <justification>
```

Puis sur une ligne séparée, le JSON :
{"categorie": "<T|C-lexical|C-sémantique|combo|Q-inférence|Q-inférence (gold-faible)|Q-saturation>", "rang_golden": <rang du golden, entier ou null si hors top-100>, "note": "<justification en 1 phrase, en français>"}
```

---

## Garde-fous

- ❌ Ne JAMAIS modifier `results/v1-annotations.json` (annotations manuelles sacrées).
- ❌ Ne jamais lancer plus de 8 sous-agents simultanément.
- ❌ Ne pas retenter automatiquement un sous-agent qui échoue — loguer et continuer.
- ❌ Refuser le bucket `perfect` (rien à annoter).
- ⚠️ Si >20% d'annotations ALARM → pause et alerte humain avant de continuer.
