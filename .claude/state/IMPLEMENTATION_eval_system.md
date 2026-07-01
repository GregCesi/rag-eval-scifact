# IMPLEMENTATION — Système d'Évaluation fait-main (v1)

## 1. Vue d'ensemble

Implémenter le harness d'évaluation fait-main : 6 métriques calculées manuellement (Recall@{1,5,10,100} + nDCG@10 + MRR), le système d'artefacts (`RESULTS.md` append-only + `results/*.json` détaillé par run), et l'orchestration complète d'un run. C'est **LE livrable** du projet — le pipeline d'ingestion/retrieval n'est que son alimentation. Première chose à attaquer : les fonctions de métriques.

Ref décisions : `rag-eval-scifact-etape2-decisions.md` (§3, §4, §6)
Ref invariants : `.claude/rules/methodologie.md` (interdiction libs d'éval), `.claude/rules/versioning.md`

---

## 2. Schémas cibles

Base de travail, pas contrat figé — à affiner au livrable correspondant.

### JSON de sortie — niveau run

```json
{
  "version": "v1-dense",
  "date": "2026-07-01T14:30:00",
  "dataset_hash": "sha256:...",
  "config": {
    "model": "all-MiniLM-L6-v2",
    "max_seq_length": 256,
    "dim": 384
  },
  "metrics": {
    "recall@1": 0.0,
    "recall@5": 0.0,
    "recall@10": 0.0,
    "recall@100": 0.0,
    "ndcg@10": 0.0,
    "mrr": 0.0
  },
  "queries": [ "..." ]
}
```

### JSON de sortie — niveau requête (un objet dans `queries`)

```json
{
  "query_id": "3",
  "query_text": "...",
  "expected_docs": [
    {"doc_id": "4983", "token_count": 412}
  ],
  "retrieved_top100": [
    {"doc_id": "1234", "rank": 1, "score": 0.87},
    {"doc_id": "4983", "rank": 3, "score": 0.72}
  ],
  "per_query_metrics": {
    "found@1": false,
    "found@5": true,
    "found@10": true,
    "found@100": true,
    "best_rank": 3
  }
}
```

### Ligne RESULTS.md (append-only)

```
| v1-dense | 2026-07-01 | R@1=... | R@5=... | R@10=... | R@100=... | nDCG@10=... | MRR=... | max_seq=256 | (note d'analyse libre, remplie après error analysis) |
```

Deux colonnes distinctes en fin de ligne :
- **dette** (fixe) : `max_seq=256` — rappel permanent de la troncature acceptée
- **note d'analyse** (libre) : remplie à la main après error analysis de chaque run (ex. "les ratés sont des docs > 256", "ce run prouve X")

---

## 3. Phases

### Phase 1 — Métriques fait-main [M]

**Objectif :** Implémenter les 6 métriques d'évaluation from scratch, sans aucune lib d'éval externe.

**Livrables :**

- [x] 1.1 — `metrics.py` — Recall@k :
  - `recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - Pour une requête : parmi les top-k ramenés, quelle fraction des docs pertinents est présente ?
  - Recall@k = |{docs pertinents ∩ top-k}| / |{docs pertinents}|
  - Macro-average sur les 300 requêtes
- [x] 1.2 — `metrics.py` — nDCG@10 :
  - `ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - DCG@k = Σᵢ₌₁ᵏ relᵢ / log₂(i+1) — relevance **binaire** (1/0). Convention pytrec_eval/BEIR exacte : rang 1 → log₂(2)=1. Avec relevance binaire, gain linéaire = gain exponentiel (2^rel−1), aucune ambiguïté. **Ne pas modifier cette formule.**
  - IDCG@k = DCG du classement parfait (les docs pertinents en premier)
  - nDCG@k = DCG / IDCG. Si IDCG = 0 → retourner 0.0 (garde-fou division par zéro). **Sur SciFact test, chaque requête a >= 1 doc pertinent, donc ce cas ne se déclenche jamais** — le garde-fou est là par sécurité.
  - Macro-average sur les 300 requêtes
- [x] 1.3 — `metrics.py` — MRR :
  - `mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float`
  - Pour une requête : 1/rang du premier doc pertinent trouvé (0 si aucun trouvé dans top-100)
  - Note : doc pertinent au-delà du rang 100 → contribution MRR = 0. Cohérent avec la profondeur de log top-100 (on ne mesure rien au-delà). À documenter en commentaire dans `metrics.py`.
  - Macro-average sur les 300 requêtes
- [x] 1.4 — Tests unitaires `test_metrics.py` avec **oracles numériques écrits en dur** (calculés indépendamment du code, pas par les mêmes fonctions testées) :

  | Cas | Setup | R@1 | R@5 | R@10 | R@100 | MRR | nDCG@10 |
  |---|---|---|---|---|---|---|---|
  | trivial | 1 doc pertinent au rang 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
  | rang 2 | 1 doc pertinent au rang 2 | 0.0 | 1.0 | 1.0 | 1.0 | 0.5 | **0.6309** |
  | rang 3 | 1 doc pertinent au rang 3 | 0.0 | 1.0 | 1.0 | 1.0 | 0.3333 | **0.5** |
  | absent | 1 doc pertinent hors top-100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
  | multi | 2 docs pertinents aux rangs 2 et 5 | 0.0 | 1.0 | 1.0 | 1.0 | 0.5 | **0.6240** |

  Détail oracles nDCG (pour vérification de la formule) :
  - rang 2 : `DCG = 1/log2(3) = 0.6309` ; `IDCG = 1/log2(2) = 1` ; nDCG = **0.6309**
  - rang 3 : `DCG = 1/log2(4) = 0.5` ; `IDCG = 1/log2(2) = 1` ; nDCG = **0.5**
  - multi : `DCG = 1/log2(3) + 1/log2(6) = 0.6309 + 0.3869 = 1.0178` ; `IDCG = 1/log2(2) + 1/log2(3) = 1.6309` ; nDCG = 1.0178/1.6309 = **0.6240**

  Tolérance : 1e-4 sur toutes les comparaisons flottantes.

✋ Verify before continuing:
- [x] Aucun import de `pytrec_eval`, `beir.retrieval.evaluation`, `sentence_transformers.evaluation` ou lib d'éval équivalente — ref `.claude/rules/methodologie.md`
- [x] Chaque cas de test compare à sa valeur numérique oracle écrite en dur ci-dessus (tolérance 1e-4)
- [x] nDCG gère correctement le cas IDCG = 0 (pas de division par zéro)

Si tout est OK : "go". Sinon dis ce qui cloche.

---

### Phase 2 — Artefacts & reporting [S]

**Objectif :** Produire les artefacts de run (JSON détaillé + ligne RESULTS.md) conformes au format décidé.

**Livrables :**

- [x] 2.1 — `run_output.py` — Génération JSON :
  - Assembler le JSON niveau run (version, date, config, métriques agrégées)
  - **`dataset_hash` : lu depuis la metadata de la collection ChromaDB** (calculé et stocké par `ingest.py`, cf. `IMPLEMENTATION_rag_ingestion_retrieval.md` A4) — ne pas recalculer
  - Assembler le JSON niveau requête (query_id, query_text, expected_docs avec token_count, top-100 avec rang+score, per_query_metrics)
  - Écrire dans `results/{version}-{date}.json`
- [x] 2.2 — `run_output.py` — Append RESULTS.md :
  - Si `RESULTS.md` n'existe pas, créer avec header du tableau
  - Append une ligne avec les 6 métriques + colonne dette `max_seq=256` + colonne note d'analyse (vide par défaut, remplie à la main post-run)
  - Ne jamais réécrire les lignes existantes (append-only strict)
- [x] 2.3 — `per_query_metrics` : pour chaque requête, calculer :
  - `found@k` (bool) = **au moins un doc pertinent dans le top-k** (coincide avec Recall@k > 0 ; booléen de succès, pas une fraction)
  - `best_rank` (int | null) = **rang du premier doc pertinent** rencontré dans le top-100 (celui qui fournit le MRR), ou `null` si aucun trouvé

✋ Verify before continuing:
- [x] Le JSON contient `token_count` pour chaque doc attendu (sinon dette troncature non jugeable)
- [x] `RESULTS.md` mentionne explicitement la dette troncature 256 tokens — ref `.claude/rules/invariants.md`
- [x] Le fichier JSON est horodaté et nommé de manière unique par run

Si tout est OK : "go". Sinon dis ce qui cloche.

---

### Phase 3 — Orchestration run complet [S]

**Objectif :** Script principal qui enchaîne retrieval + évaluation + artefacts en un seul run reproductible.

**Livrables :**

- [x] 3.1 — `run_eval.py` — Orchestrateur :
  - Charger la collection ChromaDB (ingérée par `ingest.py`)
  - Exécuter le retrieval sur les 300 requêtes test (appel à la logique de `retrieve.py`)
  - Calculer les 6 métriques agrégées + per-query via `metrics.py`
  - Générer les artefacts via `run_output.py`
  - Afficher un résumé console des résultats
- [x] 3.2 — Intégration `dataset_hash` : vérifier que le hash du corpus au moment du run correspond à celui de l'ingestion

✋ Verify before continuing:
- [x] Un run complet produit `RESULTS.md` (ligne ajoutée) + `results/*.json` (fichier créé)
- [x] Le run est reproductible : même corpus → même hash → mêmes métriques
- [x] Rappel post-run : **commiter les artefacts + taguer** (run non commité = run inexistant) — ref `.claude/rules/versioning.md`

Si tout est OK : "go". Sinon dis ce qui cloche.

---

## 4. Livrables détaillés

| # | Livrable | Done = | Taille |
|---|----------|--------|--------|
| 1.1 | Recall@k (4 valeurs de k) | formule correcte, macro-averaged, testée | S |
| 1.2 | nDCG@10 | DCG/IDCG correct, cas IDCG=0 géré, testé | S |
| 1.3 | MRR | 1/rang premier pertinent, testé | XS |
| 1.4 | Tests unitaires métriques | 5 cas avec oracles numériques en dur passent (tolérance 1e-4) | S |
| 2.1 | Génération JSON run | format conforme au schéma §2, token_count inclus | S |
| 2.2 | Append RESULTS.md | append-only, dette troncature mentionnée | XS |
| 2.3 | Per-query metrics | found@k + best_rank calculés par requête | XS |
| 3.1 | Orchestrateur `run_eval.py` | run bout-en-bout, résumé console | M |
| 3.2 | Vérification dataset_hash | hash relu depuis collection = cohérent bout-en-bout | XS |

---

## 5. Dépendances critiques

- Phase 1 (métriques) n'a **aucune dépendance externe** — peut être développée et testée en isolation avec des données mock.
- Phase 2 (artefacts) dépend de Phase 1 pour les valeurs de métriques.
- Phase 3 (orchestration) dépend du pipeline RAG (`IMPLEMENTATION_rag_ingestion_retrieval.md`, Phase 2 — retrieval) + des Phases 1 et 2 de ce document.
- **Contrat d'interface partagé** (figé dans `IMPLEMENTATION_rag_ingestion_retrieval.md` §2) :
  ```
  RetrievalResult:
    query_id: str, query_text: str
    retrieved: list[{"doc_id": str, "rank": int, "score": float}]
    # score = similarité cosinus directe (brute-force numpy), top-100 trié desc
  ```
- **Recommandation de séquençage :** démarrer par les métriques (testables en isolation), puis RAG pipeline, puis artefacts + orchestration.

---

## 6. Garde-fous

- Si un import `pytrec_eval` ou équivalent est détecté dans le code → erreur bloquante, pas warning. C'est le cœur pédagogique du projet.
- Si `RESULTS.md` est réécrit (pas append) → restaurer depuis git. L'historique des runs est la mémoire du projet.
