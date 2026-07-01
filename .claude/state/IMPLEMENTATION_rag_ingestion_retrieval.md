# IMPLEMENTATION — RAG Ingestion & Retrieval Dense (v1)

## 1. Vue d'ensemble

Construire le pipeline d'ingestion du corpus SciFact (5183 docs scientifiques) et le retrieval dense par similarité cosinus. Entrée : dataset BEIR SciFact → embedding `all-MiniLM-L6-v2` (256 tokens, troncature = dette acceptée) → indexation ChromaDB (espace cosinus) → retrieval top-100 par requête. Première chose à attaquer : setup projet + chargement du dataset.

Ref décisions : `rag-eval-scifact-etape2-decisions.md` (§1, §2, §5)
Ref invariants : `.claude/rules/invariants.md`, `.claude/rules/methodologie.md`

---

## 2. Schémas cibles

Base de travail, pas contrat figé — à affiner au livrable correspondant.

### Document indexé (ChromaDB)

```python
# Chaque document dans la collection ChromaDB
{
    "id": str,              # _id du corpus BEIR (ex. "4983")
    "document": str,        # title + " " + text (concaténation)
    "embedding": list[float],  # vecteur 384d (MiniLM)
    "metadata": {
        "title": str,
        "token_count": int   # longueur en tokens MiniLM (pour error analysis troncature)
    }
}
```

### Sortie retrieval (par requête, consommée par le harness d'éval)

**Contrat figé** — référencé à l'identique dans `IMPLEMENTATION_eval_system.md`.

```python
@dataclass
class RetrievalResult:
    query_id: str
    query_text: str
    retrieved: list[dict]
    # top-100, trié par score décroissant (similarité max en premier)
    # chaque dict = {"doc_id": str, "rank": int, "score": float}
    # score = similarité cosinus directe (brute-force numpy), top-100 trié desc
```

---

## 3. Phases

### Phase 1 — Setup projet + ingestion corpus [S]

**Objectif :** Charger les 5183 docs SciFact, les embedder avec MiniLM, et les indexer dans ChromaDB avec espace cosinus.

**Livrables :**

- [ ] 1.1 — `pyproject.toml` avec dépendances : `sentence-transformers`, `chromadb`, `numpy` (pas `datasets` — chargement depuis fichiers bruts BEIR)
- [ ] 1.2 — Script `ingest.py` :
  - Charger le corpus SciFact depuis `corpus.jsonl` (fichiers bruts BEIR, mêmes que l'inspection étape 2 — pas via `datasets` HuggingFace, qui utilise une structure différente et casserait le `dataset_hash`)
  - Concaténer `title + " " + text` pour chaque document
  - Calculer `token_count = len(tokenizer.encode(title + " " + text, add_special_tokens=True))` avec `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` — IDENTIQUE à l'inspection étape 2 (tokens spéciaux inclus, sinon le seuil 256 ne correspond plus). Stocké en metadata pour l'error analysis troncature
  - Embedder avec `all-MiniLM-L6-v2` (batch, `max_seq_length=256`)
  - **Supprimer la collection existante** si présente (`delete_collection`) puis **créer une collection fraîche** avec `hnsw:space = cosine` — `hnsw:space` n'est lu qu'à la création ; `get_or_create_collection` réutilise silencieusement l'ancienne config (piège L2)
  - Vérifier après création : `collection.metadata["hnsw:space"] == "cosine"`
  - **Calculer `dataset_hash`** (hash déterministe des fichiers sources bruts) et le stocker en metadata de la collection — c'est la source de vérité, relu par `run_eval.py` sans recalcul
  - Indexer les 5183 documents avec embeddings + metadata
- [ ] 1.3 — Vérification : script ou assertion que la collection contient exactement 5183 documents

✋ Verify before continuing:
- [ ] `collection.count() == 5183` (corpus complet, pas de filtrage par qrels) — ref `.claude/rules/invariants.md`
- [ ] Collection créée FRAÎCHE en cosinus (ancienne supprimée si présente) — `hnsw:space` n'est pas modifiable après création. `collection.metadata["hnsw:space"] == "cosine"` vérifié — ref `rag-eval-scifact-etape2-decisions.md:88`
- [ ] `token_count` calculé avec `add_special_tokens=True` (comme l'inspection étape 2) et stocké en metadata pour chaque document
- [ ] `dataset_hash` calculé depuis les fichiers sources bruts et stocké en metadata de la collection

Si tout est OK : "go". Sinon dis ce qui cloche.

---

### Phase 2 — Retrieval dense top-100 [S]

**Objectif :** Pour chaque requête du split `test` (300 requêtes), embedder la requête et récupérer les 100 documents les plus similaires avec leurs scores.

**Retrieval : exact (brute-force cosinus numpy).**
ChromaDB indexe en HNSW (approximatif) ; on ne l'utilise PAS pour le ranking en v1 — l'approximation
contaminerait le Recall@100 lui-même. On tire les embeddings et on calcule le cosinus exact.
ChromaDB reste le store (persistance, token_count, dataset_hash). Décision A3 : OPTION 1 (actée).

**Livrables :**

- [ ] 2.1 — Script `retrieve.py` :
  - Charger les requêtes SciFact split `test` (300) depuis `queries.jsonl` (fichiers bruts BEIR)
  - Récupérer les 5183 embeddings de docs depuis la collection ChromaDB (`collection.get(include=["embeddings"])`) + leurs doc_id
  - Embedder chaque requête avec le même modèle MiniLM
  - L2-normaliser requêtes ET docs, puis cosinus = produit scalaire (matriciel numpy : (300 x 384) . (384 x 5183) → matrice 300 x 5183 de similarités)
  - Pour chaque requête : top-100 par similarité DÉCROISSANTE → `{doc_id, rank, score}`, `score` = similarité cosinus directe (rang 1 = similarité max)
  - Charger les qrels du split `test` depuis `qrels/test.tsv`
- [ ] 2.2 — Sortie structurée : liste de `RetrievalResult` (ou équivalent dict) prête à être consommée par le harness d'éval
- [ ] 2.3 — **Relire le `dataset_hash`** depuis la metadata de la collection ChromaDB (calculé et stocké par `ingest.py`) pour l'inclure dans les artefacts du run

✋ Verify before continuing:
- [ ] 300 requêtes traitées, chacune avec exactement 100 résultats ordonnés par score décroissant
- [ ] `score` = similarité cosinus calculée en numpy (PAS de distance ChromaDB manipulée) ; 2 vecteurs identiques normalisés → similarité 1.0 ; rang 1 = similarité max
- [ ] Retrieval EXACT : le top-100 est le vrai top-100 (brute-force), pas l'approximation HNSW
- [ ] Le corpus interrogé est le même corpus de 5183 docs (pas de re-filtrage)
- [ ] Déterminisme vérifié : même index → mêmes top-100 sur 2 runs (garanti par le retrieval exact)

Si tout est OK : "go". Sinon dis ce qui cloche.

---

## 4. Livrables détaillés

| # | Livrable | Done = | Taille |
|---|----------|--------|--------|
| 1.1 | `pyproject.toml` | dépendances installables, `pip install -e .` fonctionne | XS |
| 1.2 | `ingest.py` | 5183 docs indexés dans ChromaDB cosinus, avec token_count en metadata | S |
| 1.3 | Vérification ingestion | assertion 5183 docs + espace cosinus confirmé | XS |
| 2.1 | `retrieve.py` | 300 requêtes → 100 résultats chacune avec doc_id/rank/score | S |
| 2.2 | Sortie structurée retrieval | format consommable par le harness d'éval | XS |
| 2.3 | `dataset_hash` (relecture) | hash relu depuis metadata collection (calculé à l'ingestion) | XS |

---

## 5. Dépendances critiques

- Livrable 2.1 (`retrieve.py`) bloque le harness d'éval → ne peut tourner qu'après ingestion complète (1.2).
- Le harness d'éval (cf. `IMPLEMENTATION_eval_system.md`) consomme la sortie de 2.2. **Contrat figé** (§2 ci-dessus) :
  ```
  RetrievalResult:
    query_id: str
    query_text: str
    retrieved: list[{"doc_id": str, "rank": int, "score": float}]
    # score = similarité cosinus directe (brute-force numpy), top-100 trié desc
  ```

---

## 6. Garde-fous

- Si les vecteurs ne sont pas L2-normalisés avant le produit scalaire → ce n'est pas du cosinus mais du dot-product brut, ranking faussé. Normaliser requêtes ET docs.
- Si `token_count` n'est pas calculé avec `add_special_tokens=True` et le tokenizer MiniLM exact → les corrélations échec/troncature seront fausses sur les docs borderline (~256 tokens). Utiliser `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")`, identique à l'inspection étape 2.
