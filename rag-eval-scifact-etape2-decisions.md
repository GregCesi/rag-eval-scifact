# RAG-Eval SciFact — Étape 2 : décisions dataset + v1 gelée

> **Artefact décisionnel + transfert.** Sortie de l'étape 2 (analyse SciFact + résolution des points ouverts §4).
> **Destinataire :** chat neuf de l'étape 3 (production du `.claude/` + `IMPLEMENTATION.md`), et l'agent d'implémentation en aval.
> **Relation au cadrage :** ce document **ne recopie pas** les invariants de `rag-eval-scifact-cadrage.md` (§1–3, §6). Il **pointe** vers eux (doctrine : pointeurs, pas copie) et se contente de **résoudre la §4** + figer le périmètre v1 sur la base des faits mesurés.
>
> État à l'écriture : étape 2 close, §4 intégralement résolue. Point d'entrée de l'étape 3.

---

## 1. Faits mesurés sur SciFact (recon Phase 0)

Mesurés par `inspect_scifact.py` sur le dataset BEIR réel — **pas** issus du leaderboard de mémoire.

**Longueur des documents** (unité `title + text`, tokenizer MiniLM exact) :
- 5183 documents.
- Médiane **316** tokens, moyenne 337, p90 **502**, p95 567, max 1939.
- **71,0 % des docs dépassent 256 tokens** (le `max_seq_length` de `all-MiniLM-L6-v2`) → tronqués en silence.
- Histogramme : ≤128 : 2,1 % · 129–256 : 26,9 % · 257–384 : 40,0 % · 385–512 : 22,3 % · >512 : 8,8 %.
- Conséquence chiffrée pour le backlog : passer la fenêtre à **512 ramènerait la troncature de 71 % à 8,8 %**.

**Docs pertinents par requête** (qrels) :
- Split `test` : 300 requêtes, **92,3 % à 1 seul doc pertinent**, moyenne 1,13, max 5.
- Split `train` : 809 requêtes, 90,5 % à 1 doc, moyenne 1,14, max 5.
- Tâche nette : pour ~9 requêtes sur 10, il y a **un unique document à retrouver**.

---

## 2. Point #1 — Chunking / unité de retrieval → **RÉSOLU**

- **Unité de retrieval = le document entier** (`title + text`). Qrels au niveau document. Aucun sous-découpage scoré séparément (casserait l'alignement qrels + la comparabilité leaderboard).
- **Embedding v1 = MiniLM par défaut, fenêtre 256, troncature ASSUMÉE.** Motif : c'est le **point de mesure de départ** (le zéro du thermomètre), et c'est la config standard sous laquelle BEIR évalue les sentence-transformers → baseline comparable, pas cassée.
- La troncature est une **dette explicite**, inscrite dans `RESULTS.md`. On tronque *parce que c'est la baseline*, pas parce que c'est bien.
- **Chunking hors v1.** Pré-enregistré comme levier, **gated** sur une condition mesurée : « les docs longs ratent réellement le retrieval » (à prouver via l'error analysis de v1, pas supposé).
- **Préférence de sortie de dette (Grégoire) : changer de modèle d'embedding** (le dataset veut retrouver LE doc entier → un modèle qui lit le doc entier est plus fidèle à la tâche qu'un découpage). Notée comme candidate, **non gelée** : le levier effectif sera désigné par l'error analysis.
- Recon « evidence-survival » (offset des phrases-évidence vs 256) **abandonnée** : redondante une fois v1 mesurée — l'échec de retrieval réel répond directement à la question, sans passer par un proxy en tokens.

## 3. Point #2 — Choix des k → **RÉSOLU**

Distinction structurante actée (survit au projet) :
- **k-de-mesure** = jeu de sondes, pluriel, pour l'éval : on observe **la même sortie classée** à plusieurs profondeurs.
- **k-de-coupe** = chiffre unique, opérationnel, pour un vrai RAG en prod. **Ne jamais fusionner les deux.**

Jeu de métriques v1 (justifié par « 92 % des requêtes à 1 doc ») :

| Métrique | Répond à | Rôle |
|---|---|---|
| Recall@1 | juste du premier coup ? | a du sens ici (1 doc pertinent) |
| Recall@5 | dans le budget RAG réel ? | k opérationnel |
| Recall@10 | filet + comparable | standard |
| **Recall@100** | seulement trouvable ? | **juge de la dette de troncature** (docs >256 vs ≤256) |
| **nDCG@10** | bien classé ? (0–1) | **métrique leaderboard BEIR, situe vs état de l'art** |
| **MRR** | à quel rang moyen ? | **lisible (`1/MRR` = rang moyen), error analysis** |

- **R-precision écartée** : sur SciFact (90 % à 1 doc) elle se confondrait avec Recall@1. Notée comme pertinente pour un futur dataset multi-docs.
- nDCG@10 et MRR sont ~0,95 corrélés ici (1 seul bon doc) mais gardés **tous les deux** : nDCG pour se situer (audience externe), MRR pour se comprendre (audience interne).

## 4. Point #3 — Format `results/*.json` → **RÉSOLU**

Principe directeur : le JSON doit permettre de répondre, **sans relancer le pipeline**, à « pourquoi cette requête a raté ? ».

**Niveau run** (carte d'identité, une fois) :
- `version` (tag git, ex. `v1-dense`), `date`
- **`dataset_hash`** (garantit que deux runs portent sur les mêmes données)
- config embedding : `model`, **`max_seq_length`** (témoin de la dette : passera à 512 le jour venu), `dim`
- les 6 métriques agrégées (doublon machine ; `RESULTS.md` = vue humaine)

**Niveau requête** (un objet par requête, cœur de l'error analysis) :
- `query_id` + texte de la requête
- **docs attendus** (`_id` des qrels) **avec leur longueur en tokens** ← permet de corréler échec ↔ troncature ; sans ça, dette non jugeable
- **top-100 ramenés** : `_id` + **rang** + **score de similarité** (le score dit *à quel point* le système s'est trompé)
- métriques **de cette requête** (found@k, rang du bon doc)

- **Profondeur de log = top-100**, aligné sur le Recall max. Au-delà on ne mesure rien → on ne stocke rien. (~300 × 100 = 30 000 lignes/run, négligeable en git.)
- Un fichier JSON par run, horodaté/tagué, commité avec le même tag que le code.

## 5. Point #4 — Périmètre v1 → **RÉSOLU**

**DANS v1 :**
- Ingestion : `corpus.jsonl` (5183) → embed `title+text` (MiniLM, 256, tronqué) → ChromaDB.
- Retrieval : embed requête → similarité **cosinus** → liste classée **top-100**.
- **Harness d'éval fait-main** (métriques calculées soi-même, cf. cadrage §2.6) : Recall@{1,5,10,100} + nDCG@10 + MRR contre les qrels. **← c'est LE livrable.**
- Artefacts : `RESULTS.md` (append-only) + `results/*.json` + `dataset_hash`, commités sous tag `v1-dense`.

**Nœuds tranchés (sinon résolus mal par défaut dans le code) :**
- **A — split d'éval : `test` (300).** Retrieval zero-shot, comparable au leaderboard. `train` = sanity-check optionnel, hors chiffre officiel.
- **B — corpus indexé : les 5183, toujours**, quel que soit le split (protocole BEIR ; les distracteurs font partie de la tâche).
- **C — similarité cosinus explicite. `✋ Verify before continue`** : vérifier que ChromaDB n'est pas en L2 par défaut *avant* toute mesure (config `hnsw:space = cosine` ou normalisation des vecteurs). MiniLM est entraîné pour le cosinus ; le L2 par défaut classerait moins bien **sans lever d'erreur** → baseline silencieusement faussée. (Même nature que la leçon XSS/DOMPurify : décision négative qui doit devenir un critère de vérification explicite.)

**HORS v1** (leviers pré-enregistrés, activés seulement si l'error analysis les désigne) :
BM25 · fusion RRF · reranking · LLM de génération · sortie de dette troncature (512 / autre modèle / chunking) · front / Inspector.

**Backlog de sortie de dette, par coût croissant** (l'error analysis désigne lequel, pas l'intuition) :
1. `max_seq_length` → 512 (une ligne ; 71 % → 8,8 % de troncature ; qualité MiniLM sur 256–512 à **mesurer**).
2. autre modèle d'embedding à fenêtre longue (candidat préféré de Grégoire ; change la brique centrale).
3. chunk-for-embedding + remap chunk→doc (le plus lourd ; réalignement au scoring).

---

## 6. Critères `✋ Verify` pour l'IMPLEMENTATION (récap)

- `✋` **espace de similarité = cosinus**, pas L2 (Nœud C).
- `✋` **corpus indexé = 5183 docs entiers**, pas un sous-ensemble filtré par qrels (Nœud B).
- `✋` **troncature notée comme dette** dans `RESULTS.md` (pas silencieuse).
- `✋` **métriques calculées à la main**, pas déléguées à une lib d'éval clé-en-main (cadrage §2.6).
- `✋` **run non commité = run inexistant** : `RESULTS.md` + `results/*.json` produits et tagués à chaque run (cadrage §2.4).

---

## 7. Ce qui reste pour l'étape 3

Tout le cadrage dataset est gelé. Restent des décisions **d'implémentation** (pas de dataset), à trancher en ouvrant l'étape 3 :
- **Un seul `IMPLEMENTATION.md` ou séparé (éval / embed+dense) ?** (cf. cadrage §5.3.2 — décision de séquençage, appartient au kickoff.)
- Contenu exact du `.claude/` encodant les invariants (§2 du cadrage) + les 5 critères `✋` ci-dessus.

**Entrée de l'étape 3 :** `rag-eval-scifact-cadrage.md` (invariants) **+** ce document (décisions + faits). **Chat neuf.**
