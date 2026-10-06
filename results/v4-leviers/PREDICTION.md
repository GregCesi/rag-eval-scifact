# Prédiction — campagne v4-leviers

Écrite le 7 octobre 2026 par Grégoire, à l'instinct, avant tout run.

Référence : Qwen3-Embedding-0.6B, passages, dense seul, sans reranker.
nDCG@10 0,7319 sur les 300 affirmations, 0,8728 sur les 188 avec preuve.
Les écarts se lisent sur les 188 affirmations avec preuve.

| Levier | Prédiction |
|---|---|
| Sans instruction de requête | Sans avis |
| HyDE | Moins bien, au mieux aussi bien |
| Qwen3-Embedding-4B | Un peu mieux ; +0,02 serait déjà excellent |
| MedCPT | En dessous de la référence |

Levier attendu gagnant : Qwen 4B, de peu.
Goldens « raisonnement » dans les 5 premiers (référence 63 sur 71) : sans avis.

Hypothèse de départ, posée la veille : le vrai levier est HyDE ou le modèle plus gros.
Au moment de prédire, Grégoire ne croit plus à HyDE.
