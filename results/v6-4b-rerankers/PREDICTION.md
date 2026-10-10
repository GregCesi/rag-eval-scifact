# Prédiction — campagne v6-4b-rerankers

Écrite le 10 octobre 2026 par Grégoire, à l'instinct, avant tout run.

Base fixe : Qwen3-Embedding-4B, passages, dense seul.
Les rerankers relisent les 20 premiers documents.
Repères en nDCG@10 (188 affirmations avec preuve / 300) :
Qwen 4B seul 0,9070 / 0,7912 (v4-leviers) ;
Qwen 0.6B + Qwen3-Reranker-0.6B 0,9103 / 0,7690 ;
Qwen 0.6B + Qwen3-Reranker-4B 0,9266 / 0,7836 (v5-rerankers).

| Question | Prédiction |
|---|---|
| Qwen 4B + Reranker 4B dépasse-t-il 0,9266 (le même reranker sur le 0.6B) ? | Oui. |
| Qwen 4B + Reranker 0.6B, au-dessus ou en dessous du 4B seul (0,9070) ? | Un peu au-dessus du 4B seul. |
| Sur les 300, Qwen 4B + Reranker 4B passe-t-il devant le 4B seul (0,7912) ? | Oui, largement. |
| L'écart entre les deux rerankers sur le 4B, plus grand ou plus petit que sur le 0.6B (+0,016) ? | Plus petit. Le reranker corrige les défauts du premier étage ; le 4B en a moins, il reste moins à corriger. |

Crainte principale : le reranker n'améliore que légèrement. Il corrige les
mêmes défauts que sur le 0.6B, mais le Qwen 4B en a déjà supprimé une partie ;
le reranker ne peut corriger que la différence.

Espoir : le Qwen 4B remonte plus de bons documents dans ses 20 premiers, ce qui
donne au reranker plus de bon contenu à faire monter.
