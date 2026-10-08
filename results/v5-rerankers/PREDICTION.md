# Prédiction — campagne v5-rerankers

Écrite le 8 octobre 2026 par Grégoire, à l'instinct, avant tout run.

Base fixe : Qwen3-Embedding-0.6B, passages, dense seul.
Les rerankers relisent les 20 premiers documents.
Repères sur les 188 affirmations avec preuve (nDCG@10) :
référence sans reranker 0,8728, Qwen3-Embedding-4B seul 0,9070 (v4-leviers).

| Question | Prédiction |
|---|---|
| MiniLM de v2, sur 20 documents | Presque rien sur les « confirme ». Un peu, pas trop, sur les « contredit » (sceptique). Fait tout descendre sur les « sans preuve ». |
| Meilleur des quatre nouveaux rerankers | Qwen3-Reranker-4B. MedCPT Cross-Encoder pas si mauvais. |
| Un reranker sur le 0.6B atteint-il le 4B seul (0,9070) ? | Non. Mais Qwen3-Reranker-4B fait bien monter le 0.6B : le reranker est utile pour la première fois. |
| MedCPT au-dessus ou en dessous de bge-reranker-v2-m3 ? | Au-dessus. Sinon, les rerankers science ne valent rien. |

Levier attendu gagnant : Qwen3-Reranker-4B.

Hypothèse de départ (v4-leviers) : le gain du Qwen 4B est un gain de tri ;
un reranker plus faible que le premier étage dégrade, c'était le problème de v2-grid.
