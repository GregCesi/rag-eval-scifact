Campagne : v5-rerankers

## confirme

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| qwen3-passages-sans-reranker | 124 | 0.7984 | 0.9758 | 1.0000 |
| rerank-bge-m3-top20 | 124 | 0.8548 | 0.9839 | 1.0000 |
| rerank-medcpt-top20 | 124 | 0.8468 | 0.9758 | 1.0000 |
| rerank-minilm-top20 | 124 | 0.8306 | 0.9677 | 1.0000 |
| rerank-qwen3-0.6b-top20 | 124 | 0.8710 | 0.9839 | 1.0000 |
| rerank-qwen3-4b-top20 | 124 | 0.8548 | 0.9839 | 1.0000 |

## contredit

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| qwen3-passages-sans-reranker | 64 | 0.6719 | 0.9844 | 1.0000 |
| rerank-bge-m3-top20 | 64 | 0.7656 | 1.0000 | 1.0000 |
| rerank-medcpt-top20 | 64 | 0.7656 | 0.9844 | 1.0000 |
| rerank-minilm-top20 | 64 | 0.7031 | 0.9844 | 1.0000 |
| rerank-qwen3-0.6b-top20 | 64 | 0.7656 | 1.0000 | 1.0000 |
| rerank-qwen3-4b-top20 | 64 | 0.8906 | 0.9844 | 1.0000 |

## sans preuve

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| qwen3-passages-sans-reranker | 112 | 0.3304 | 0.6518 | 0.8750 |
| rerank-bge-m3-top20 | 112 | 0.3125 | 0.6875 | 0.8750 |
| rerank-medcpt-top20 | 112 | 0.4018 | 0.7321 | 0.8750 |
| rerank-minilm-top20 | 112 | 0.2411 | 0.6607 | 0.8750 |
| rerank-qwen3-0.6b-top20 | 112 | 0.4018 | 0.7054 | 0.8750 |
| rerank-qwen3-4b-top20 | 112 | 0.3661 | 0.7411 | 0.8750 |

## nDCG@10 par ensemble de claims

| run | 300 claims | 188 claims avec preuve | 112 claims sans preuve |
|---|---|---|---|
| qwen3-passages-sans-reranker | 0.7319 | 0.8728 | 0.4857 |
| rerank-bge-m3-top20 | 0.7443 | 0.9044 | 0.4821 |
| rerank-medcpt-top20 | 0.7738 | 0.8996 | 0.5577 |
| rerank-minilm-top20 | 0.7114 | 0.8801 | 0.4358 |
| rerank-qwen3-0.6b-top20 | 0.7690 | 0.9103 | 0.5391 |
| rerank-qwen3-4b-top20 | 0.7836 | 0.9266 | 0.5418 |

## Écart de nDCG@10 face à la référence qwen3-passages-sans-reranker (188 claims avec preuve)

| run | Δ nDCG@10 | p |
|---|---|---|
| rerank-bge-m3-top20 | +0.0316 | 0.0163 |
| rerank-medcpt-top20 | +0.0268 | 0.0579 |
| rerank-minilm-top20 | +0.0073 | 0.6741 |
| rerank-qwen3-0.6b-top20 | +0.0375 | 0.0092 |
| rerank-qwen3-4b-top20 | +0.0538 | 0.0001 |

Lecture (`.claude/rules/methodologie.md`) : « avec plusieurs dizaines de runs, des écarts « significatifs » apparaissent par hasard. On lit les gros écarts, pas les 0,01. »

## Niveaux de lecture par groupe

### qwen3-passages-sans-reranker

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 60/60 | 1.0000 | 60/60 | 1.0000 |
| Vocabulaire | 51/55 | 0.9273 | 55/55 | 1.0000 |
| Raisonnement | 63/71 | 0.8873 | 70/71 | 0.9859 |
| Avec preuve, ne répond pas | 11/18 | 0.6111 | 14/18 | 0.7778 |
| Sans preuve, répond | 17/22 | 0.7727 | 18/22 | 0.8182 |
| Sans preuve, ne répond pas | 60/105 | 0.5714 | 67/105 | 0.6381 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### rerank-bge-m3-top20

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 57/60 | 0.9500 | 59/60 | 0.9833 |
| Vocabulaire | 53/55 | 0.9636 | 55/55 | 1.0000 |
| Raisonnement | 64/71 | 0.9014 | 71/71 | 1.0000 |
| Avec preuve, ne répond pas | 13/18 | 0.7222 | 15/18 | 0.8333 |
| Sans preuve, répond | 14/22 | 0.6364 | 18/22 | 0.8182 |
| Sans preuve, ne répond pas | 55/105 | 0.5238 | 70/105 | 0.6667 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### rerank-medcpt-top20

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 57/60 | 0.9500 | 60/60 | 1.0000 |
| Vocabulaire | 54/55 | 0.9818 | 54/55 | 0.9818 |
| Raisonnement | 64/71 | 0.9014 | 68/71 | 0.9577 |
| Avec preuve, ne répond pas | 14/18 | 0.7778 | 15/18 | 0.8333 |
| Sans preuve, répond | 17/22 | 0.7727 | 17/22 | 0.7727 |
| Sans preuve, ne répond pas | 65/105 | 0.6190 | 77/105 | 0.7333 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### rerank-minilm-top20

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 59/60 | 0.9833 | 60/60 | 1.0000 |
| Vocabulaire | 51/55 | 0.9273 | 53/55 | 0.9636 |
| Raisonnement | 61/71 | 0.8592 | 64/71 | 0.9014 |
| Avec preuve, ne répond pas | 13/18 | 0.7222 | 16/18 | 0.8889 |
| Sans preuve, répond | 14/22 | 0.6364 | 17/22 | 0.7727 |
| Sans preuve, ne répond pas | 50/105 | 0.4762 | 68/105 | 0.6476 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### rerank-qwen3-0.6b-top20

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 58/60 | 0.9667 | 60/60 | 1.0000 |
| Vocabulaire | 54/55 | 0.9818 | 55/55 | 1.0000 |
| Raisonnement | 66/71 | 0.9296 | 70/71 | 0.9859 |
| Avec preuve, ne répond pas | 12/18 | 0.6667 | 15/18 | 0.8333 |
| Sans preuve, répond | 14/22 | 0.6364 | 18/22 | 0.8182 |
| Sans preuve, ne répond pas | 60/105 | 0.5714 | 73/105 | 0.6952 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### rerank-qwen3-4b-top20

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 60/60 | 1.0000 | 60/60 | 1.0000 |
| Vocabulaire | 55/55 | 1.0000 | 55/55 | 1.0000 |
| Raisonnement | 68/71 | 0.9577 | 71/71 | 1.0000 |
| Avec preuve, ne répond pas | 13/18 | 0.7222 | 14/18 | 0.7778 |
| Sans preuve, répond | 16/22 | 0.7273 | 18/22 | 0.8182 |
| Sans preuve, ne répond pas | 65/105 | 0.6190 | 77/105 | 0.7333 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

