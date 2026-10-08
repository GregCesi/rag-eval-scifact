Campagne : v4-leviers

## confirme

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| medcpt-passages | 124 | 0.7581 | 0.9839 | 1.0000 |
| qwen3-4b-passages | 124 | 0.8306 | 0.9919 | 1.0000 |
| qwen3-passages-hyde | 124 | 0.7984 | 0.9597 | 1.0000 |
| qwen3-passages-reference | 124 | 0.7984 | 0.9758 | 1.0000 |
| qwen3-passages-sans-instruction | 124 | 0.7984 | 0.9677 | 1.0000 |

## contredit

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| medcpt-passages | 64 | 0.7500 | 0.9688 | 1.0000 |
| qwen3-4b-passages | 64 | 0.8281 | 0.9844 | 1.0000 |
| qwen3-passages-hyde | 64 | 0.7656 | 0.9531 | 1.0000 |
| qwen3-passages-reference | 64 | 0.6719 | 0.9844 | 1.0000 |
| qwen3-passages-sans-instruction | 64 | 0.7031 | 0.9688 | 1.0000 |

## sans preuve

| run | n | rang 1 | top 10 | top 100 |
|---|---|---|---|---|
| medcpt-passages | 112 | 0.3036 | 0.7232 | 0.9286 |
| qwen3-4b-passages | 112 | 0.4018 | 0.8214 | 0.9286 |
| qwen3-passages-hyde | 112 | 0.3482 | 0.6696 | 0.9107 |
| qwen3-passages-reference | 112 | 0.3304 | 0.6518 | 0.8750 |
| qwen3-passages-sans-instruction | 112 | 0.2589 | 0.6161 | 0.8214 |

## nDCG@10 par ensemble de claims

| run | 300 claims | 188 claims avec preuve | 112 claims sans preuve |
|---|---|---|---|
| medcpt-passages | 0.7281 | 0.8622 | 0.4995 |
| qwen3-4b-passages | 0.7912 | 0.9070 | 0.5940 |
| qwen3-passages-hyde | 0.7401 | 0.8748 | 0.5031 |
| qwen3-passages-reference | 0.7319 | 0.8728 | 0.4857 |
| qwen3-passages-sans-instruction | 0.7025 | 0.8670 | 0.4223 |

## Écart de nDCG@10 face à la référence qwen3-passages-reference (188 claims avec preuve)

| run | Δ nDCG@10 | p |
|---|---|---|
| medcpt-passages | -0.0106 | 0.5733 |
| qwen3-4b-passages | +0.0342 | 0.0018 |
| qwen3-passages-hyde | +0.0020 | 0.8744 |
| qwen3-passages-sans-instruction | -0.0057 | 0.6161 |

Lecture (`.claude/rules/methodologie.md`) : « avec plusieurs dizaines de runs, des écarts « significatifs » apparaissent par hasard. On lit les gros écarts, pas les 0,01. »

## Niveaux de lecture par groupe

### medcpt-passages

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 53/60 | 0.8833 | 56/60 | 0.9333 |
| Vocabulaire | 53/55 | 0.9636 | 54/55 | 0.9818 |
| Raisonnement | 60/71 | 0.8451 | 67/71 | 0.9437 |
| Avec preuve, ne répond pas | 13/18 | 0.7222 | 15/18 | 0.8333 |
| Sans preuve, répond | 18/22 | 0.8182 | 19/22 | 0.8636 |
| Sans preuve, ne répond pas | 59/105 | 0.5619 | 73/105 | 0.6952 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### qwen3-4b-passages

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 59/60 | 0.9833 | 60/60 | 1.0000 |
| Vocabulaire | 55/55 | 1.0000 | 55/55 | 1.0000 |
| Raisonnement | 65/71 | 0.9155 | 68/71 | 0.9577 |
| Avec preuve, ne répond pas | 14/18 | 0.7778 | 17/18 | 0.9444 |
| Sans preuve, répond | 19/22 | 0.8636 | 21/22 | 0.9545 |
| Sans preuve, ne répond pas | 72/105 | 0.6857 | 83/105 | 0.7905 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### qwen3-passages-hyde

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 60/60 | 1.0000 | 60/60 | 1.0000 |
| Vocabulaire | 50/55 | 0.9091 | 53/55 | 0.9636 |
| Raisonnement | 64/71 | 0.9014 | 66/71 | 0.9296 |
| Avec preuve, ne répond pas | 13/18 | 0.7222 | 14/18 | 0.7778 |
| Sans preuve, répond | 16/22 | 0.7273 | 17/22 | 0.7727 |
| Sans preuve, ne répond pas | 58/105 | 0.5524 | 69/105 | 0.6571 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### qwen3-passages-reference

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 60/60 | 1.0000 | 60/60 | 1.0000 |
| Vocabulaire | 51/55 | 0.9273 | 55/55 | 1.0000 |
| Raisonnement | 63/71 | 0.8873 | 70/71 | 0.9859 |
| Avec preuve, ne répond pas | 11/18 | 0.6111 | 14/18 | 0.7778 |
| Sans preuve, répond | 17/22 | 0.7727 | 18/22 | 0.8182 |
| Sans preuve, ne répond pas | 60/105 | 0.5714 | 67/105 | 0.6381 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

### qwen3-passages-sans-instruction

| groupe | 5 premiers | part (5 premiers) | 10 premiers | part (10 premiers) |
|---|---|---|---|---|
| Direct | 58/60 | 0.9667 | 59/60 | 0.9833 |
| Vocabulaire | 51/55 | 0.9273 | 54/55 | 0.9818 |
| Raisonnement | 64/71 | 0.9014 | 69/71 | 0.9718 |
| Avec preuve, ne répond pas | 11/18 | 0.6111 | 15/18 | 0.8333 |
| Sans preuve, répond | 14/22 | 0.6364 | 17/22 | 0.7727 |
| Sans preuve, ne répond pas | 50/105 | 0.4762 | 63/105 | 0.6000 |
| Non jugé | 8/8 | 1.0000 | 8/8 | 1.0000 |

