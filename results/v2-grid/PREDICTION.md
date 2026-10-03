# Prédiction — campagne v2-grid

Écrite par Grégoire le 3 octobre 2026, avant le premier run de la campagne.
Règle : `.claude/rules/methodologie.md` — la campagne vérifie la prédiction, elle ne la remplace pas.

## Ce que j'ai déjà vu

Cinq nDCG@10 de runs de vérification (campagne `dev`, non commités) m'ont été montrés avant d'écrire :
v1 MiniLM tronqué 0,6451 · BM25 document 0,6549 · Qwen abstract entier 0,6996 ·
hybride RRF MiniLM tronqué 0,6800 · MiniLM tronqué + reranker 0,6892.
Ma prédiction est biaisée pour ces cinq combinaisons ; elle porte surtout sur le reste.

## Rappel de l'error analysis v1

300 requêtes : 151 perfect (rang 1), 87 near_miss (top 10 hors rang 1),
40 deep_miss (top 100 hors top 10), 22 miss_100 (hors top 100). 71 % des documents tronqués à 256 tokens.

## 1. Le gagnant

Aucune idée arrêtée. Mon penchant de départ : le dense sur passages de 128 tokens, des morceaux de
document plus précis pour un retrieval dense très bon avec les termes scientifiques.

Mais je doute de ce penchant : vu les bons résultats déjà obtenus avec la troncature, j'ai peur que
l'unité (tronqué ou passages) ne change pas grand-chose au score. Le changement de modèle pourrait
faire plus de bien.

Ordre de grandeur du meilleur nDCG@10 : 0,75 avec un peu de chance. 0,80 est impossible avec ces
stratégies et ce dataset un peu particulier.

## 2. Le levier qui compte le plus

Le modèle. Je ne connais pas la différence entre MiniLM et Qwen ni l'avantage de l'un sur l'autre, mais
un modèle plus gros doit mieux connaître les données et le sens des termes scientifiques. À longueur
égale (tous deux tronqués à 256 tokens), je présuppose que Qwen fait mieux que MiniLM.

BM25 ne sera pas tellement utile, mais il comblera quand même quelques lacunes dans certains cas.

Le vrai levier, à mon avis, n'est pas dans la grille : quelque chose de plus spécialisé dans la
compréhension scientifique. Je ne sais pas où cela se joue, dans le modèle d'embedding ou dans le retrieve.

## 3. Les passages

Peu d'écart attendu entre passages de 128 tokens et troncature à 256 (voir section 1). Pas de
prédiction séparée par modèle, ni sur passages contre abstract entier pour Qwen.

## 4. L'hybride

Gain faible : BM25 comble quelques lacunes, sans plus. L'union simple fera moins bien que RRF.

## 5. Le reranker

Un reranker est toujours mieux, quelle que soit la stratégie de premier étage : il améliore les 17
retrievers, aucun n'est dégradé.

## 6. Les buckets v1

Ce qu'il faut vraiment corriger, ce sont les deep_miss et les miss_100. Il me paraît impossible que le
golden ne soit pas dans les 10 premiers, et encore moins qu'il ne soit pas dans les 100 premiers.

L'objectif raisonnable est d'avoir le golden dans les 5 premiers, pas forcément premier à chaque fois.
Le projet n'a pas d'usage derrière (pas de chatbot), donc la cible reste difficile à fixer.

- near_miss (87) : pas une priorité, le golden est déjà dans le top 10.
- deep_miss (40) : à corriger en priorité. Pas de levier désigné.
- miss_100 (22) : devrait tomber à 0 avec la meilleure stratégie. Pas de levier désigné.

Limite que j'ai vue pendant l'error analysis v1 : beaucoup de claims où plusieurs documents répondent
aussi bien que le golden, et d'autres où le golden lui-même ne répond pas parfaitement. Sur ceux-là,
il sera difficile de faire quelque chose, quel que soit le levier.

## 7. Ce qui me ferait dire que j'avais tort

Aucun résultat ne me choquerait vraiment, sauf un nDCG@10 autour de 0,9 : trop haut au vu de la
qualité du dataset.
