Campagne : v3-juge

## Juges

### local

- paires jugées : 438
- SUPPORTS : 187
- REFUTES : 171
- NOT_ENOUGH_INFO : 79
- illisibles : 1
- citations introuvables : 35
- refus du juge : 0
- durée moyenne par jugement : 14.0134 s

### claude

- paires jugées : 438
- SUPPORTS : 144
- REFUTES : 99
- NOT_ENOUGH_INFO : 186
- illisibles : 3
- citations introuvables : 3
- refus du juge : 6
- durée moyenne par jugement : 6.7307 s

### etapes

absent

## Jeu de données

### local

- part des documents attendus jugés SUPPORTS ou REFUTES : 0.8437
- part des documents non attendus classés devant jugés SUPPORTS ou REFUTES : 0.7273
- claims avec au moins un document non attendu jugé SUPPORTS ou REFUTES : 37
- claims sans aucun document attendu jugé SUPPORTS ou REFUTES : 49

### claude

- part des documents attendus jugés SUPPORTS ou REFUTES : 0.6136
- part des documents non attendus classés devant jugés SUPPORTS ou REFUTES : 0.3535
- claims avec au moins un document non attendu jugé SUPPORTS ou REFUTES : 27
- claims sans aucun document attendu jugé SUPPORTS ou REFUTES : 113

## Étiquette d'origine (documents attendus)

### local

- tableau croisé (verdict / étiquette d'origine) :
  - SUPPORTS / SUPPORT : 106
  - SUPPORTS / CONTRADICT : 14
  - SUPPORTS / SANS_PREUVE : 35
  - REFUTES / SUPPORT : 30
  - REFUTES / CONTRADICT : 57
  - REFUTES / SANS_PREUVE : 44
  - NOT_ENOUGH_INFO / SUPPORT : 1
  - NOT_ENOUGH_INFO / CONTRADICT : 0
  - NOT_ENOUGH_INFO / SANS_PREUVE : 51
- paires valides (dénominateur) : 338
- part d'accord : 0.6331

### claude

- tableau croisé (verdict / étiquette d'origine) :
  - SUPPORTS / SUPPORT : 114
  - SUPPORTS / CONTRADICT : 1
  - SUPPORTS / SANS_PREUVE : 9
  - REFUTES / SUPPORT : 6
  - REFUTES / CONTRADICT : 65
  - REFUTES / SANS_PREUVE : 13
  - NOT_ENOUGH_INFO / SUPPORT : 15
  - NOT_ENOUGH_INFO / CONTRADICT : 3
  - NOT_ENOUGH_INFO / SANS_PREUVE : 105
- paires valides (dénominateur) : 331
- part d'accord : 0.8580

## Paires refusées par le juge

### local

aucun refus

### claude

- 115:33872649 : famille attendu, étiquette CONTRADICT
- 314:2638387 : famille devant, étiquette 
- 478:14767844 : famille attendu, étiquette SANS_PREUVE
- 507:30774694 : famille attendu, étiquette SANS_PREUVE
- 1020:9433958 : famille attendu, étiquette SUPPORT
- 1021:9433958 : famille attendu, étiquette CONTRADICT

## Documents classés devant

### local

- SUPPORTS : 35
- REFUTES : 43
- NOT_ENOUGH_INFO : 27
- claims avec au moins un document devant jugé SUPPORTS ou REFUTES : 39
- liste de ces claims : 141, 148, 183, 212, 239, 261, 274, 295, 314, 343, 385, 431, 475, 513, 569, 598, 619, 674, 742, 744, 756, 759, 793, 845, 852, 859, 971, 993, 1019, 1041, 1049, 1107, 1110, 1132, 1180, 1259, 1274, 1382, 1385

### claude

- SUPPORTS : 21
- REFUTES : 17
- NOT_ENOUGH_INFO : 66
- claims avec au moins un document devant jugé SUPPORTS ou REFUTES : 28
- liste de ces claims : 148, 183, 261, 274, 295, 314, 343, 385, 431, 513, 598, 619, 674, 744, 756, 759, 793, 859, 971, 1019, 1041, 1049, 1107, 1110, 1132, 1180, 1259, 1385

## Niveau de lecture

### local

- documents attendus (SUPPORT ou CONTRADICT) :
  - DIRECT : 48
  - VOCABULARY : 159
  - REASONING : 0
- documents devant :
  - DIRECT : 6
  - VOCABULARY : 72
  - REASONING : 0

### claude

- documents attendus (SUPPORT ou CONTRADICT) :
  - DIRECT : 60
  - VOCABULARY : 55
  - REASONING : 71
- documents devant :
  - DIRECT : 4
  - VOCABULARY : 12
  - REASONING : 22

## Accord entre juges

### local / claude

- paires communes : 428
- tableau croisé (3 × 3) :
  - NOT_ENOUGH_INFO / NOT_ENOUGH_INFO : 75
  - NOT_ENOUGH_INFO / REFUTES : 2
  - NOT_ENOUGH_INFO / SUPPORTS : 1
  - REFUTES / NOT_ENOUGH_INFO : 60
  - REFUTES / REFUTES : 81
  - REFUTES / SUPPORTS : 26
  - SUPPORTS / NOT_ENOUGH_INFO : 51
  - SUPPORTS / REFUTES : 16
  - SUPPORTS / SUPPORTS : 116
- part d'accord : 0.6355
- kappa : 0.4700
- part d'accord (répond au claim ou non) : 0.7336
- kappa (répond au claim ou non) : 0.4190

## Juge par étapes — causes

absent

## Étiquettes d'error analysis v1

| fichier | étiquette | claims | local (attendu / non attendu) | claude (attendu / non attendu) |
|---|---|---|---|---|
| v1-annotations.json |  | 4 | 2 / 0 | 0 / 0 |
| v1-annotations.json | C | 4 | 2 / 0 | 1 / 0 |
| v1-annotations.json | C (BM25-partiellement-récupérable) | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | C (BM25-récupérable) | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | C (T secondaire) | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | C mixte | 1 | 1 / 0 | 0 / 0 |
| v1-annotations.json | C ou Q (difficile a définir) | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | C sémantique léger | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | C-lexical | 8 | 7 / 2 | 6 / 1 |
| v1-annotations.json | C-lexical faible | 1 | 1 / 0 | 1 / 0 |
| v1-annotations.json | C-lexical modéré | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | C-semantique (C-lexical faible) | 1 | 1 / 0 | 1 / 0 |
| v1-annotations.json | C-sémantique | 3 | 2 / 1 | 1 / 0 |
| v1-annotations.json | C-sémantique relationnel | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | Complet | 1 | 1 / 0 | 0 / 0 |
| v1-annotations.json | DOUBLON | 1 | 1 / 0 | 1 / 0 |
| v1-annotations.json | DOUBLON ? | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | N (multi-gold, couverture top-3 complète) + Q-saturation (rang-1 pertinent non annoté) | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | N (near-cosmetic multi-gold) + Q-saturation sous-jacent | 1 | 1 / 0 | 1 / 0 |
| v1-annotations.json | N + Q-saturation | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | Q | 6 | 2 / 0 | 0 / 0 |
| v1-annotations.json | Q C-semantique | 1 | 1 / 1 | 1 / 0 |
| v1-annotations.json | Q de saturation | 1 | 1 / 1 | 1 / 1 |
| v1-annotations.json | Q-inference | 7 | 5 / 0 | 0 / 0 |
| v1-annotations.json | Q-inference. | 1 | 1 / 0 | 0 / 0 |
| v1-annotations.json | Q-inférence | 4 | 1 / 0 | 1 / 0 |
| v1-annotations.json | Q-inférence leger (il reste le meilleur doc mais ne repond pas au claim) | 1 | 0 / 0 | 0 / 0 |
| v1-annotations.json | Q-saturation | 5 | 4 / 1 | 3 / 1 |
| v1-annotations.json | Q-saturation. | 1 | 1 / 0 | 0 / 0 |
| v1-annotations.json | T | 2 | 2 / 1 | 1 / 1 |
| v1-annotations.json | T ( recuperation lexical manqué). | 3 | 0 / 0 | 0 / 0 |
| v1-annotations.json | T + C-lexical | 1 | 1 / 0 | 1 / 0 |
| v1-deep-miss-annotations.json | C-lexical | 3 | 3 / 2 | 3 / 2 |
| v1-deep-miss-annotations.json | C-sémantique | 6 | 5 / 2 | 3 / 2 |
| v1-deep-miss-annotations.json | Q-inférence | 10 | 6 / 0 | 0 / 0 |
| v1-deep-miss-annotations.json | Q-inférence (gold-faible) | 10 | 3 / 0 | 0 / 0 |
| v1-deep-miss-annotations.json | Q-saturation | 2 | 2 / 2 | 1 / 0 |
| v1-deep-miss-annotations.json | T | 2 | 2 / 0 | 1 / 0 |
| v1-deep-miss-annotations.json | combo | 7 | 7 / 2 | 3 / 1 |
