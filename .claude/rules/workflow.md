# Régime d'exécution

Un ticket, un worktree. Séquentiel : jamais deux tickets en vol.

Ce fichier décrit **comment tu te conduis**. Ce que tu dois produire est dans le
ticket. Ce qui fait qu'un travail est bien fait est dans `quality-gate.md` et
dans les règles du projet — `methodologie.md`, `invariants.md` et `versioning.md`. Le premier porte sur le rendu des
commandes, les autres sur la méthode d'évaluation et la forme des runs. Tous sont nommés dans `CLAUDE.md`.

Ce fichier est repris de la boucle agent-codeloop telle qu'elle tourne sur
job-search (TCK-235) ; seules la liste des règles du projet et la section Git (les
données, plus de front) diffèrent. Portage du 1er octobre 2026.

---

## Ordre de démarrage

Dans cet ordre, avant toute autre chose.

**Avant tout lancement, contrôle humain hors session** — `git worktree list`
depuis le dépôt principal. Un worktree présent signale une exécution en vol ou
morte en vol. Une session isolée ne peut rien lire hors de sa racine : ce contrôle
n'est pas le tien.

**Étape zéro — il n'y en a plus pour toi.** L'aptitude de l'environnement est
contrôlée par le lanceur, dans ton worktree, avant qu'il ne te démarre : si tu
tournes, c'est qu'elle est acquise. Tu ne lances pas `scripts/preflight.sh`, tu
n'as aucun code de sortie à interpréter, et tu ne décides d'aucun arrêt sur ce
motif. Retiré le 7 septembre 2026, TCK-157 — au jet 14, EXE-12 a rendu la main
sans rien écrire en attendant une notification de fin de pré-vol qui n'existe
pas. Aucune résistance rencontrée, donc aucun des six points d'arrêt ne couvrait
le cas.

1. **Lis le ticket.** Il ne se cherche pas. La file d'exécution est une vue
   Notion à adresse fixe, filtrée sur `Statut = Validé` ou `En exécution`, triée par `Ref` croissant :
   https://app.notion.com/9fc073006cc7472bb8397f8695fcab66?v=3cd268d8af3481979b8f000cce1692cc
   Lis-la par `query_data_sources` en **mode `view`**, jamais en SQL : le mode SQL
   est plafonné par un quota partagé avec le chat, et son épuisement en cours de jet
   arrête le tour sans recours — constaté le 31 août 2026, EXE-10. La vue rend
   `Titre`, `Problème`, `Critères d'acceptation`, `Hypothèses déclarées`, `Régime`
   et l'URL de chaque fiche.
   Le ticket `EXE-n` est celui dont le champ `Ref` vaut `n`. S'il est absent de la
   file : arrêt, porte de sortie (objet nommé absent). Absent veut dire que son
   `Statut` vaut autre chose que `Validé` ou `En exécution`. Un ticket `Cadré`
   ne franchit pas la porte amont : tu ne le lances pas.
   Ta fiche est en `En exécution` : le lanceur la passe dans cet état avant de te
   démarrer, et il vérifie son écriture. Pas une exécution concurrente — la tienne.
   Tu ne vérifies rien sur ce `Statut`.

2. **Point d'arrêt 1** — si ce que le ticket demande est ambigu, pose tes
   questions maintenant, en un seul message, et attends. Rien n'est écrit dans
   Notion à ce stade.

3. **Le venv** — voir règle 2 de la section Git.

**Le `Statut` de la fiche ne t'appartient qu'au verdict.** Le lanceur l'a passée à
`En exécution` avant de te démarrer, et tu ne l'écris qu'une fois, à la fin, pour
poser `Fait` ou `Arrêté`. Tu ne poses aucun marqueur sur le disque : la présence du
worktree est la marque, et elle ne dépend pas de toi.

---

## Git

1. Le worktree part de `origin/HEAD`. Jamais du `HEAD` local.
2. **Le venv est déjà posé.** Le lanceur a créé `.venv/` dans ton worktree et y
   a installé exactement `requirements.txt` et `requirements-dev.txt` avant de te
   démarrer. Toute commande Python passe par `.venv/bin/python` ; `ruff` et
   `pytest` par `.venv/bin/`. N'utilise ni le `python3` du système ni un autre venv.
   **Tu n'installes rien d'autre**, sauf dans le cas unique où le ticket demande
   une dépendance : tu l'ajoutes alors à `requirements.txt`, puis
   `.venv/bin/python -m pip install -r requirements.txt`. Jamais de
   `pip install <paquet>` nu, qui installe sans rien écrire.
   Ton entrée Journal le déclare : la dépendance, le critère du ticket qui la
   demande, et le fait que `requirements.txt` a changé. Une installation non
   déclarée est un écart — c'est le seul moyen qu'a la relecture de la voir.
   **Les données.** `data/scifact/` est un lien posé par le lanceur vers le jeu
   SciFact du dépôt principal : tu le lis, tu n'y écris jamais. `chroma_data/` est
   absent d'un worktree neuf : si ton ticket a besoin de l'index, tu le construis
   par `.venv/bin/python -m rag_eval_scifact.ingest` (quelques minutes), et tu ne le
   commites pas — il est ignoré par git.
3. La branche n'est jamais poussée.
4. Réintégration par merge local dans `main`, après relecture du diff par un
   humain. Ce n'est pas toi qui merges.
5. **La boucle ne pousse jamais `main`.** Depuis TCK-152 (2 septembre 2026),
   chaque tour est mergé dans `integration/jet-<n>`, qui est la seule branche
   poussée. `main` avance par une commande de promotion, lancée par un humain
   après relecture des écarts. Et le worktree du tour suivant part de cette
   branche, plus de `origin/HEAD`.

---

## Les six points d'arrêt

La porte de sortie s'évalue **avant** les compteurs, à chaque tour.

| #   | Déclencheur                                                                                                                 | Évalué quand                                 | Conduite                                                                | Écrit                                                                           |
| --- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | **Ambiguïté sur ce qui est demandé** (le _quoi_), à la lecture du ticket                                                    | Avant tout travail                           | Tour de questions **groupé** — un seul message, ou aucun — puis attente | Rien dans Notion                                                                |
| 2   | **Porte de sortie** — objet nommé absent, objet pas de la nature supposée, prémisse explicite fausse                        | À tout moment                                | Arrêt                                                                   | `Arrêté`. Journal : le cas heurté, l'objet, ce qui était attendu                |
| 3   | **Test passé au vert, jugé faux ensuite**                                                                                   | À tout moment après un vert                  | Arrêt. Le test **n'est pas modifié**                                    | `Arrêté`. Journal : cas _prémisse fausse_                                       |
| 4   | **Stagnation** — 2 tours consécutifs, même critère de gate, même signature d'erreur, après une modification censée corriger | Après chaque passage de gate, à partir du 2ᵉ | Arrêt                                                                   | `Arrêté`. Journal : critère, signature, tentatives menées, hypothèse de blocage |
| 5   | **Plafond** — 12 tours de gate sur la même tâche                                                                            | Après chaque passage de gate                 | Arrêt                                                                   | `Arrêté`. Journal : motif _plafond_                                             |
| 6   | **Coupure** — quota ou fenêtre de contexte                                                                                  | Subi, non évalué                             | Aucune, la session meurt                                                | Rien. La fiche reste `En exécution` et le worktree reste présent — c'est à ces deux présences que la coupure se constate                 |

### Règle de priorité

**Le fait constatable gagne sur le compteur.** Une situation qui heurte à la fois
une porte de sortie (2, 3) et un seuil de gate (4, 5) s'arrête sur la porte, avec
son motif. Un compteur dit _ça n'avance pas_ ; un fait dit _pourquoi_.

---

## Ce qui n'est pas un arrêt

| Situation                                         | Conduite                                                                                 | Écrit                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| Ambiguïté en route, aucun cas de fait constatable | Retenir **l'interprétation qui produit le moins de ce qui est demandé**, puis poursuivre | Journal : `lu X comme Y`                     |
| Gate rouge, progrès constaté                      | Nouvelle boucle                                                                          | Rien                                         |
| Première gate verte                               | Commit dans le worktree                                                                  | Rien                                         |
| Tâche terminée                                    | Confrontation aux critères d'acceptation, un par un                                      | `Fait`. Journal : verdict |

### L'interprétation étroite

_Produit le moins de ce qui est demandé_ — pas _la plus raisonnable_, pas _celle
qui touche le moins de code_. Le critère se constate sur ce que la barre demande,
sans regarder le coût d'implémentation.

L'asymétrie est voulue : une interprétation fausse doit laisser un **manque**,
jamais un surplus. La relecture d'écart n'a alors que du travail à ajouter,
jamais à défaire.

**Toute interprétation retenue s'écrit**, dans le Journal, sous la forme
littérale `lu X comme Y`. X est la formulation du ticket, citée. Y est ce que tu
as fait.

---

## Invariants

Vrais aux six points d'arrêt, sans exception.

- La branche de travail ne se pousse pas sous son nom ; quand un tour casse, le lanceur la pousse sous `arret/jet-<n>/<TOUR>`, jamais mergée (TCK-123). Seule `integration/jet-<n>` est poussée en régime nominal, et `main` ne bouge pas avant la promotion (point 5 ci-dessus).
- Le worktree est laissé en l'état. Exception : à la coupure il n'est pas laissé,
  il est **abandonné** — aucune reprise automatique d'une exécution coupée.
- Tu ne crées aucun ticket et n'écris aucune proposition de suite.
- Sur le ticket d'exécution, `Statut` en écriture **au verdict final seulement** ; tout le reste en
  lecture seule — titre, problème, critères d'acceptation, hypothèses déclarées,
  **et la relation `Journal`, qui ne se lit ni ne s'écrit** — elle est renseignée
  par le lanceur avant ton démarrage (voir *Où écrire ton entrée*).
  Tu n'écris jamais la barre du ticket que tu exécutes.
- Une exécution morte en vol ne se constate pas depuis ta session : elle se voit
  hors session, à la présence du worktree et à la fiche restée `En exécution`. Tu
  n'as rien à poser ni à nettoyer pour ça.

---

## Écriture

Tu écris des faits : entrée de Journal, verdict, constats. Jamais des intentions,
jamais un arbitrage.

Toute entrée de Journal que tu écris ouvre sur un **bloc verdict à cinq
sections, dans cet ordre, avant le récit**.

### 1. Barre

Une ligne. Ce qui avait été demandé, et où c'est écrit : `EXE-n`, ou
`IMPLEMENTATION.md` étape N. Sans ce pointeur, les trois sections suivantes ne
confrontent rien et produisent un second récit.

La barre est **citée** ici, jamais définie : elle vit dans le ticket, en lecture
seule. Les règles permanentes — `quality-gate.md` et les règles d'architecture —
n'y entrent pas ; elles ont leur propre section.

### 2. Confrontation

Une ligne par item de la barre, dans l'ordre de la barre, préfixée
`tenu` / `non tenu` / `non abordé`.

L'item est **cité tel qu'il est écrit**, jamais reformulé. Une reformulation
déplace la cible. Un item impossible à citer signale que la barre était mal
écrite — c'est une information, pas une gêne.

Une ligne `tenu` **nomme le fait qui le prouve** : commande passée, fichier
produit, test vert, commit. Sans fait, c'est un avis, et ton avis sur ton propre
travail ne vaut rien.

Un `non tenu` dit ce qui manque, en termes observables. Pas pourquoi — le pourquoi
est du récit.

### 3. Hors demande

Ce qui a été fait que la barre ne demandait pas. Section obligatoire : `aucun`
s'écrit. Une section absente est ambiguë entre _rien à signaler_ et _pas regardé_.

### 4. Hypothèses tombées

Celles déclarées au cadrage que l'exécution a invalidées — champ
`Hypothèses déclarées` du ticket. En implémentation : celles que tu as dû prendre
faute d'information. `aucune` s'écrit.

### 5. Règles permanentes

Confrontation à `quality-gate.md`, puis aux règles d'architecture. Section
obligatoire. Contrairement à la barre, ces documents ne varient pas d'un ticket
à l'autre : c'est pourquoi ils ne sont pas cités dans les critères d'acceptation,
et pourquoi rien ne te dispense de les confronter quand le ticket les ignore.

`quality-gate.md` — une ligne par critère, `vert` / `rouge` / `vide`, recopiée
de la sortie de `bash scripts/barre.sh` au dernier passage.

Règles du projet (`methodologie.md`, `invariants.md` et `versioning.md`) — une ligne par règle applicable, préfixée
`tenu` / `non tenu`, règle citée telle qu'elle est écrite, avec son fichier et sa
section. Les règles non applicables sont **groupées par motif**, une ligne par
motif, références listées : `non applicable — scoring.md, sources.md : aucun
fichier sous orchestrator/ touché`. Le motif
se constate sur le diff, sans jugement ; un motif qui demande d'apprécier signale
que la règle s'applique et doit avoir sa ligne.

Cette section dit _est-ce bien fait_. La section 2 dit _est-ce ce qui était
demandé_. Les confondre produit un verdict où le second se perd dans le premier.

### Ordre

Le verdict précède le récit. Un récit écrit en premier fixe une version des faits
que le verdict n'a plus qu'à ratifier. `Résumé` reste à deux lignes.

### Où écrire ton entrée

**Ton entrée existe déjà.** Le lanceur l'a créée dans la base Journal et
rattachée à ta fiche avant de te démarrer ; son ID est dans ton prompt. Elle
porte déjà `Auteur`, `Type`, `Date`, `Projet`, `Tickets`, et un titre provisoire.

Ordre fixe, chaque étape close avant la suivante.

1. **Écris dans l'entrée dont l'ID t'a été donné** : remplace le titre
   provisoire, écris `Résumé` (deux lignes), et pose le corps — verdict à cinq
   sections, puis récit.
2. **Statut de la fiche** — une fois l'entrée écrite, et seulement après, écris
   le statut final sur `EXE-n`, `Fait` ou `Arrêté`.

*Inversé le 7 septembre 2026, TCK-158.* L'ordre précédent posait le statut
d'abord, au motif que c'est ce qui rend la session lisible même si tout ce qui
suit échoue. Ce compromis datait d'avant la branche d'intégration : une coupure
entre les deux écritures laissait `Fait` sur un tour sans récit, et le lanceur
mergeait. Constaté au jet 14, EXE-10 — `API Error: 529 Overloaded`, six fichiers
mergés, aucun récit. Dans ce sens-ci, une coupure laisse la fiche `En exécution`,
c'est-à-dire le point d'arrêt 6, que le lanceur détecte et libère déjà.

**Tu ne crées aucune page.** Ni pour cette entrée, ni pour autre chose. La
permission t'en est retirée : `notion-create-pages` est en `deny`. Une session
produit exactement une entrée, et c'est celle qu'on t'a donnée.

**Tu n'écris ni `Projet` ni `Tickets`.** Ta fiche ne porte aucune source pour
`Projet` — la base Exécution n'a pas ce champ — et les chercher est ce qui t'a
fait lire la base Projets et les entrées des jets précédents, dont les récits
décrivent un terrain qui n'est pas le tien. Constaté aux jets 12, 13 et 14. Le
lanceur les a renseignés, il les tient d'une source que tu n'as pas.

**Ne touche pas la relation `Journal` de la fiche `EXE-n`.** Ne la lis pas, ne
l'écris pas, ne la relis pas. Le rattachement est fait, et il l'était avant que
tu démarres.

*Retiré le 7 septembre 2026, TCK-157. Ce qui a disparu d'ici : le choix du
parent, le refetch de contrôle, la création de la page, et l'écriture de `Projet`
et `Tickets`.*
