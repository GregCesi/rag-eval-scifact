# Quality gate

Barre générique, identique pour tous les tickets. Elle dit _est-ce bien fait_.
Elle ne dit jamais _est-ce ce qui était demandé_ — c'est le rôle des critères
d'acceptation du ticket. Une quality gate sans eux donne du code impeccable
qui fait la mauvaise chose.

**Elle se passe en une commande, depuis la racine du worktree :**

```bash
bash scripts/barre.sh
```

Le script rend 0 ou 1 et imprime une ligne par critère, `vert` / `rouge` / `vide`.
Un ticket n'est fini que s'il rend 0. Tu ne rejoues pas les critères à la main pour
conclure : c'est le rendu du script qui fait foi, et c'est lui que ton verdict recopie.

1. **`ruff check`** sur les `.py` créés ou modifiés par le ticket — zéro erreur.
2. **`ruff format --check`** sur les mêmes fichiers — aucun fichier à reformater.
   Un fichier existant que tu touches passe entier sous ces deux critères, y compris
   ses lignes anciennes : le code existant n'est pas encore propre, et chaque ticket
   nettoie ce qu'il touche. Ce nettoyage n'est pas _hors demande_ ; déclare-le dans
   le récit, pas dans la section 3 du verdict.
3. **`python -m pytest`** — la suite entière, verte.
4. Aucun `TODO`, `FIXME` ou `XXX` introduit par le ticket.
5. **Aucun import interdit** dans les `.py` du ticket : ni lib d'évaluation (`pytrec_eval`,
   `beir`, `ranx`, `ir_measures`, `trectools`, `sentence_transformers.evaluation`), ni API
   LLM payante (`openai`, `anthropic`, `google.generativeai`, `google.genai`, `cohere`,
   `mistralai`). C'est la traduction machine de `methodologie.md`. Sinon `vide`.
6. **Reconstructibilité** — si le ticket touche `requirements.txt`,
   `requirements-dev.txt` ou `pyproject.toml` : un venv neuf les installe. Sinon `vide`.

Le périmètre des critères 1 et 2 se constate sur les fichiers modifiés depuis la
création de ta branche, commités ou non, plus les fichiers nouveaux. Le script le
calcule ; tu n'as pas à le faire.

La configuration de ruff et de pytest vit dans `pyproject.toml`, leurs versions dans
`requirements-dev.txt`. Tu n'écris dans aucun des deux : ils sont la barre, et
l'agent ne définit jamais sa barre.

**Red-green.** Un test qui n'a jamais été rouge ne prouve rien. Tout test que tu
écris pour un critère d'acceptation, tu le lances d'abord contre le code d'avant ton
changement, et il doit y être rouge. Les tests s'écrivent depuis le ticket, pas
depuis le code.

Chaque critère rend 0 ou 1 sans œil humain. Un critère qui demande un jugement n'est
pas encore un critère : c'est une intuition, et elle ramène un humain dans la boucle
à chaque tour. Elle ne s'ajoute pas ici. L'évaluation d'ECC (`gan-evaluator`, notes
sur 10) n'est pas un critère de cette barre, pour la même raison.

La propreté n'est pas un bonus : ce dépôt est aussi un livrable de positionnement.
Elle est un critère de fin.

_Repris de job-search (TCK-235) le 1er octobre 2026, portage de la boucle sur rag-eval-scifact :
le critère front est remplacé par le critère d'imports interdits._
