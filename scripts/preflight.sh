#!/usr/bin/env bash
# Pré-vol d'aptitude de l'environnement d'exécution — rag-eval-scifact.
# Repris de job-search (TCK-235) le 1er octobre 2026, portage de la boucle
# agent-codeloop. Une seule différence : les données.
# Répond à « mon environnement est-il apte », jamais à « mon travail est-il bon ».
# Lancé par jet.sh dans le worktree avant l'agent, et à la racine après chaque merge.
# Codes lus par jet.sh : 0 apte, 2 requirements.txt absent (rien à vérifier),
# tout autre code = inapte.
#
# Un worktree est une copie fraîche : .venv/ et data/scifact/ sont ignorés par git,
# donc absents. Ce script pose le venv, y installe exactement requirements.txt et
# requirements-dev.txt, relie data/scifact au jeu de données du dépôt principal,
# puis vérifie que la suite de tests passe sur l'état d'entrée.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f "$ROOT/requirements.txt" ] || {
  echo "PRÉ-VOL: requirements.txt absent à $ROOT — environnement non initialisé." >&2
  exit 2
}

# --- Données : BEIR SciFact, téléchargées une fois dans le dépôt principal --------
# Un lien, pas une copie : les fichiers ne sont jamais écrits par le pipeline.
# Le dépôt principal est le parent du répertoire git commun à tous les worktrees.
PRINCIPAL="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
SOURCE="$PRINCIPAL/data/scifact"
for f in corpus.jsonl queries.jsonl qrels/test.tsv; do
  [ -f "$SOURCE/$f" ] || {
    echo "PRÉ-VOL: $SOURCE/$f absent — télécharge SciFact dans le dépôt principal (README, Quickstart étape 2)." >&2
    exit 1
  }
done
if [ "$ROOT" != "$PRINCIPAL" ] && [ ! -e "$ROOT/data/scifact" ]; then
  mkdir -p "$ROOT/data"
  ln -s "$SOURCE" "$ROOT/data/scifact"
  echo "PRÉ-VOL: data/scifact relié à $SOURCE"
fi

VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "PRÉ-VOL: .venv absent, création."
  if command -v uv >/dev/null 2>&1; then
    uv venv --quiet --python 3.14 "$VENV" || uv venv --quiet "$VENV"
  else
    PYBIN=""
    for c in python3.14 python3.13 python3.12 python3.11 python3; do
      command -v "$c" >/dev/null 2>&1 && { PYBIN="$c"; break; }
    done
    [ -n "$PYBIN" ] || { echo "PRÉ-VOL: aucun python3 trouvé." >&2; exit 1; }
    "$PYBIN" -m venv "$VENV"
  fi
fi

echo "PRÉ-VOL: installation de requirements.txt et requirements-dev.txt"
if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet --python "$VENV/bin/python" -r requirements.txt -r requirements-dev.txt
else
  "$VENV/bin/python" -m pip install --quiet -r requirements.txt -r requirements-dev.txt
fi

"$VENV/bin/ruff" --version

echo "PRÉ-VOL: suite de tests sur l'état d'entrée"
PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -m pytest -q

echo "PRÉ-VOL: apte."
