#!/usr/bin/env bash
# Barre de fin de ticket — les critères de .claude/rules/quality-gate.md, dans leur ordre.
# Rend 0 si tout est vert ou vide, 1 dès qu'un critère est rouge. Aucun jugement.
# Reprise de job-search (TCK-235) le 1er octobre 2026, portage de la boucle sur
# rag-eval-scifact : sans critère front, avec un critère d'imports interdits.
#
# Usage : bash scripts/barre.sh            (depuis la racine du worktree)
#         BARRE_BASE=<ref> bash scripts/barre.sh   (point de comparaison imposé)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
RUFF="$ROOT/.venv/bin/ruff"
[ -x "$PY" ] && [ -x "$RUFF" ] || {
  echo "BARRE: .venv absent ou incomplet à $ROOT — lance d'abord scripts/preflight.sh."
  exit 1
}

# Point de départ de la branche : sa première entrée de reflog, c'est-à-dire l'état
# à sa création. Ce dépôt n'a pas d'origin/HEAD fiable pour ça.
branche="$(git rev-parse --abbrev-ref HEAD)"
BASE="${BARRE_BASE:-$(git reflog show --format=%H "$branche" 2>/dev/null | tail -1)}"
[ -n "$BASE" ] || { echo "BARRE: point de départ de $branche introuvable."; exit 1; }

# Fichiers du ticket : suivis et modifiés depuis BASE (commités ou non), plus les nouveaux.
changes="$( { git diff --name-only --diff-filter=ACMR "$BASE"; git ls-files --others --exclude-standard; } | sort -u)"
py="$(printf '%s\n' "$changes" | grep -E '\.py$' | while read -r f; do [ -f "$f" ] && echo "$f"; done)"

rouge=0
ligne() { printf '%-3s %-6s %s\n' "$1" "$2" "$3"; [ "$2" = rouge ] && rouge=1; return 0; }

echo "BARRE — base $BASE, branche $branche"

# 1. ruff check sur les .py du ticket
if [ -z "$py" ]; then ligne 1 vide "ruff check — aucun .py touché"
elif "$RUFF" check --no-cache $py; then ligne 1 vert "ruff check"
else ligne 1 rouge "ruff check"; fi

# 2. ruff format --check sur les .py du ticket
if [ -z "$py" ]; then ligne 2 vide "ruff format — aucun .py touché"
elif "$RUFF" format --no-cache --check $py; then ligne 2 vert "ruff format --check"
else ligne 2 rouge "ruff format --check"; fi

# 3. Suite de tests entière
if "$PY" -m pytest -q; then ligne 3 vert "python -m pytest"
else ligne 3 rouge "python -m pytest"; fi

# 4. Aucun TODO, FIXME ou XXX introduit
ajouts="$( { git diff "$BASE" -- . ':!scripts/barre.sh' | grep -E '^\+' | grep -vE '^\+\+\+';
            git ls-files --others --exclude-standard | while read -r f; do [ -f "$f" ] && sed 's/^/+/' "$f"; done; } )"
if printf '%s\n' "$ajouts" | grep -qE '\b(TODO|FIXME|XXX)\b'; then
  printf '%s\n' "$ajouts" | grep -nE '\b(TODO|FIXME|XXX)\b' | head -5
  ligne 4 rouge "TODO / FIXME / XXX introduit"
else ligne 4 vert "aucun TODO / FIXME / XXX introduit"; fi

# 5. Aucun import interdit par .claude/rules/methodologie.md dans les .py du ticket :
#    libs d'éval (les métriques, la fusion et les tests stats se codent à la main)
#    et API LLM payantes (Ollama uniquement).
interdits='^[[:space:]]*(import|from)[[:space:]]+(pytrec_eval|beir|ranx|ir_measures|trectools|sentence_transformers\.evaluation|openai|anthropic|google\.generativeai|google\.genai|cohere|mistralai)\b'
if [ -z "$py" ]; then ligne 5 vide "imports interdits — aucun .py touché"
elif grep -nE "$interdits" $py; then ligne 5 rouge "import interdit (lib d'éval ou API payante)"
else ligne 5 vert "aucun import interdit"; fi

# 6. Reconstructibilité : si le ticket touche les dépendances, un venv neuf les installe
if printf '%s\n' "$changes" | grep -qE '^(requirements[^/]*\.txt|pyproject\.toml)$'; then
  TMP="$(mktemp -d)"
  if "$PY" -m venv "$TMP/venv" \
     && "$TMP/venv/bin/python" -m pip install -q -r requirements.txt -r requirements-dev.txt; then
    ligne 6 vert "reconstructibilité — venv neuf installé"
  else ligne 6 rouge "reconstructibilité — venv neuf en échec"; fi
  rm -rf "$TMP"
else ligne 6 vide "dépendances — non touchées"; fi

if [ "$rouge" -eq 0 ]; then echo "BARRE: verte"; else echo "BARRE: rouge"; fi
exit "$rouge"
