#!/usr/bin/env bash
# Stop hook: when src/ or tests/ have uncommitted changes, run the test suite,
# refresh test-results/junit.xml, and (on failure) block the stop so Claude fixes it.
# Keeps the eval-traceability discipline (CLAUDE.md §5) green automatically.
set -u

# Read the hook payload; bail out of the re-entrant stop to avoid loops.
input=$(cat 2>/dev/null || true)
case "$input" in *'"stop_hook_active":true'*) exit 0 ;; esac

DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$DIR" 2>/dev/null || exit 0

# Only act when source or tests changed (staged, unstaged, or untracked).
[ -n "$(git status --porcelain -- src tests 2>/dev/null)" ] || exit 0

# Find an interpreter that has pytest (local test venv first, then fallbacks).
PY=""
for c in .venv-test/bin/python .venv/bin/python "$(command -v python3 2>/dev/null)"; do
  [ -n "$c" ] && [ -x "$c" ] && "$c" -c "import pytest" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || exit 0   # no pytest available → skip silently

mkdir -p test-results
log=$(mktemp)
if "$PY" -m pytest tests/ --junitxml=test-results/junit.xml -q >"$log" 2>&1; then
  rm -f "$log"
  exit 0
fi

echo "❌ 테스트 실패: src/tests 변경 후 스위트가 깨졌습니다. 완료 전에 수정하세요 (CLAUDE.md §5.1)." >&2
grep -E "FAILED|ERROR|assert" "$log" | head -20 >&2
rm -f "$log"
exit 2
