#!/usr/bin/env bash
# The recipe's checks that need no Spark: every script parses, shellcheck when it is installed, and the tests here.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
for f in start.sh start-tp6.sh start-tp6-main.sh stop-tp6.sh scripts/*.sh scripts/exlr8/*.sh scripts/tests/*.sh; do
  bash -n "$f"
done
echo 'syntax: pass'
if command -v shellcheck >/dev/null; then
  shellcheck -x start.sh start-tp6.sh start-tp6-main.sh stop-tp6.sh scripts/*.sh scripts/tests/*.sh && echo 'shellcheck: pass'
else
  echo 'shellcheck: not installed, skipped'
fi
for f in scripts/*.py scripts/exlr8/*.py scripts/tests/*.py; do
  python3 -B -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$f"
done
echo 'python: pass'
for t in scripts/tests/*.sh; do
  [[ "$t" != scripts/tests/run.sh ]] || continue
  bash "$t"
done
