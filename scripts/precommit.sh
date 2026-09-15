#!/bin/bash
# Pre-commit integrity gate. Exists because of two real incidents (both editor
# paste-overs into open buffers, discovered 2026-09-01): analysis/traces.sh
# (self-paste at :131, mtime Aug 31 — bash -n failed) and setup.sh (terminal
# session pasted at :57 — shipped broken to the remote in commit 10452d1).
#
# Checks STAGED CONTENT (git show :path), not the worktree — what gets
# committed is what gets checked. Raise-don't-warn: any hit blocks the commit
# and names the offending file and line.
#   1. every staged *.sh must parse        (bash -n)
#   2. every staged *.py must parse        (python ast)
#   3. NO staged file may contain terminal-paste artifacts (shell prompts) —
#      the corruption class is format-agnostic, so ALL files are scanned.
#      Sole exemption: docs/audit_log.md, which legitimately quotes incident
#      material by design (see its header).
# COPIED from sm120-nulltest@a997e27 scripts/precommit.sh (handoff rule 4:
# reuse tooling with headers intact; cite the source). Unmodified.
set -u
fail=0
say() { echo "precommit: $*" >&2; }

STAGED=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$STAGED" ] && exit 0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while IFS= read -r f; do
    blob="$TMP/staged_blob"
    git show ":$f" > "$blob" 2>/dev/null || continue

    case "$f" in
        *.sh)
            if ! err=$(bash -n "$blob" 2>&1); then
                say "SYNTAX FAIL (bash -n) in staged $f:"
                echo "$err" | sed "s|$blob|$f|g" >&2
                fail=1
            fi ;;
        *.py)
            if ! err=$(python3 -c 'import ast, sys
ast.parse(open(sys.argv[1]).read(), filename=sys.argv[2])' "$blob" "$f" 2>&1); then
                say "SYNTAX FAIL (ast.parse) in staged $f:"
                echo "$err" >&2
                fail=1
            fi ;;
    esac

    case "$f" in docs/audit_log.md) continue ;; esac
    # pattern assembled from split literals so this file never matches itself
    ART='muhit@muhit''-pc|\(base\) [a-z_]+@|[a-z_][a-z0-9_]*@[a-zA-Z0-9-]+:[^ ]*\$ '
    if hits=$(grep -nI -E "$ART" "$blob"); then
        say "PASTE ARTIFACT (terminal-prompt material) in staged $f:"
        echo "$hits" | sed "s|^|  $f:|" >&2
        fail=1
    fi
done <<< "$STAGED"

if [ "$fail" -ne 0 ]; then
    say "COMMIT BLOCKED — fix the staged content and retry."
    exit 1
fi
exit 0
