#!/bin/bash
# Codex review gate — owner instruction 2026-09-15: "use codex gpt 6 astra
# with ultra effort to review and help execute every step."
# Runs codex-cli 0.154.0 (user-prefix install on ml-beast) non-interactively,
# read-only sandbox, against this repo's mount path; the review lands in
# docs/codex/<slug>.md (committed) so the review loop is auditable.
#
# Usage (ON ml-beast):  codex_gate.sh <slug> <prompt-file>
# The caller (muhit-pc) invokes it detached over ssh and polls for the
# DONE marker file.
set -u
SLUG="${1:?slug}"
PROMPT_FILE="${2:?prompt file}"
REPO="/mnt/Muhit/Home/ms/axial-layout-scoping"
OUT="$REPO/docs/codex/${SLUG}.md"
MARK="$HOME/axial/codex_${SLUG}.done"
mkdir -p "$REPO/docs/codex" "$HOME/axial"
rm -f "$MARK"

export PATH="$HOME/.local/bin:$PATH"   # codex 0.154 (ultra-capable), not the 0.58 in /usr/local
{
  echo "# Codex review — ${SLUG}"
  echo
  echo "model: gpt-6-astra · effort: ultra · sandbox: read-only · $(date -Is)"
  echo
} > "$OUT"

codex exec \
    --skip-git-repo-check \
    -C "$REPO" \
    -s read-only \
    -m gpt-6-astra \
    -c model_reasoning_effort=ultra \
    -o /tmp/codex_last_${SLUG}.txt \
    - < "$PROMPT_FILE" >> "$HOME/axial/codex_${SLUG}.log" 2>&1
RC=$?
{
  cat "/tmp/codex_last_${SLUG}.txt" 2>/dev/null
  echo
  echo "---"
  echo "exit: $RC"
} >> "$OUT"
echo "$RC" > "$MARK"
