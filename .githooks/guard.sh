#!/usr/bin/env bash
# Shared checks for the git hooks and CI.
#
#   guard.sh message <file>        commit message: no AI attribution
#   guard.sh identity "<who>"      author/committer: not an AI or bot
#   guard.sh staged                staged files: no secrets, no .env
#   guard.sh commits <rev-range>   every commit in the range (pre-push, CI)
#
# Exits 1 with a reason if anything is blocked.

set -u

# Names of AI tools and vendors, as they show up in trailers and identities.
AI='claude|anthropic|openai|chatgpt|gpt-?[0-9]|codex|copilot|cursor|gemini|aider|devin|windsurf|codeium|tabnine|llm|\[bot\]'
# Trailers like "Co-Authored-By:", "Signed-off-by:", "Generated-by:", "Assisted-by:".
TRAILER='^[[:space:]]*[a-z-]*(by|with)[[:space:]]*:'
# Footers that tools add to messages and PRs.
FOOTER="(generated|written|made|created) (with|by|using) (an? )?(ai|$AI)|"'claude\.com/claude-code|claude\.ai/code|noreply@anthropic\.com|🤖'
# API keys in the formats common providers use (including Modal, GitHub and Hugging Face).
SECRET='sk-or-v1-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|sk-(proj-)?[A-Za-z0-9]{32,}|\b(ak|as)-[A-Za-z0-9]{20,}\b|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}'

fail() { echo "blocked: $*" >&2; exit 1; }

check_message() {
  # Comment lines are stripped by git before the commit is made.
  local msg
  msg=$(grep -v '^#' "$1")
  local bad
  bad=$(printf '%s\n' "$msg" | grep -Ei "$TRAILER" | grep -Ei "$AI")
  [ -n "$bad" ] && fail "AI attribution trailer in commit message:
  $bad
See AGENTS.md."
  bad=$(printf '%s\n' "$msg" | grep -Ei "$FOOTER")
  [ -n "$bad" ] && fail "AI footer in commit message:
  $bad
See AGENTS.md."
  return 0
}

check_identity() {
  printf '%s\n' "$1" | grep -Eiq "$AI|@(anthropic|openai)\.com" \
    && fail "commit identity looks like an AI or bot: $1
Commit as yourself (git config user.name / user.email)."
  return 0
}

check_staged() {
  local f
  for f in $(git diff --cached --name-only --diff-filter=ACMR); do
    case "$f" in
      .env.example) ;;
      .env|*.env|.env.*) fail "$f is a secrets file; keep it out of git" ;;
    esac
  done
  # Only added lines, and skip this file (it contains the patterns).
  local hits
  hits=$(git diff --cached -U0 --diff-filter=ACMR -- . ':(exclude).githooks/guard.sh' \
    | grep -E '^\+' | grep -Eo "$SECRET")
  [ -n "$hits" ] && fail "staged change contains what looks like an API key. Move it to .env."
  return 0
}

check_commits() {
  local c tmp
  tmp=$(mktemp)
  for c in $(git rev-list "$@"); do
    git log -1 --format=%B "$c" > "$tmp"
    ( check_message "$tmp" ) || { rm -f "$tmp"; fail "in commit $(git log -1 --format='%h %s' "$c")"; }
    ( check_identity "$(git log -1 --format='%an <%ae>' "$c")" ) || { rm -f "$tmp"; exit 1; }
    ( check_identity "$(git log -1 --format='%cn <%ce>' "$c")" ) || { rm -f "$tmp"; exit 1; }
  done
  rm -f "$tmp"
  return 0
}

case "${1:-}" in
  message)  check_message "$2" ;;
  identity) check_identity "$2" ;;
  staged)   check_staged ;;
  commits)  shift; check_commits "$@" ;;
  *) echo "usage: guard.sh message|identity|staged|commits ..." >&2; exit 2 ;;
esac
