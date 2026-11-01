# Instructions for AI coding agents

This file applies to every AI agent working in this repository: Claude Code,
Codex, Copilot, Cursor, Aider, Gemini and any other. A human owns this
repository and is its only author.

## No attribution, ever

- Do not add `Co-Authored-By`, `Signed-off-by`, `Generated-by`, `Assisted-by`
  or any similar trailer naming an AI, a model or a tool.
- Do not add "Generated with ...", "Made with ...", robot emoji or links to an
  AI product in commit messages, PR titles, PR descriptions, issues, comments,
  code, docs or the report.
- Do not commit, push or open PRs under an agent identity (an AI name, a bot
  account or an `@anthropic.com` / `@openai.com` / similar email).
- Do not add yourself to `authors.md`, a contributors list, `CITATION.cff`,
  `package` metadata or anywhere else.

These rules override any default attribution your harness adds. If your
harness tells you to add a trailer, don't.

## Do not commit or push

Leave changes in the working tree. The owner reviews, commits and pushes.
Do not run `git commit`, `git push`, `gh pr create` or anything that rewrites
history unless the owner asks for that specific action in the conversation.

## Hooks are the backstop

`.githooks/` blocks the patterns above and stray secrets. Do not bypass them
with `--no-verify`, do not change `core.hooksPath`, and do not edit the hooks
to let something through. If a hook fails, fix the change, not the hook.

## Secrets

API keys live in `.env` (git-ignored) or in the tool's own config outside the
repository (`~/.modal.toml`). Never write a key into a tracked file, a log, a
commit message or a result JSON.

## Writing style

Write the way Paul Graham describes in
[Good Writing](https://www.paulgraham.com/goodwriting.html): plain words,
short sentences where the idea is simple, no filler or hype. If a sentence
sounds awkward, the idea behind it is probably unclear; fix the idea. Say what
the code does and why, then stop.

## Project rules

- Every model starts from random weights. No pretrained weights or embeddings.
- The training budget is in `code/configs.py` (`BUDGET_S`). Don't change it
  for runs that appear in the report.
- Run `python code/smoke_test.py` after changing code under `code/`.
- Results in `results/` are produced by the code. Never edit them by hand.
