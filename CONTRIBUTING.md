# Contributing

Issues and pull requests are welcome.

1. Install the hooks once after cloning:

   ```bash
   git config core.hooksPath .githooks
   ```

2. Run the smoke test before opening a PR:

   ```bash
   python code/smoke_test.py
   ```

3. Keep commits under your own name and email.

You may use AI tools to help you, but you are the author of what you submit.
Commits and PRs must not carry AI attribution: no `Co-Authored-By` lines for
models or tools, no "Generated with ..." footers, no bot identities. The hooks
and CI reject them. See [AGENTS.md](AGENTS.md) for the full rules agents follow
here.
