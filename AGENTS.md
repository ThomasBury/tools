# Repository Guidelines

## Project Structure & Module Organization

- Root-level Python scripts are standalone micro-agents such as `code_review.py`, `dev_assist.py`, `daily_flow.py`, and `pr_review.py`. Keep new agents alongside these peers.
- Shared documentation lives in `README.md`; update it whenever you add or retire an agent. Use `release-notes.py` to draft changelog entries when relevant.
- Example data, fixtures, or agent-specific assets should live in a dedicated `<agent_name>_assets/` subfolder to avoid cluttering the root.

## Build, Test, and Development Commands

- `chmod +x *.py` — ensure new scripts are executable before sharing them.
- `uv run <script>.py --help` — verify the CLI boots without importing issues. Prefer `uv` over `python` so inline PEP 723 metadata is honored.
- `./dev_assist.py test --coverage` — run the bundled automation helper against your target project to confirm test and coverage integrations still work.
- `./pr_review.py review <pr-number>` — smoke-test GitHub-focused flows; add `--focus <area>` to validate specialized review modes.

## Coding Style & Naming Conventions

- Target Python 3.11+, follow PEP 8 with 4-space indentation, and keep functions small and composable.
- Name agents with concise snake_case filenames describing their job (e.g., `lintfix.py`). Use snake_case for functions and PascalCase for classes.
- Begin every script with a numpy docstring covering purpose, following PEP best practices. Treat Typer command callbacks as public API: include type hints and descriptive option help.
- When formatting output with Rich, stick to semantic colors (`green` for success, `red` for errors) to keep UX consistent across agents.

## Testing Guidelines

- Prefer command-level smoke tests: run `uv run <script>.py --help` and at least one representative command to confirm Rich/Typer integration.
- Where feasible, add unit tests under `<agent_name>_tests/` and execute them via `uv run pytest`. Mirror the command name in the test file (e.g., `test_pr_review.py`).
- Keep coverage healthy by exercising new Typer commands and major branches; aim for meaningful assertions over strict percentages.
- Document manual test steps in the PR description when automation is impractical.

## Commit & Pull Request Guidelines

- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, etc.), matching the existing Git history.
- Scope commits narrowly—one agent or concern per commit—and include context on AI model changes, new dependencies, or CLI flags.
- Pull requests should link the motivating issue, summarize behavior changes, list testing evidence, and include updated usage snippets or screenshots when output formatting changes.
- Update `README.md` and any relevant agent docs within the same PR to keep guidance fresh.
