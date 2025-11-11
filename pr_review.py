#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
#   "llm-gemini>=0.24",
#   "httpx>=0.27",
#   "pydantic>=2.0",
# ]
# ///
"""
AI-powered GitHub PR reviewer using gh CLI.

Extended Summary
----------------
This module provides a comprehensive tool for managing GitHub pull requests
with AI assistance. It integrates with the GitHub CLI (gh) for repository
operations and uses large language models for intelligent code review and
description generation.

Features
--------
- Pre-flight PR preparation with best-practice checks
- AI-crafted PR descriptions using configurable models
- Pull request creation via GitHub CLI
- Autonomous AI-driven reviews that analyze diffs and determine focus areas
- Chained review execution across multiple focus areas
- AI-suggested follow-up actions for addressing review findings
- Integration with GitHub's review system for posting comments
- Support for multiple LLM models via the llm library

Usage
-----
Basic usage examples:

.. code-block:: bash

    ./pr_review.py prepare --describe  # Inspect branch and draft PR description
    ./pr_review.py create --title "Add feature" --body "..."  # Create PR
    ./pr_review.py review 123  # Autonomous AI-driven review
    ./pr_review.py check  # List PRs needing review
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess as sp
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import llm
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Config file handling
CONFIG_FILE = Path(__file__).parent / "pr_review_config.json"

def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Default model handling
DEFAULT_MODEL_FALLBACK = "gemini-2.5-flash-lite"


def get_default_model() -> str:
    """Return the persisted default model or fallback."""

    return load_config().get("model", DEFAULT_MODEL_FALLBACK)


def resolve_model_selection(selected: Optional[str]) -> str:
    """Resolve the model option and persist new selections."""

    default_model = get_default_model()
    if selected:
        choice = selected.strip()
        if choice and choice != load_config().get("model"):
            config = load_config()
            config["model"] = choice
            save_config(config)
        return choice or default_model
    return default_model


DEFAULT_MODEL_DISPLAY = get_default_model()

PR_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Concise pull request title (<= 72 characters).",
        },
        "body": {
            "type": "string",
            "description": "Markdown formatted pull request body.",
        },
    },
    "required": ["title", "body"],
}
STRUCTURED_CAPABILITY_KEYS = {"schema", "json_schema", "structured-output"}


class RepositoryStateError(Exception):
    """Custom error for repository inspection issues."""


def model_supports_structured_output(model: Any) -> bool:
    """Best-effort detection for models that can honour JSON schemas."""

    for attr_name in (
        "supports_schema",
        "supports_schemas",
        "supports_structured_output",
        "supports_json_schema",
    ):
        attr = getattr(model, attr_name, None)
        if isinstance(attr, bool):
            return attr
        if callable(attr):
            try:
                return bool(attr())
            except TypeError:
                continue

    capabilities = getattr(model, "capabilities", None)
    if isinstance(capabilities, (set, list, tuple)):
        for capability in capabilities:
            if capability in STRUCTURED_CAPABILITY_KEYS:
                return True

    return False


def parse_structured_response(raw: str) -> Optional[Any]:
    """Extract structured JSON content (object or array) from raw model output."""

    candidates = [raw.strip()]

    fenced_matcher = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    candidates.extend(match.group(1).strip() for match in fenced_matcher.finditer(raw))

    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start:end].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            if isinstance(payload, (dict, list)):
                return payload
        except json.JSONDecodeError:
            continue

    return None


def build_fallback_body(raw: str) -> str:
    """Create a safe markdown body when structured parsing fails."""

    snippet = raw.strip() or "No additional output provided."
    if len(snippet) > 1500:
        snippet = snippet[:1500].rstrip() + "\n..."

    safe_snippet = snippet.replace("```", "``\\`")
    return (
        "## Summary\n"
        "- Structured output parsing failed; using raw model text below.\n\n"
        "## Raw Model Output\n"
        "```\n"
        f"{safe_snippet}\n"
        "```\n"
    )


def extract_summary_section(review_text: str) -> str:
    """Pull the summary section from an AI-generated review."""

    lines = review_text.splitlines()
    collected: list[str] = []
    capturing = False

    def starts_summary(line: str) -> bool:
        normalized = line.lower()
        return "**summary**" in normalized or normalized.startswith("summary")

    def starts_next_section(line: str) -> bool:
        normalized = line.lower()
        if not normalized:
            return False
        if "**strength" in normalized or "**issues" in normalized or "**suggestion" in normalized:
            return True
        if normalized[0].isdigit() and normalized[:2].isdigit():
            return True
        if normalized.startswith("strength"):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if not capturing:
            if starts_summary(stripped):
                capturing = True
                after = stripped.split("**Summary**", 1)
                if len(after) == 2 and after[1].strip():
                    collected.append(after[1].strip())
                    continue
                after = stripped.split("Summary", 1)
                if len(after) == 2 and after[1].strip():
                    collected.append(after[1].strip())
                continue
        else:
            if starts_next_section(stripped):
                break
            collected.append(line)

    summary = "\n".join(s.strip() for s in collected if s.strip())
    return summary.strip()


def git_output(args: list[str], *, strip: bool = True) -> Optional[str]:
    """Run a git command and return its stdout.

    Parameters
    ----------
    args : list[str]
        Arguments to pass to the git command.
    strip : bool, optional
        Whether to strip whitespace from the output. Default is True.

    Returns
    -------
    Optional[str]
        The stdout of the command, or None if the command failed.
    """

    try:
        result = sp.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (sp.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip() if strip else result.stdout


def is_git_repository() -> bool:
    """Return True when current working directory is inside a git repository."""

    return git_output(["rev-parse", "--is-inside-work-tree"]) == "true"


def warn_if_gh_missing() -> None:
    """Emit a warning if the GitHub CLI is not available."""

    if shutil.which("gh") is None:
        console.print(
            "[yellow]Warning:[/yellow] GitHub CLI (gh) not detected. "
            "Install from https://cli.github.com/ for full functionality."
        )


def list_local_branches() -> list[str]:
    """List all local branches in the repository.

    Returns
    -------
    list[str]
        List of local branch names, excluding any remote tracking branches.
    """
    output = git_output(["for-each-ref", "--format=%(refname:short)", "refs/heads"], strip=False)
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def prompt_for_base_branch(initial: RepoState) -> tuple[str, RepoState]:
    """Prompt the user to choose the base branch for the PR.

    Displays a list of candidate branches and allows selection by name or number.
    Refreshes repository state with the selected base branch.

    Parameters
    ----------
    initial : RepoState
        Initial repository state to determine default candidates.

    Returns
    -------
    tuple[str, RepoState]
        Selected base branch name and updated repository state.
    """

    default_base = initial.base_branch or "main"
    candidates = list(dict.fromkeys([
        default_base,
        "main",
        "develop",
        *list_local_branches(),
    ]))

    console.print("\n[bold]Select a base branch to merge into.[/bold]")
    for idx, branch in enumerate(candidates, start=1):
        marker = "(default)" if branch == default_base else ""
        console.print(f"  [cyan]{idx}[/cyan]. {branch} {marker}")

    value = typer.prompt(
        "Enter branch name or number",
        default=default_base,
    ).strip()

    selected: str
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(candidates):
            selected = candidates[index]
        else:
            selected = default_base
    else:
        selected = value

    if selected.startswith("origin/"):
        selected = selected.split("/", 1)[1]

    try:
        refreshed_state = inspect_repository(selected)
    except RepositoryStateError as exc:
        console.print(
            f"[yellow]Falling back to current branch comparison; unable to inspect '{selected}'.[/yellow]"
        )
        console.print(f"[yellow]- Reason: {exc}[/yellow]")
        refreshed_state = initial
    return selected, refreshed_state


def get_branch_remote(branch: str) -> tuple[Optional[str], bool]:
    """Determine remote for the given branch.

    Parameters
    ----------
    branch : str
        Name of the branch.

    Returns
    -------
    tuple[Optional[str], bool]
        Remote name and whether it came from explicit branch configuration.
    """
    remote = git_output(["config", f"branch.{branch}.remote"])
    if remote:
        return remote, True

    remotes_output = git_output(["remote"])
    if not remotes_output:
        return None, False

    for candidate in remotes_output.splitlines():
        candidate = candidate.strip()
        if candidate:
            return candidate, False

    return None, False


def push_branch_if_needed(state: RepoState) -> None:
    """Push the current branch to remote if necessary for PR creation.

    Checks if the branch has an upstream or is ahead of remote, and prompts
    the user to push if needed. Aborts PR creation if push fails or is declined.

    Parameters
    ----------
    state : RepoState
        Current repository state information.

    Raises
    ------
    typer.Exit
        If the user declines to push or if the push command fails.
    """
    upstream_missing = not state.upstream
    ahead_only = state.ahead > 0

    if not upstream_missing and not ahead_only:
        return

    reason = "No upstream configured" if upstream_missing else "Local branch is ahead of remote"
    console.print(f"[yellow]{reason}. A push is required before creating the PR.[/yellow]")

    if not typer.confirm("Push current branch now?", default=True):
        console.print("[red]Cannot create PR without pushing the branch. Aborting.[/red]")
        raise typer.Exit(1)

    remote, remote_configured = get_branch_remote(state.branch)
    if not remote:
        console.print(
            "[red]No git remotes are configured. Add a remote (e.g., `git remote add origin <url>`) before continuing.[/red]"
        )
        raise typer.Exit(1)
    if not remote_configured:
        console.print(
            f"[yellow]Branch '{state.branch}' has no upstream configured. Using remote '{remote}' for the push.[/yellow]"
        )

    if upstream_missing:
        push_cmd = ["git", "push", "-u", remote, state.branch]
    else:
        push_cmd = ["git", "push"]

    console.print(f"[cyan]Running: {' '.join(push_cmd)}[/cyan]")
    try:
        sp.run(push_cmd, check=True)
    except sp.CalledProcessError as exc:
        console.print("[red]Failed to push branch.[/red]")
        console.print(f"[red]- Command:[/red] {' '.join(push_cmd)}")
        console.print(f"[red]- Exit code:[/red] {exc.returncode}")

        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        output_chunks = []
        if stderr:
            output_chunks.append(("stderr", stderr))
        if stdout:
            output_chunks.append(("stdout", stdout))

        if output_chunks:
            for label, chunk in output_chunks:
                console.print(
                    Panel(
                        chunk,
                        title=f"git push {label}",
                        border_style="red",
                    )
                )
        else:
            console.print(
                "[yellow]No output captured. Re-run with `GIT_TRACE=1` or `GIT_CURL_VERBOSE=1` for additional details.[/yellow]"
            )
        raise typer.Exit(1)

def get_remote_default_branch(remote: str) -> Optional[str]:
    """Return the remote's HEAD branch if available."""

    remote_head = git_output(["symbolic-ref", f"refs/remotes/{remote}/HEAD"])
    if remote_head:
        parts = remote_head.split("/")
        if parts:
            return parts[-1]

    remote_show = git_output(["remote", "show", remote], strip=False)
    if not remote_show:
        return None

    for line in remote_show.splitlines():
        line = line.strip()
        if line.lower().startswith("head branch:"):
            return line.split(":", 1)[1].strip()
    return None


def resolve_base_reference(explicit_base: Optional[str]) -> tuple[str, str, Optional[str]]:
    """Determine the base branch and diff reference for comparisons.

    Parameters
    ----------
    explicit_base : Optional[str]
        Explicit base branch name, if provided.

    Returns
    -------
    tuple[str, str, Optional[str]]
        Base branch name, diff reference, and upstream branch (if any).
    """

    upstream = git_output(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    current_branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    if explicit_base:
        if "/" in explicit_base:
            _, branch = explicit_base.split("/", 1)
            base_branch = branch
        else:
            base_branch = explicit_base
        diff_ref = explicit_base
        return base_branch, diff_ref, upstream

    if upstream:
        if "/" in upstream:
            remote, branch = upstream.split("/", 1)
            if branch == current_branch:
                default_branch = get_remote_default_branch(remote)
                if default_branch:
                    return default_branch, f"{remote}/{default_branch}", upstream
                console.print(
                    f"[yellow]Unable to infer base from upstream '{upstream}'. Falling back to default branches.[/yellow]"
                )
            else:
                return branch, f"{remote}/{branch}", upstream
        elif upstream != current_branch:
            return upstream, upstream, upstream

    remotes_output = git_output(["remote"], strip=False)
    remotes = [line.strip() for line in remotes_output.splitlines()] if remotes_output else []
    remotes = [remote for remote in remotes if remote]

    if not remotes:
        console.print(
            "[yellow]No git remotes detected. Defaulting to local 'main' for comparisons.[/yellow]"
        )
        return "main", "main", upstream

    preferred_remote = "origin" if "origin" in remotes else remotes[0]
    if preferred_remote != "origin":
        console.print(
            "[yellow]Remote 'origin' not found. Unable to infer default base branch automatically.[/yellow]"
        )
        try:
            if typer.confirm(
                "Would you like to provide the base reference manually?",
                default=False,
            ):
                manual_base = typer.prompt(
                    "Base reference (e.g., main or upstream/main)", default="main"
                ).strip()
                if manual_base:
                    if "/" in manual_base:
                        _, branch = manual_base.split("/", 1)
                        base_branch = branch
                        diff_ref = manual_base
                    else:
                        base_branch = manual_base
                        diff_ref = manual_base
                    return base_branch, diff_ref, upstream
        except typer.Abort as exc:
            raise RepositoryStateError("Base reference selection aborted.") from exc

        console.print(
            f"[yellow]Falling back to remote '{preferred_remote}' for base comparison.[/yellow]"
        )

    default_branch = get_remote_default_branch(preferred_remote)
    if default_branch:
        return default_branch, f"{preferred_remote}/{default_branch}", upstream

    base_branch = "main"
    diff_ref = f"{preferred_remote}/main"
    return base_branch, diff_ref, upstream


def inspect_repository(base: Optional[str] = None) -> RepoState:
    """Collect repository status details for PR preparation.

    Parameters
    ----------
    base : Optional[str], optional
        Base branch to compare against. If None, uses upstream or defaults.

    Returns
    -------
    RepoState
        Repository state snapshot ready for display or further processing.

    Raises
    ------
    RepositoryStateError
        If not in a git repository or if HEAD is detached.
    """

    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is None:
        raise RepositoryStateError("Not inside a git repository.")
    if branch == "HEAD":
        raise RepositoryStateError(
            "Detached HEAD detected. Switch to a branch before continuing "
            "(e.g., `git switch <branch>` or `git switch -c <new-branch>`)."
        )

    base_branch, diff_ref, upstream = resolve_base_reference(base)

    status_output = git_output(["status", "--porcelain"])
    clean = status_output == "" if status_output is not None else True
    untracked = []
    if status_output:
        untracked = [line[3:] for line in status_output.splitlines() if line.startswith("?? ")]

    ahead = behind = 0
    ahead_output = git_output(["rev-list", "--left-right", "--count", f"{diff_ref}...HEAD"])
    if ahead_output:
        try:
            behind_str, ahead_str = ahead_output.split()
            behind = int(behind_str)
            ahead = int(ahead_str)
        except ValueError:
            behind = ahead = 0

    commits_output = git_output(["log", "--oneline", f"{diff_ref}..HEAD"])
    commits = commits_output.splitlines()[:10] if commits_output else []

    diff_stat_output = git_output(["diff", "--stat", f"{diff_ref}...HEAD"], strip=False)
    diff_stat = diff_stat_output.strip() if diff_stat_output else "(no diff)"

    return RepoState(
        branch=branch,
        base_branch=base_branch,
        diff_ref=diff_ref,
        upstream=upstream,
        clean=clean,
        ahead=ahead,
        behind=behind,
        untracked=untracked,
        commits=commits,
        diff_stat=diff_stat,
    )


def get_diff(
    diff_ref: str,
    *,
    allow_staged_fallback: bool = True,
    prompt_on_fallback: bool = True,
) -> str:
    """Return diff between base reference and HEAD.

    Parameters
    ----------
    diff_ref : str
        Base reference for comparison (e.g., branch name or commit).
    allow_staged_fallback : bool, optional
        Whether to fall back to staged changes when the base diff is unavailable.
        Defaults to True.
    prompt_on_fallback : bool, optional
        Prompt the user before using staged changes as a fallback. Defaults to True.

    Returns
    -------
    str
        Git diff output, or empty string if no diff available.

    Raises
    ------
    RepositoryStateError
        If the diff cannot be computed and no staged changes are available.
    """

    diff_output = git_output(["diff", f"{diff_ref}...HEAD"], strip=False)
    if diff_output is None:
        if not allow_staged_fallback:
            raise RepositoryStateError(
                f"Unable to compute diff against '{diff_ref}', and staged fallback is disabled."
            )
        staged_output = git_output(["diff", "--staged"], strip=False)
        if staged_output:
            if prompt_on_fallback:
                try:
                    use_staged = typer.confirm(
                        (
                            f"Unable to diff against '{diff_ref}'. "
                            "Use staged changes instead?"
                        ),
                        default=True,
                    )
                except typer.Abort as exc:
                    raise RepositoryStateError("Diff generation aborted by user.") from exc
                if not use_staged:
                    raise RepositoryStateError(
                        f"Unable to compute diff against '{diff_ref}'. "
                        "Provide a valid base reference or fetch the base branch."
                    )
            console.print(
                "[yellow]Warning:[/yellow] Unable to diff against "
                f"'{diff_ref}'. Using staged changes instead. "
                "Run `git fetch` or specify --base to compare against a valid reference."
            )
            return staged_output
        raise RepositoryStateError(
            f"Unable to compute diff against '{diff_ref}'. Ensure the reference exists or fetch the base branch."
        )
    return diff_output


def generate_pr_description(diff: str, model_name: str) -> PRDescription:
    """Use an LLM to craft a PR title and body.

    Parameters
    ----------
    diff : str
        Git diff content to analyze for description generation.
    model_name : str
        Name of the LLM model to use for generation.

    Returns
    -------
    PRDescription
        Generated PR title and body.

    Raises
    ------
    Exception
        If LLM generation fails, returns a default description.
    """

    if not diff.strip():
        return PRDescription(
            title="chore: update",
            body="## Summary\n- No diff available for description generation.\n\n## Testing\n- Not specified\n",
        )

    prompt = f"""
You are an expert GitHub contributor preparing a pull request.
Analyze the diff below and produce a concise title (<= 72 characters) and a
Markdown body using this structure:

## Summary
- Bullet list of key changes

## Testing
- Bullet list describing validations run (invent reasonable defaults if not provided)

## Notes
- Optional reminders or follow-ups

Respond with a JSON object containing exactly two fields: "title" and "body".
Ensure the JSON is valid and does not contain code fences or extra text.

Git Diff:
```diff
{diff[:15000]}
```
"""

    try:
        model = llm.get_model(model_name)
    except Exception as exc:  # pragma: no cover - LLM failures
        console.print(f"[red]LLM error while loading model '{model_name}':[/red] {exc}")
        return PRDescription(
            title="chore: update",
            body="## Summary\n- Description generation failed.\n\n## Testing\n- Not specified\n",
        )

    structured_response = None
    if model_supports_structured_output(model):
        try:
            structured_response = model.prompt(prompt, schema=PR_DESCRIPTION_SCHEMA)
        except TypeError:
            structured_response = None
        except Exception as exc:
            console.print(
                f"[yellow]Structured output not available for model '{model_name}':[/yellow] {exc}"
            )
            structured_response = None

    response = structured_response
    if response is None:
        try:
            response = model.prompt(prompt)
        except Exception as exc:  # pragma: no cover - LLM failures
            console.print(f"[red]LLM error while generating PR description:[/red] {exc}")
            return PRDescription(
                title="chore: update",
                body="## Summary\n- Description generation failed.\n\n## Testing\n- Not specified\n",
            )

    raw = response.text().strip()

    data = parse_structured_response(raw)
    if data is None and structured_response is not None:
        console.print(
            "[yellow]Warning: structured response parse failed; falling back to raw text parsing.[/yellow]"
        )
        # Try again without schema prompt
        try:
            fallback_response = model.prompt(prompt)
            raw = fallback_response.text().strip()
            data = parse_structured_response(raw)
        except Exception:  # pragma: no cover - avoid masking original result
            data = None

    if isinstance(data, dict):
        title = data.get("title") or "chore: update"
        body = data.get("body") or "## Summary\n- Description unavailable\n"
        return PRDescription(title=title.strip(), body=body.strip())

    console.print("[yellow]Warning: LLM returned invalid JSON. Using raw model text as fallback.[/yellow]")
    return PRDescription(title="chore: update", body=build_fallback_body(raw))


def display_repo_state(state: RepoState) -> None:
    """Render repository insights and best-practice checks.

    Parameters
    ----------
    state : RepoState
        Repository state to display, including branch info and status.
    """

    summary = Table(show_header=False, box=None)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Current branch", state.branch)
    summary.add_row("Base branch", state.base_branch)
    summary.add_row("Diff reference", state.diff_ref)
    summary.add_row("Upstream", state.upstream or "(none)")
    summary.add_row("Ahead/Behind", f"{state.ahead}/{state.behind}")

    console.print(Panel(summary, title="Repository State", border_style="cyan"))

    checks = Table(title="PR Readiness", show_header=False)
    checks.add_column("Status", style="bold")
    checks.add_column("Check", style="white")
    checks.add_column("Details", style="dim")

    def status_row(condition: bool, label: str, ok_msg: str, warn_msg: str) -> None:
        status = "[green]PASS[/green]" if condition else "[yellow]WARN[/yellow]"
        checks.add_row(status, label, ok_msg if condition else warn_msg)

    status_row(state.clean, "Working tree clean", "No outstanding changes", "Commit or stash changes")
    status_row(
        state.ahead > 0,
        "Commits ahead of base",
        f"{state.ahead} commit(s) ready",
        "No new commits to include",
    )
    status_row(
        state.behind == 0,
        "Branch up-to-date",
        "Synchronized with base",
        f"Behind by {state.behind} commit(s)",
    )
    status_row(
        not state.untracked,
        "No untracked files",
        "Clean working tree",
        f"{len(state.untracked)} untracked file(s)",
    )

    console.print(checks)

    console.print(Panel(state.diff_stat or "(no diff)", title="Diff Summary", border_style="magenta"))

    if state.commits:
        commits_table = Table(title="Commits Since Base", show_header=False)
        commits_table.add_column("Commit", style="white")
        for entry in state.commits:
            commits_table.add_row(entry)
        console.print(commits_table)

    if state.untracked:
        untracked_panel = Panel(
            "\n".join(state.untracked),
            title="Untracked Files",
            border_style="yellow",
        )
        console.print(untracked_panel)

class ReviewFocus(str, Enum):
    """Enumeration of review focus areas.

    Attributes
    ----------
    SECURITY : str
        Focus on security issues.
    PERFORMANCE : str
        Focus on performance concerns.
    TESTS : str
        Focus on testing coverage and quality.
    DOCS : str
        Focus on documentation completeness.
    GENERAL : str
        General balanced review.
    """
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTS = "tests"
    DOCS = "docs"
    GENERAL = "general"


@dataclass
class PullRequest:
    """Data class representing pull request information.

    Attributes
    ----------
    number : int
        Pull request number.
    title : str
        Pull request title.
    author : str
        Author login name.
    base_branch : str
        Base branch name.
    head_branch : str
        Head branch name.
    repo : str
        Repository name (owner/name).
    additions : int
        Number of added lines.
    deletions : int
        Number of deleted lines.
    changed_files : int
        Number of changed files.
    draft : bool, optional
        Whether the PR is a draft. Default is False.
    """
    number: int
    title: str
    author: str
    base_branch: str
    head_branch: str
    repo: str
    additions: int
    deletions: int
    changed_files: int
    draft: bool = False
    
    @property
    def size_category(self) -> str:
        """Categorize the pull request size based on total changes.

        Returns
        -------
        str
            Size category: 'tiny' (<50 changes), 'small' (50-249),
            'medium' (250-999), or 'large' (>=1000 changes).
        """
        total = self.additions + self.deletions
        if total < 50:
            return "tiny"
        elif total < 250:
            return "small"
        elif total < 1000:
            return "medium"
        else:
            return "large"


@dataclass
class PRDescription:
    """Data class for auto-generated pull request description.

    Attributes
    ----------
    title : str
        Generated PR title.
    body : str
        Generated PR body in Markdown format.
    """

    title: str
    body: str


@dataclass
class RepoState:
    """Data class representing a snapshot of local repository state.

    Attributes
    ----------
    branch : str
        Current branch name.
    base_branch : str
        Base branch for comparison.
    diff_ref : str
        Reference for diff calculation.
    upstream : Optional[str]
        Upstream branch if configured.
    clean : bool
        Whether working tree is clean.
    ahead : int
        Number of commits ahead of base.
    behind : int
        Number of commits behind base.
    untracked : list[str]
        List of untracked files.
    commits : list[str]
        Recent commits since base.
    diff_stat : str
        Git diff --stat output.
    """

    branch: str
    base_branch: str
    diff_ref: str
    upstream: Optional[str]
    clean: bool
    ahead: int
    behind: int
    untracked: list[str]
    commits: list[str]
    diff_stat: str

class GitHubCLI:
    """Wrapper class for GitHub CLI operations.

    Provides static methods to interact with GitHub via the gh CLI tool.
    """
    
    @staticmethod
    def check_auth() -> bool:
        """Check if gh CLI is authenticated.

        Returns
        -------
        bool
            True if authenticated, False otherwise.
        """
        try:
            result = sp.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def create_pr(
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
        base: Optional[str] = None,
        head: Optional[str] = None,
        draft: bool = False,
        fill: bool = False,
        repo: Optional[str] = None,
        reviewers: tuple[str, ...] = (),
        assignees: tuple[str, ...] = (),
        labels: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Create a pull request using gh CLI and return metadata.

        Parameters
        ----------
        title : Optional[str], optional
            PR title. Default is None.
        body : Optional[str], optional
            PR body. Default is None.
        base : Optional[str], optional
            Base branch. Default is None.
        head : Optional[str], optional
            Head branch. Default is None.
        draft : bool, optional
            Create as draft. Default is False.
        fill : bool, optional
            Autofill title/body. Default is False.
        repo : Optional[str], optional
            Repository (owner/name). Default is None.
        reviewers : tuple[str, ...], optional
            Reviewers to assign. Default is empty tuple.
        assignees : tuple[str, ...], optional
            Assignees to assign. Default is empty tuple.
        labels : tuple[str, ...], optional
            Labels to apply. Default is empty tuple.

        Returns
        -------
        dict[str, Any]
            Metadata about the created PR, including number, url, etc.

        Raises
        ------
        sp.CalledProcessError
            If the gh command fails.
        """

        def build_args(include_json: bool) -> list[str]:
            args = ["gh", "pr", "create"]
            if include_json:
                args.extend(["--json", "number,url"])
            if repo:
                args.extend(["--repo", repo])
            if base:
                args.extend(["--base", base])
            if head:
                args.extend(["--head", head])
            if title:
                args.extend(["--title", title])
            if body:
                args.extend(["--body", body])
            if draft:
                args.append("--draft")
            if fill:
                args.append("--fill")
            for reviewer in reviewers:
                args.extend(["--reviewer", reviewer])
            for assignee in assignees:
                args.extend(["--assignee", assignee])
            for label in labels:
                args.extend(["--label", label])
            return args

        cmd = build_args(include_json=True)
        result = sp.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0 and result.stderr:
            stderr_lower = result.stderr.lower()
            if "--json" in stderr_lower and ("unknown flag" in stderr_lower or "flag provided but not defined" in stderr_lower):
                # Retry without --json (older gh release)
                cmd = build_args(include_json=False)
                legacy = sp.run(cmd, capture_output=True, text=True, check=False)
                if legacy.returncode != 0:
                    raise sp.CalledProcessError(
                        legacy.returncode,
                        cmd,
                        output=legacy.stdout,
                        stderr=legacy.stderr,
                    )
                output_text = (legacy.stdout or legacy.stderr or "").strip()
                return {"output": output_text}

        if result.returncode != 0:
            raise sp.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {"output": result.stdout.strip()}

    @staticmethod
    def get_pr_info(pr_number: int, repo: Optional[str] = None) -> dict[str, Any]:
        """Fetch PR information.

        Parameters
        ----------
        pr_number : int
            Pull request number.
        repo : Optional[str], optional
            Repository (owner/name). Default is None.

        Returns
        -------
        dict[str, Any]
            PR information as returned by gh CLI.
        """
        cmd = ["gh", "pr", "view", str(pr_number), "--json",
               "number,title,author,baseRefName,headRefName,additions,deletions,changedFiles,isDraft"]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    @staticmethod
    def get_pr_diff(pr_number: int, repo: Optional[str] = None) -> str:
        """Fetch PR diff.

        Parameters
        ----------
        pr_number : int
            Pull request number.
        repo : Optional[str], optional
            Repository (owner/name). Default is None.

        Returns
        -------
        str
            The diff content as string.
        """
        cmd = ["gh", "pr", "diff", str(pr_number)]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    
    @staticmethod
    def get_pr_files(pr_number: int, repo: Optional[str] = None) -> list[str]:
        """Get list of files changed in PR.

        Parameters
        ----------
        pr_number : int
            Pull request number.
        repo : Optional[str], optional
            Repository (owner/name). Default is None.

        Returns
        -------
        list[str]
            List of file paths changed in the PR.
        """
        cmd = ["gh", "pr", "view", str(pr_number), "--json", "files"]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return [f["path"] for f in data.get("files", [])]
    
    @staticmethod
    def list_prs_to_review(repo: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
        """List PRs that need review.

        Parameters
        ----------
        repo : Optional[str], optional
            Repository (owner/name). Default is None.
        limit : int, optional
            Maximum number of PRs to return. Default is 10.

        Returns
        -------
        list[dict[str, Any]]
            List of PR data dictionaries.
        """
        cmd = [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,author,createdAt,isDraft",
            "--limit",
            str(limit),
            "--search",
            "review-requested:@me -author:@me",
        ]
        if repo:
            cmd.extend(["--repo", repo])

        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    @staticmethod
    def post_review_comment(pr_number: int, body: str, repo: Optional[str] = None) -> None:
        """Post a review comment to PR.

        Parameters
        ----------
        pr_number : int
            Pull request number.
        body : str
            Comment body in Markdown.
        repo : Optional[str], optional
            Repository (owner/name). Default is None.
        """
        cmd = ["gh", "pr", "comment", str(pr_number), "--body", body]
        if repo:
            cmd.extend(["--repo", repo])
        
        sp.run(cmd, check=True)


def extract_pr_number(reference: str) -> Optional[int]:
    """Extract a PR number from a string such as a URL.

    Parameters
    ----------
    reference : str
        String containing PR reference, e.g., URL or number.

    Returns
    -------
    Optional[int]
        Extracted PR number, or None if not found.
    """
    match = re.search(r"/(?:pull|compare)/(?P<num>\d+)", reference)
    if match:
        try:
            return int(match.group("num"))
        except ValueError:
            return None
    digits = re.search(r"\b(\d{1,6})\b", reference)
    if digits:
        try:
            return int(digits.group(1))
        except ValueError:
            return None
    return None


def analyze_diff_for_focus_areas(diff: str, model_name: str) -> list[ReviewFocus]:
    """Use AI to analyze diff and determine relevant focus areas.

    Parameters
    ----------
    diff : str
        Git diff content.
    model_name : str
        LLM model to use for analysis.

    Returns
    -------
    list[ReviewFocus]
        Ordered list of focus areas to review, prioritized by relevance.
    """
    try:
        ai_model = llm.get_model(model_name)
    except llm.UnknownModelError:
        console.print(f"[red]Error:[/red] Unknown model '{model_name}' for focus analysis")
        return [ReviewFocus.GENERAL]

    prompt = f"""Analyze this git diff and determine which review focus areas are most relevant.
Return a JSON array of focus area names in priority order (most important first).

Available focus areas:
- security: SQL injection, XSS, CSRF, hardcoded secrets, unsafe deserialization, path traversal, auth flaws
- performance: N+1 queries, memory leaks, inefficient algorithms, blocking operations, cache opportunities
- tests: missing coverage, edge cases, test quality, mock usage, test performance
- docs: missing docstrings, outdated docs, README updates, code comments, type hints
- general: code quality, bugs, best practices, maintainability

Consider:
- File types and languages changed
- Size and complexity of changes
- Potential security implications
- Performance-sensitive code
- Test coverage needs
- Documentation requirements

Return only a JSON array like: ["security", "performance", "tests"]

Diff:
```diff
{diff[:10000]}  # Limit diff size for analysis
```

JSON array:"""

    try:
        with console.status(
            f"[cyan]Analyzing diff for focus areas with {model_name}...[/cyan]",
            spinner="dots",
        ):
            response = ai_model.prompt(prompt)
        raw = response.text().strip()

        # Parse JSON response
        focus_payload = parse_structured_response(raw)
        focus_candidates: list[str] = []

        if isinstance(focus_payload, list):
            focus_candidates = [str(item) for item in focus_payload]
        elif isinstance(focus_payload, dict):
            for key in ("focus", "focuses", "focus_areas", "focusAreas", "areas", "reviewAreas"):
                value = focus_payload.get(key)
                if value:
                    if isinstance(value, list):
                        focus_candidates = [str(item) for item in value]
                    else:
                        focus_candidates = [str(value)]
                    break
            if not focus_candidates:
                for value in focus_payload.values():
                    if isinstance(value, list):
                        focus_candidates = [str(item) for item in value]
                        break

        focus_candidates = [name.strip() for name in focus_candidates if name and str(name).strip()]

        if focus_candidates:
            focus_areas: list[ReviewFocus] = []
            for name in focus_candidates:
                try:
                    focus_areas.append(ReviewFocus(name.lower()))
                except ValueError:
                    continue  # Skip invalid focus areas
            if focus_areas:
                return focus_areas

    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Failed to analyze focus areas: {e}")

    # Fallback to general review
    return [ReviewFocus.GENERAL]


def suggest_followup_actions(review_text: str, pr: PullRequest, model_name: str) -> list[str]:
    """Suggest actionable follow-up steps based on the review.

    Parameters
    ----------
    review_text : str
        The AI-generated review text.
    pr : PullRequest
        Pull request information.
    model_name : str
        LLM model to use for suggestions.

    Returns
    -------
    list[str]
        List of suggested follow-up actions.
    """
    try:
        ai_model = llm.get_model(model_name)
    except llm.UnknownModelError:
        return []

    prompt = f"""Based on this AI review of a pull request, suggest specific actionable follow-up steps.
Focus on concrete actions the author can take to address issues found.

PR Info:
- Title: {pr.title}
- Size: {pr.size_category}
- Files: {pr.changed_files}

Review:
{review_text}

Suggest 2-4 specific, actionable follow-up steps. Format as a JSON array of strings.
Each suggestion should be clear and executable.

JSON array:"""

    try:
        with console.status(
            f"[cyan]Generating follow-up suggestions with {model_name}...[/cyan]",
            spinner="dots",
        ):
            response = ai_model.prompt(prompt)
        raw = response.text().strip()

        suggestions_payload = parse_structured_response(raw)
        suggestions_list: list[str] = []

        if isinstance(suggestions_payload, list):
            suggestions_list = [str(item) for item in suggestions_payload]
        elif isinstance(suggestions_payload, dict):
            for key in ("suggestions", "actions", "follow_up", "followUp", "steps"):
                value = suggestions_payload.get(key)
                if value:
                    if isinstance(value, list):
                        suggestions_list = [str(item) for item in value]
                    else:
                        suggestions_list = [str(value)]
                    break
            if not suggestions_list:
                for value in suggestions_payload.values():
                    if isinstance(value, list):
                        suggestions_list = [str(item) for item in value]
                        break

        suggestions_list = [item.strip() for item in suggestions_list if str(item).strip()]
        if suggestions_list:
            return suggestions_list

    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Failed to generate follow-up suggestions: {e}")

    return []


def build_review_prompt(pr: PullRequest, diff: str, focus: ReviewFocus) -> str:
    """Build AI review prompt based on PR and focus area.

    Parameters
    ----------
    pr : PullRequest
        Pull request information.
    diff : str
        Git diff content.
    focus : ReviewFocus
        Review focus area.

    Returns
    -------
    str
        Formatted prompt for AI review.
    """
    
    focus_prompts = {
        ReviewFocus.SECURITY: """
Focus on security issues:
- SQL injection, XSS, CSRF vulnerabilities
- Hardcoded secrets or credentials
- Unsafe deserialization
- Path traversal risks
- Authentication/authorization flaws
""",
        ReviewFocus.PERFORMANCE: """
Focus on performance concerns:
- Database query optimization (N+1 queries, missing indexes)
- Memory leaks or excessive allocations
- Inefficient algorithms or data structures
- Unnecessary blocking operations
- Cache opportunities
""",
        ReviewFocus.TESTS: """
Focus on testing:
- Missing test coverage for new functionality
- Edge cases not covered
- Test quality and assertions
- Mock/stub usage appropriateness
- Test performance and maintainability
""",
        ReviewFocus.DOCS: """
Focus on documentation:
- Missing or outdated docstrings
- API documentation completeness
- README updates needed
- Code comments clarity
- Type hints and annotations
""",
        ReviewFocus.GENERAL: """
Provide a balanced review covering:
- Code quality and maintainability
- Potential bugs and edge cases
- Best practices adherence
- Performance considerations
- Security implications
"""
    }
    
    return f"""You are a senior software engineer reviewing a pull request.

PR Information:
- Title: {pr.title}
- Author: {pr.author}
- Size: {pr.size_category} ({pr.additions} additions, {pr.deletions} deletions)
- Files changed: {pr.changed_files}
- Branch: {pr.head_branch} → {pr.base_branch}

Review Focus:
{focus_prompts[focus]}

Provide a structured review with:
1. **Summary** - Brief overview of changes
2. **Strengths** - What's done well
3. **Issues** - Problems found (if any)
4. **Suggestions** - Specific improvements
5. **Decision** - Approve, Request Changes, or Comment

Format as markdown. Be constructive and specific. Reference file names and line numbers where relevant.

Here's the diff to review:

```diff
{diff[:15000]}  # Truncate very large diffs
```
"""


def format_pr_table(prs: list[dict[str, Any]]) -> Table:
    """Format PRs as a rich table.

    Parameters
    ----------
    prs : list[dict[str, Any]]
        List of PR data dictionaries.

    Returns
    -------
    Table
        Rich table object for display.
    """
    table = Table(title="Pull Requests Needing Review")
    table.add_column("PR", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Author", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("Draft", style="red")
    
    for pr in prs:
        table.add_row(
            f"#{pr['number']}",
            pr["title"][:50] + ("..." if len(pr["title"]) > 50 else ""),
            pr["author"]["login"],
            pr["createdAt"].split("T")[0],
            "✓" if pr.get("isDraft") else ""
        )
    
    return table


@app.command()
def prepare(
    base: Optional[str] = typer.Option(None, "--base", help="Base branch to compare"),
    describe: bool = typer.Option(
        False,
        "--describe",
        help="Generate an AI-assisted PR title and body",
    ),
    skip_staged_fallback: bool = typer.Option(
        False,
        "--skip-staged-fallback",
        help="Do not use staged changes when the base diff is unavailable.",
    ),
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="Allow generating descriptions with uncommitted changes present.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help=f"LLM model to use for description generation (default: {DEFAULT_MODEL_DISPLAY})",
    ),
) -> None:
    """Inspect local repository state before creating a PR.

    Parameters
    ----------
    base : Optional[str], optional
        Base branch to compare against.
    describe : bool, optional
        Generate AI-assisted PR title and body.
    skip_staged_fallback : bool, optional
        Disable fallback to staged changes if the base diff cannot be computed.
    allow_dirty : bool, optional
        Allow generating descriptions when the working tree has uncommitted changes.
    model : Optional[str], optional
        LLM model for description generation. When provided, it becomes the new default.
    """

    if not is_git_repository():
        console.print("[red]Error:[/red] Not inside a git repository. Run this command from a project managed by git.")
        raise typer.Exit(1)

    warn_if_gh_missing()

    try:
        state = inspect_repository(base)
    except RepositoryStateError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    display_repo_state(state)

    resolved_model = resolve_model_selection(model)

    if describe:
        if not allow_dirty and not state.clean:
            console.print(
                "[red]Working tree has uncommitted changes. Use --allow-dirty to generate a description anyway.[/red]"
            )
            raise typer.Exit(1)
        try:
            diff = get_diff(
                state.diff_ref,
                allow_staged_fallback=not skip_staged_fallback,
            )
        except RepositoryStateError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        description = generate_pr_description(diff, resolved_model)
        console.print(
            Panel(
                f"[bold]Suggested Title[/bold]: {description.title}\n\n{description.body}",
                title="AI-Generated PR Description",
                border_style="green",
            )
        )


@app.command()
def create(
    title: Optional[str] = typer.Option(None, "--title", "-t", help="PR title"),
    body: Optional[str] = typer.Option(None, "--body", "-b", help="PR body"),
    base: Optional[str] = typer.Option(None, "--base", help="Base branch"),
    head: Optional[str] = typer.Option(None, "--head", help="Head branch"),
    draft: bool = typer.Option(False, "--draft", help="Create as draft"),
    fill: bool = typer.Option(False, "--fill", help="Autofill title/body from commits"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository (owner/name)"),
    reviewer: list[str] = typer.Option(
        [],
        "--reviewer",
        help="Assign reviewer (repeat for multiple)",
    ),
    assignee: list[str] = typer.Option(
        [],
        "--assignee",
        help="Assign user to PR (repeat for multiple)",
    ),
    label: list[str] = typer.Option(
        [],
        "--label",
        help="Apply label to PR (repeat for multiple)",
    ),
    describe: bool = typer.Option(
        False,
        "--describe",
        help="Use AI to generate PR title/body when not provided",
    ),
    skip_staged_fallback: bool = typer.Option(
        False,
        "--skip-staged-fallback",
        help="Do not use staged changes when the base diff is unavailable.",
    ),
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="Allow creating a PR with uncommitted changes",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    run_review: bool = typer.Option(
        False,
        "--run-review",
        help="Run the AI review after creating the PR",
    ),
    focus: Optional[ReviewFocus] = typer.Option(
        None,
        "--focus",
        help="Force a specific focus area when --run-review is used (default: AI selects)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help=f"Model used for AI generation (default: {DEFAULT_MODEL_DISPLAY})",
    ),
    post: bool = typer.Option(
        False,
        "--post",
        help="Post the AI review as a comment when --run-review is set",
    ),
    show_diff: bool = typer.Option(
        False,
        "--show-diff",
        help="Show diff before review when --run-review is used",
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary-only",
        help="Output a short AI summary when --run-review is used",
    ),
) -> None:
    """Create a pull request using the GitHub CLI.

    Parameters
    ----------
    title : Optional[str], optional
        PR title.
    body : Optional[str], optional
        PR body.
    base : Optional[str], optional
        Base branch.
    head : Optional[str], optional
        Head branch.
    draft : bool, optional
        Create as draft.
    repo : Optional[str], optional
        Repository (owner/name).
    describe : bool, optional
        Use AI to generate title/body.
    skip_staged_fallback : bool, optional
        Disable fallback to staged changes if the base diff cannot be computed.
    focus : Optional[ReviewFocus], optional
        Review focus area when --run-review is used. Defaults to AI-selected priorities.
    summary_only : bool, optional
        Output only the summary portion of the AI review when provided.
    model : Optional[str], optional
        Model for AI generation. When provided, it becomes the new default.
    """

    if not is_git_repository():
        console.print("[red]Error:[/red] Not inside a git repository. Run this command from a project managed by git.")
        raise typer.Exit(1)

    warn_if_gh_missing()

    if not GitHubCLI.check_auth():
        console.print("[red]Error:[/red] gh CLI not found or not authenticated")
        console.print("Install: https://cli.github.com")
        console.print("Then run: gh auth login")
        raise typer.Exit(1)

    resolved_model = resolve_model_selection(model)

    if fill and describe:
        console.print("[red]Error:[/red] --fill cannot be combined with --describe.")
        raise typer.Exit(1)

    if fill and (title or body):
        console.print("[red]Error:[/red] --fill cannot be combined with explicit title/body")
        raise typer.Exit(1)

    try:
        state = inspect_repository(base)
    except RepositoryStateError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if base is None:
        base, state = prompt_for_base_branch(state)
    else:
        base = base.strip()
        if base and base != state.base_branch:
            try:
                state = inspect_repository(base)
            except RepositoryStateError as exc:
                console.print(f"[yellow]Warning:[/yellow] Unable to inspect base '{base}': {exc}")

    display_repo_state(state)

    push_branch_if_needed(state)
    try:
        state = inspect_repository(base)
    except RepositoryStateError as exc:
        console.print(f"[red]Error:[/red] Unable to inspect repository after push: {exc}")
        raise typer.Exit(1)

    if not allow_dirty and not state.clean:
        console.print("[red]Working tree has uncommitted changes. Use --allow-dirty to override.[/red]")
        raise typer.Exit(1)

    if state.behind > 0:
        console.print(
            f"[yellow]Warning:[/yellow] Branch is behind base by {state.behind} commit(s). Consider pulling updates."
        )
    if state.ahead == 0:
        console.print(
            "[yellow]Warning:[/yellow] No new commits detected relative to the base reference."
        )

    try:
        diff = get_diff(
            state.diff_ref,
            allow_staged_fallback=not skip_staged_fallback,
            prompt_on_fallback=not yes,
        )
    except RepositoryStateError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    use_ai = describe or (not fill and (title is None or body is None))
    generated = None
    if use_ai:
        console.print(
            f"[cyan]Generating PR title and body with {resolved_model}...[/cyan]"
        )
        generated = generate_pr_description(diff, resolved_model)
        title = title or generated.title
        body = body or generated.body
        console.print(
            Panel(
                f"[bold]Suggested Title[/bold]: {title}\n\n{body}",
                title="AI-Generated Description",
                border_style="green",
            )
        )

    if not any([title, body, fill]):
        console.print(
            "[red]PR title and body are required unless --fill is specified or AI generation is enabled.[/red]"
        )
        raise typer.Exit(1)

    preview = Table(show_header=False, box=None)
    preview.add_column("Field", style="cyan")
    preview.add_column("Value", style="white")
    preview.add_row("Base branch", base or state.base_branch)
    preview.add_row("Head branch", head or state.branch)
    preview.add_row("Draft", "Yes" if draft else "No")
    if reviewer:
        preview.add_row("Reviewers", ", ".join(reviewer))
    if assignee:
        preview.add_row("Assignees", ", ".join(assignee))
    if label:
        preview.add_row("Labels", ", ".join(label))

    console.print(Panel(preview, title="PR Preview", border_style="cyan"))

    if title and body:
        console.print(Panel(body, title=title, border_style="magenta"))

    if not yes:
        confirm = typer.confirm("Create pull request with the details above?", default=True)
        if not confirm:
            console.print("[yellow]Pull request creation cancelled.[/yellow]")
            raise typer.Exit()

    def attempt_create(current_state: RepoState) -> dict[str, Any]:
        return GitHubCLI.create_pr(
            title=title,
            body=body,
            base=base,
            head=head,
            draft=draft,
            fill=fill,
            repo=repo,
            reviewers=tuple(reviewer),
            assignees=tuple(assignee),
            labels=tuple(label),
        )

    try:
        result = attempt_create(state)
    except sp.CalledProcessError as exc:
        message = (exc.stderr or exc.output or "").lower()
        if "must first push" in message:
            console.print("[yellow]Branch not pushed. Attempting to push before retrying...[/yellow]")
            push_branch_if_needed(state)
            try:
                state = inspect_repository(base)
            except RepositoryStateError as state_exc:
                console.print(f"[red]Error:[/red] Unable to inspect repository after push: {state_exc}")
                raise typer.Exit(1)
            try:
                result = attempt_create(state)
            except sp.CalledProcessError as retry_exc:
                console.print("[red]Failed to create pull request via gh CLI.[/red]")
                console.print((retry_exc.stderr or retry_exc.output or "").strip())
                raise typer.Exit(1)
        else:
            console.print("[red]Failed to create pull request via gh CLI.[/red]")
            console.print((exc.stderr or exc.output or "").strip())
            raise typer.Exit(1)

    pr_url = result.get("url")
    pr_number = result.get("number")

    if not pr_number and pr_url:
        pr_number = extract_pr_number(pr_url)

    if not pr_url and result.get("output"):
        pr_url = result["output"].splitlines()[-1]
        if not pr_number:
            pr_number = extract_pr_number(result["output"])

    message_lines = ["[green]✓ Pull request created successfully![/green]"]
    if pr_url:
        message_lines.append(f"[cyan]{pr_url}[/cyan]")
    if draft:
        message_lines.append("[yellow]Created as draft.[/yellow]")

    console.print(Panel("\n".join(message_lines), title="PR Created", border_style="green"))

    if run_review:
        if pr_number is None:
            console.print(
                "[red]Unable to determine PR number for review. Skipping automated review.[/red]"
            )
            return
        focus_label = focus.value if focus else "auto"
        console.print(
            f"[cyan]Running AI review for PR #{pr_number} using {resolved_model} ({focus_label}).[/cyan]"
        )
        review(
            pr_number=pr_number,
            repo=repo,
            focus=focus,
            model=resolved_model,
            post=post,
            show_diff=show_diff,
            summary_only=summary_only,
        )


@app.command()
def review(
    pr_number: int = typer.Argument(..., help="PR number to review"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository (owner/name)"),
    focus: Optional[ReviewFocus] = typer.Option(
        None,
        "--focus",
        "-f",
        help="Force a specific focus area; omit to let AI prioritize multiple areas.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=f"LLM model to use (default: {DEFAULT_MODEL_DISPLAY})",
    ),
    post: bool = typer.Option(False, "--post", "-p", help="Post review as GitHub comment"),
    show_diff: bool = typer.Option(False, "--show-diff", help="Show the diff before review"),
    max_focus_areas: int = typer.Option(
        3,
        "--max-focus",
        help="Maximum AI-selected focus areas when --focus is not provided.",
    ),
    suggest_actions: bool = typer.Option(
        True,
        "--suggest-actions/--no-suggest-actions",
        help="Generate follow-up action suggestions when AI selects focus areas.",
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary-only",
        "-s",
        help="Only display the summary section(s) of the AI review output.",
    ),
) -> None:
    """Review a GitHub pull request using AI assistance.

    When ``--focus`` is supplied, a single targeted review is generated.
    Otherwise, the AI selects up to ``max_focus`` focus areas and produces an
    aggregated review, optionally suggesting follow-up actions. Use
    ``--summary-only`` for a condensed output suitable for status updates.
    """
    resolved_model = resolve_model_selection(model)

    warn_if_gh_missing()

    if not GitHubCLI.check_auth():
        console.print("[red]Error:[/red] gh CLI not found or not authenticated")
        console.print("Install: https://cli.github.com")
        console.print("Then run: gh auth login")
        raise typer.Exit(1)

    final_review_text = ""
    pr: Optional[PullRequest] = None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching PR information...", total=None)
            pr_data = GitHubCLI.get_pr_info(pr_number, repo)

            pr = PullRequest(
                number=pr_data["number"],
                title=pr_data["title"],
                author=pr_data["author"]["login"],
                base_branch=pr_data["baseRefName"],
                head_branch=pr_data["headRefName"],
                repo=repo or "current",
                additions=pr_data.get("additions", 0),
                deletions=pr_data.get("deletions", 0),
                changed_files=pr_data.get("changedFiles", 0),
                draft=pr_data.get("isDraft", False),
            )

            progress.update(task, description="Fetching PR diff...")
            diff = GitHubCLI.get_pr_diff(pr_number, repo)

            if show_diff:
                preview = diff[:2000] + ("..." if len(diff) > 2000 else "")
                console.print(Panel(preview, title="PR Diff Preview", border_style="dim"))

            if focus:
                selected_focuses = [focus]
                console.print(f"[cyan]Using requested focus area:[/cyan] {focus.value}")
            else:
                progress.update(task, description="Analyzing diff for focus areas...")
                selected_focuses = analyze_diff_for_focus_areas(diff, resolved_model)
                selected_focuses = selected_focuses[: max(1, max_focus_areas)]
                if not selected_focuses:
                    selected_focuses = [ReviewFocus.GENERAL]
                console.print(
                    f"[cyan]AI-determined focus areas:[/cyan] {', '.join(area.value for area in selected_focuses)}"
                )

            progress.update(task, description="Loading model...")
            try:
                ai_model = llm.get_model(resolved_model)
            except llm.UnknownModelError:
                console.print(f"[red]Error:[/red] Unknown model '{resolved_model}'")
                console.print("Available models: " + ", ".join([m.model_id for m in llm.get_models()]))
                raise typer.Exit(1)
            except Exception as exc:  # pragma: no cover - unexpected LLM backend failure
                console.print(f"[red]Error:[/red] Failed to load model '{resolved_model}': {exc}")
                raise typer.Exit(1)

            all_reviews: list[tuple[ReviewFocus, str]] = []
            for index, focus_area in enumerate(selected_focuses, start=1):
                progress.update(
                    task,
                    description=f"Generating {focus_area.value} review ({index}/{len(selected_focuses)})...",
                )
                prompt = build_review_prompt(pr, diff, focus_area)
                try:
                    with console.status(
                        f"[cyan]Generating {focus_area.value} insights with {resolved_model}...[/cyan]",
                        spinner="dots",
                    ):
                        response = ai_model.prompt(prompt)
                    review_output = response.text().strip()
                    if review_output:
                        all_reviews.append((focus_area, review_output))
                except Exception as exc:  # pragma: no cover - LLM runtime failure
                    console.print(
                        f"[yellow]Warning:[/yellow] Failed to generate {focus_area.value} review: {exc}"
                    )
                    continue

            if not all_reviews:
                final_review_text = (
                    "# 🤖 AI Review\n\n"
                    "The AI was unable to generate review feedback for this pull request."
                )
            else:
                if summary_only:
                    header_title = "# 🤖 AI Review Summary"
                else:
                    header_title = "# 🤖 AI Review" if len(all_reviews) == 1 else "# 🤖 Autonomous AI Review"
                meta_lines = [
                    header_title,
                    f"**PR #{pr.number}:** {pr.title}",
                    f"**Author:** {pr.author} | **Size:** {pr.size_category}",
                    f"**Branch:** {pr.head_branch} → {pr.base_branch}",
                ]
                if len(all_reviews) == 1:
                    meta_lines.append(f"**Focus Area:** {all_reviews[0][0].value}")
                else:
                    meta_lines.append(
                        f"**Focus Areas Reviewed:** {', '.join(area.value for area, _ in all_reviews)}"
                    )

                final_sections = ["\n".join(meta_lines)]

                if summary_only:
                    summary_sections = []
                    for area, review_text in all_reviews:
                        summary = extract_summary_section(review_text)
                        if summary:
                            summary_sections.append(
                                f"## {area.value.title()} Summary\n\n{summary}"
                            )
                        else:
                            summary_sections.append(
                                f"## {area.value.title()} Summary\n\n(No summary available.)"
                            )
                    final_sections.extend(summary_sections)
                else:
                    review_sections = [
                        f"## {area.value.title()} Review\n\n{review_text.strip()}"
                        for area, review_text in all_reviews
                    ]
                    final_sections.extend(review_sections)

                if suggest_actions and not focus and not summary_only:
                    combined_feedback = "\n\n".join(text for _, text in all_reviews)
                    suggestions = suggest_followup_actions(combined_feedback, pr, resolved_model)
                    if suggestions:
                        suggestion_lines = [
                            "## 🤖 Suggested Follow-up Actions",
                            *(
                                f"{idx}. {item}"
                                for idx, item in enumerate(suggestions, 1)
                            ),
                        ]
                        final_sections.append("\n".join(suggestion_lines))

                final_review_text = "\n\n---\n\n".join(final_sections)

    except sp.CalledProcessError as exc:
        console.print(f"[red]Error:[/red] Failed to fetch PR: {exc}")
        raise typer.Exit(1)

    if not final_review_text:
        final_review_text = (
            "# 🤖 AI Review\n\n"
            "No review content was generated. Try rerunning with a different focus."
        )

    if pr is not None:
        panel_text = (
            f"[bold]PR #{pr.number}:[/bold] {pr.title}\n"
            f"[dim]by {pr.author} | {pr.size_category} ({pr.changed_files} files)[/dim]"
        )
    else:
        panel_text = f"[bold]PR #{pr_number}[/bold]"

    console.print(Panel.fit(panel_text, border_style="cyan"))
    console.print(Markdown(final_review_text))

    if post:
        console.print("\n[yellow]Posting review to GitHub...[/yellow]")
        descriptor_parts = []
        if not focus:
            descriptor_parts.append("auto-focus")
        if summary_only:
            descriptor_parts.append("summary-only")
        descriptor = f" ({', '.join(descriptor_parts)})" if descriptor_parts else ""
        comment_body = (
            f"{final_review_text}\n---\n"
            f"*Generated by pr-review using {resolved_model}{descriptor}*"
        )
        try:
            GitHubCLI.post_review_comment(pr_number, comment_body, repo)
            console.print("[green]✓ Review posted successfully![/green]")
        except sp.CalledProcessError as exc:
            console.print(f"[red]Error posting review:[/red] {exc}")



@app.command()
def check(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository (owner/name)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of PRs to show"),
) -> None:
    """Check for pull requests needing review.

    Parameters
    ----------
    repo : Optional[str], optional
        Repository (owner/name).
    limit : int, optional
        Number of PRs to show.
    """
    
    if not GitHubCLI.check_auth():
        console.print("[red]Error:[/red] gh CLI not found or not authenticated")
        raise typer.Exit(1)
    
    try:
        prs = GitHubCLI.list_prs_to_review(repo, limit)
        
        if not prs:
            console.print("[green]No pull requests currently requesting your review.[/green]")
            return
        
        table = format_pr_table(prs)
        console.print(table)
        
        console.print(f"\n[dim]Found {len(prs)} pull request(s) requesting your review.[/dim]")
        console.print("[dim]Run [cyan]./pr_review.py review <number>[/cyan] to start an AI review.[/dim]")
        
    except sp.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Failed to fetch PRs: {e}")
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """List available AI models.

    Displays a table of installed LLM models, their providers, and structured output support.
    """
    models = llm.get_models()
    if not models:
        console.print("[yellow]No models found. Install LLM plugins first.[/yellow]")
        console.print("Example: uv tool install llm-gemini")
        return
    
    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Structured Output", style="white")
    
    for model in models:
        provider = getattr(model, "provider", None) or (
            model.model_id.split("-", 1)[0] if "-" in model.model_id else "unknown"
        )
        supports_structured = model_supports_structured_output(model)
        structured_label = "[green]Yes[/green]" if supports_structured else "[red]No[/red]"
        table.add_row(model.model_id, str(provider), structured_label)
    
    console.print(table)


@app.command()
def configure() -> None:
    """Configure the default LLM model."""

    current_model = get_default_model()

    console.print("\n[bold]Configure Default Model[/bold]")
    console.print(f"The current default model is: [cyan]{current_model}[/cyan]")

    new_model = typer.prompt("Enter new default model name", default=current_model).strip()

    if new_model:
        config = load_config()
        config["model"] = new_model
        save_config(config)
        console.print(f"[green]✓ Default model updated to: [bold]{new_model}[/bold][/green]")
    else:
        console.print("[yellow]No changes made.[/yellow]")


if __name__ == "__main__":
    app()
