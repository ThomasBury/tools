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
- AI-assisted code reviews with focus areas (security, performance, etc.)
- Integration with GitHub's review system for posting comments
- Support for multiple LLM models via the llm library

Usage
-----
Basic usage examples:

.. code-block:: bash

    ./pr_review.py prepare --describe  # Inspect branch and draft PR description
    ./pr_review.py create --title "Add feature" --body "..."  # Create PR
    ./pr_review.py review 123  # Review PR #123
    ./pr_review.py check  # List PRs needing review
"""

from __future__ import annotations

import json
import re
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

# Default model for reviews
DEFAULT_MODEL = "gemini-2.5-flash-lite"


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

    refreshed_state = inspect_repository(selected)
    if refreshed_state is None:
        console.print(
            f"[yellow]Falling back to current branch comparison; unable to inspect '{selected}'.[/yellow]"
        )
        refreshed_state = initial
    return selected, refreshed_state


def get_branch_remote(branch: str) -> str:
    """Get the remote associated with a branch.

    Parameters
    ----------
    branch : str
        Name of the branch.

    Returns
    -------
    str
        Remote name for the branch, defaults to "origin" if not configured.
    """
    remote = git_output(["config", f"branch.{branch}.remote"])
    return remote or "origin"


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

    remote = get_branch_remote(state.branch)
    if upstream_missing:
        push_cmd = ["git", "push", "-u", remote, state.branch]
    else:
        push_cmd = ["git", "push"]

    console.print(f"[cyan]Running: {' '.join(push_cmd)}[/cyan]")
    try:
        sp.run(push_cmd, check=True)
    except sp.CalledProcessError as exc:
        console.print(f"[red]Failed to push branch:[/red]\n{exc.stderr or exc.stdout}")
        raise typer.Exit(1)

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
    if explicit_base:
        base_branch = explicit_base
        diff_ref = explicit_base
    elif upstream:
        if "/" in upstream:
            remote, branch = upstream.split("/", 1)
            base_branch = branch
            diff_ref = f"{remote}/{branch}"
        else:
            base_branch = upstream
            diff_ref = upstream
    else:
        base_branch = "main"
        diff_ref = "origin/main"

    return base_branch, diff_ref, upstream


def inspect_repository(base: Optional[str] = None) -> Optional[RepoState]:
    """Collect repository status details for PR preparation.

    Parameters
    ----------
    base : Optional[str], optional
        Base branch to compare against. If None, uses upstream or defaults.

    Returns
    -------
    Optional[RepoState]
        Repository state snapshot, or None if not in a git repository.
    """

    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is None:
        return None

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


def get_diff(diff_ref: str) -> str:
    """Return diff between base reference and HEAD.

    Parameters
    ----------
    diff_ref : str
        Base reference for comparison (e.g., branch name or commit).

    Returns
    -------
    str
        Git diff output, or empty string if no diff available.
    """

    diff_output = git_output(["diff", f"{diff_ref}...HEAD"], strip=False)
    if diff_output is not None:
        return diff_output
    staged_output = git_output(["diff", "--staged"], strip=False)
    return staged_output or ""


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
        response = model.prompt(prompt)
        raw = response.text().strip()
    except Exception as exc:  # pragma: no cover - LLM failures
        console.print(f"[red]LLM error while generating PR description:[/red] {exc}")
        return PRDescription(
            title="chore: update",
            body="## Summary\n- Description generation failed.\n\n## Testing\n- Not specified\n",
        )

    payload_text = raw
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        payload_text = raw[start:end]

    try:
        data = json.loads(payload_text)
        title = data.get("title") or "chore: update"
        body = data.get("body") or "## Summary\n- Description unavailable\n"
        return PRDescription(title=title.strip(), body=body.strip())
    except json.JSONDecodeError:
        console.print("[yellow]Warning: LLM returned invalid JSON. Using raw text as body.[/yellow]")
        return PRDescription(title="chore: update", body=raw)


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
        cmd = ["gh", "pr", "list", "--json",
               "number,title,author,createdAt,isDraft", "--limit", str(limit)]
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
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="LLM model to use for description generation",
    ),
) -> None:
    """Inspect local repository state before creating a PR.

    Parameters
    ----------
    base : Optional[str], optional
        Base branch to compare against.
    describe : bool, optional
        Generate AI-assisted PR title and body.
    model : str, optional
        LLM model for description generation.
    """

    state = inspect_repository(base)
    if state is None:
        console.print("[red]Error:[/red] Not inside a git repository or HEAD is detached.")
        raise typer.Exit(1)

    display_repo_state(state)

    if describe:
        diff = get_diff(state.diff_ref)
        description = generate_pr_description(diff, model)
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
    focus: ReviewFocus = typer.Option(
        ReviewFocus.GENERAL,
        "--focus",
        help="Review focus area when --run-review is used",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Model used for AI generation",
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
    model : str, optional
        Model for AI generation.
    """

    if not GitHubCLI.check_auth():
        console.print("[red]Error:[/red] gh CLI not found or not authenticated")
        console.print("Install: https://cli.github.com")
        console.print("Then run: gh auth login")
        raise typer.Exit(1)

    if fill and describe:
        console.print("[red]Error:[/red] --fill cannot be combined with --describe.")
        raise typer.Exit(1)

    if fill and (title or body):
        console.print("[red]Error:[/red] --fill cannot be combined with explicit title/body")
        raise typer.Exit(1)

    state = inspect_repository(base)
    if state is None:
        console.print("[red]Error:[/red] Not inside a git repository or HEAD is detached.")
        raise typer.Exit(1)

    if base is None:
        base, state = prompt_for_base_branch(state)
    else:
        base = base.strip()
        if base and base != state.base_branch:
            refreshed = inspect_repository(base)
            if refreshed:
                state = refreshed

    display_repo_state(state)

    push_branch_if_needed(state)
    state = inspect_repository(base)
    if state is None:
        console.print("[red]Error:[/red] Unable to inspect repository after push.")
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

    diff = get_diff(state.diff_ref)

    use_ai = describe or (not fill and (title is None or body is None))
    generated = None
    if use_ai:
        console.print(
            f"[cyan]Generating PR title and body with {model}...[/cyan]"
        )
        generated = generate_pr_description(diff, model)
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
            state = inspect_repository(base)
            if state is None:
                console.print("[red]Error:[/red] Unable to inspect repository after push.")
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
        console.print(
            f"[cyan]Running AI review for PR #{pr_number} using {model} ({focus.value}).[/cyan]"
        )
        review(
            pr_number=pr_number,
            repo=repo,
            focus=focus,
            model=model,
            post=post,
            show_diff=show_diff,
        )


@app.command()
def review(
    pr_number: int = typer.Argument(..., help="PR number to review"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository (owner/name)"),
    focus: ReviewFocus = typer.Option(ReviewFocus.GENERAL, "--focus", "-f", help="Review focus area"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    post: bool = typer.Option(False, "--post", "-p", help="Post review as GitHub comment"),
    show_diff: bool = typer.Option(False, "--show-diff", help="Show the diff before review"),
) -> None:
    """Review a GitHub pull request using AI.

    Parameters
    ----------
    pr_number : int
        PR number to review.
    repo : Optional[str], optional
        Repository (owner/name).
    focus : ReviewFocus, optional
        Review focus area.
    model : str, optional
        LLM model to use.
    post : bool, optional
        Post review as GitHub comment.
    show_diff : bool, optional
        Show diff before review.
    """
    
    # Check gh CLI authentication
    if not GitHubCLI.check_auth():
        console.print("[red]Error:[/red] gh CLI not found or not authenticated")
        console.print("Install: https://cli.github.com")
        console.print("Then run: gh auth login")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Fetch PR information
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
                console.print(Panel(diff[:2000] + ("..." if len(diff) > 2000 else ""), 
                                  title="PR Diff Preview", border_style="dim"))
            
            progress.update(task, description="Generating AI review...")
            prompt = build_review_prompt(pr, diff, focus)
            
            # Get the model and generate review
            try:
                ai_model = llm.get_model(model)
            except llm.UnknownModelError:
                console.print(f"[red]Error:[/red] Unknown model '{model}'")
                console.print("Available models: " + ", ".join([m.model_id for m in llm.get_models()]))
                raise typer.Exit(1)
            
            response = ai_model.prompt(prompt)
            review_text = response.text()
    
    except sp.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Failed to fetch PR: {e}")
        raise typer.Exit(1)
    
    # Display the review
    console.print(Panel.fit(
        f"[bold]PR #{pr.number}:[/bold] {pr.title}\n"
        f"[dim]by {pr.author} | {pr.size_category} ({pr.changed_files} files)[/dim]",
        border_style="cyan"
    ))
    
    console.print(Markdown(review_text))
    
    # Optionally post to GitHub
    if post:
        console.print("\n[yellow]Posting review to GitHub...[/yellow]")
        comment_body = f"## 🤖 AI Review (Focus: {focus.value})\n\n{review_text}\n\n---\n*Generated by pr-review using {model}*"
        try:
            GitHubCLI.post_review_comment(pr_number, comment_body, repo)
            console.print("[green]✓ Review posted successfully![/green]")
        except sp.CalledProcessError as e:
            console.print(f"[red]Error posting review:[/red] {e}")


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
            console.print("[green]No open pull requests found.[/green]")
            return
        
        table = format_pr_table(prs)
        console.print(table)
        
        console.print(f"\n[dim]Found {len(prs)} open PR(s)")
        console.print("Use [cyan]pr-review review <number>[/cyan] to review a specific PR[/dim]")
        
    except sp.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Failed to fetch PRs: {e}")
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """List available AI models.

    Displays a table of installed LLM models and their providers.
    """
    models = llm.get_models()
    if not models:
        console.print("[yellow]No models found. Install LLM plugins first.[/yellow]")
        console.print("Example: uv tool install llm-gemini")
        return
    
    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="yellow")
    
    for model in models:
        provider = model.model_id.split("-")[0] if "-" in model.model_id else "unknown"
        table.add_row(model.model_id, provider)
    
    console.print(table)


if __name__ == "__main__":
    app()
