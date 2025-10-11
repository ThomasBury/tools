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
pr-review: AI-powered GitHub PR reviewer using gh CLI.

Features:
- Pre-flight PR preparation with best-practice checks and AI-crafted descriptions
- Creates pull requests via the GitHub CLI (gh must be installed and authenticated)
- Fetches PR diffs using gh CLI for AI-assisted reviews
- Provides intelligent code review with AI (supports multiple models via llm)
- Checks for common issues, security concerns, and suggests improvements
- Integrates with GitHub's review system to post comments

Usage:
    ./pr_review.py prepare --describe  # Inspect branch and draft a PR description
    ./pr_review.py create --title "Add feature" --body "..."  # Create a PR using gh
    ./pr_review.py review 123  # Review PR #123 in current repo
    ./pr_review.py review owner/repo 123  # Review PR in specific repo
    ./pr_review.py check  # Check recent PRs needing review
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
    """Run a git command and return its stdout."""

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


def resolve_base_reference(explicit_base: Optional[str]) -> tuple[str, str, Optional[str]]:
    """Determine the base branch and diff reference for comparisons."""

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
    """Collect repository status details for PR preparation."""

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
    """Return diff between base reference and HEAD."""

    diff_output = git_output(["diff", f"{diff_ref}...HEAD"], strip=False)
    if diff_output is not None:
        return diff_output
    staged_output = git_output(["diff", "--staged"], strip=False)
    return staged_output or ""


def generate_pr_description(diff: str, model_name: str) -> PRDescription:
    """Use an LLM to craft a PR title and body."""

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
    """Render repository insights and best-practice checks."""

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
    """Types of review focus."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTS = "tests"
    DOCS = "docs"
    GENERAL = "general"


@dataclass
class PullRequest:
    """Pull request information."""
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
        """Categorize PR size."""
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
    """Auto-generated pull request description."""

    title: str
    body: str


@dataclass
class RepoState:
    """Snapshot of local repository state relevant to PR creation."""

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
    """Wrapper for GitHub CLI operations."""
    
    @staticmethod
    def check_auth() -> bool:
        """Check if gh CLI is authenticated."""
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
        """Create a pull request using gh CLI and return metadata."""

        cmd = ["gh", "pr", "create", "--json", "number,url"]

        if repo:
            cmd.extend(["--repo", repo])
        if base:
            cmd.extend(["--base", base])
        if head:
            cmd.extend(["--head", head])
        if title:
            cmd.extend(["--title", title])
        if body:
            cmd.extend(["--body", body])
        if draft:
            cmd.append("--draft")
        if fill:
            cmd.append("--fill")
        for reviewer in reviewers:
            cmd.extend(["--reviewer", reviewer])
        for assignee in assignees:
            cmd.extend(["--assignee", assignee])
        for label in labels:
            cmd.extend(["--label", label])

        result = sp.run(cmd, capture_output=True, text=True, check=False)

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
        """Fetch PR information."""
        cmd = ["gh", "pr", "view", str(pr_number), "--json",
               "number,title,author,baseRefName,headRefName,additions,deletions,changedFiles,isDraft"]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    @staticmethod
    def get_pr_diff(pr_number: int, repo: Optional[str] = None) -> str:
        """Fetch PR diff."""
        cmd = ["gh", "pr", "diff", str(pr_number)]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    
    @staticmethod
    def get_pr_files(pr_number: int, repo: Optional[str] = None) -> list[str]:
        """Get list of files changed in PR."""
        cmd = ["gh", "pr", "view", str(pr_number), "--json", "files"]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return [f["path"] for f in data.get("files", [])]
    
    @staticmethod
    def list_prs_to_review(repo: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
        """List PRs that need review."""
        cmd = ["gh", "pr", "list", "--json",
               "number,title,author,createdAt,isDraft", "--limit", str(limit)]
        if repo:
            cmd.extend(["--repo", repo])
        
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    @staticmethod
    def post_review_comment(pr_number: int, body: str, repo: Optional[str] = None) -> None:
        """Post a review comment to PR."""
        cmd = ["gh", "pr", "comment", str(pr_number), "--body", body]
        if repo:
            cmd.extend(["--repo", repo])
        
        sp.run(cmd, check=True)


def extract_pr_number(reference: str) -> Optional[int]:
    """Extract a PR number from a string such as a URL."""
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
    """Build AI review prompt based on PR and focus area."""
    
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
    """Format PRs as a rich table."""
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
    """Inspect local repository state before creating a PR."""

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
    """Create a pull request using the GitHub CLI."""

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

    display_repo_state(state)

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

    try:
        result = GitHubCLI.create_pr(
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
    except sp.CalledProcessError as exc:
        console.print("[red]Failed to create pull request via gh CLI.[/red]")
        if exc.stderr:
            console.print(exc.stderr.strip())
        elif exc.output:
            console.print(exc.output.strip())
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
    """Review a GitHub pull request using AI."""
    
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
    """Check for pull requests needing review."""
    
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
    """List available AI models."""
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
