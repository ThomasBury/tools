#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
#   "httpx>=0.27",
#   "pydantic>=2.0",
# ]
# ///
"""
pr-review: AI-powered GitHub PR reviewer using gh CLI.

Features:
- Fetches PR diffs using gh CLI (must be installed and authenticated)
- Provides intelligent code review with AI (supports multiple models via llm)
- Checks for common issues, security concerns, and suggests improvements
- Integrates with GitHub's review system to post comments

Usage:
    ./pr_review.py review 123  # Review PR #123 in current repo
    ./pr_review.py review owner/repo 123  # Review PR in specific repo
    ./pr_review.py check  # Check recent PRs needing review
"""

from __future__ import annotations

import json
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
DEFAULT_MODEL = "gemini-2.5-flash"

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