#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
#   "pygments>=2.17",
#   "pathspec>=0.12",
# ]
# ///
"""
code-review: Lightweight AI-powered code reviewer.

Features:
- Quick code reviews using AI (supports multiple models via llm)
- Analyzes Python files for bugs, style, security, and best practices
- Context-aware reviews using project structure
- Supports .gitignore patterns
- Beautiful terminal output with syntax highlighting

Usage:
    ./code_review.py review file.py      # Review a single file
    ./code_review.py review src/          # Review all Python files in directory
    ./code_review.py review . --focus security  # Security-focused review
"""

from __future__ import annotations

import os
import subprocess as sp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import llm
import pathspec
import typer
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Defaults
DEFAULT_MODEL = os.environ.get("CODE_REVIEW_MODEL", "gemini-2.5-flash")
MAX_FILE_SIZE = int(os.environ.get("CODE_REVIEW_MAX_SIZE", "100000"))  # 100KB


class ReviewFocus(str, Enum):
    """Types of review focus."""
    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    STYLE = "style"


@dataclass
class CodeFile:
    """Represents a code file to review."""
    path: Path
    content: str
    lines: int
    size: int

def get_gitignore_spec(root: Path) -> Optional[pathspec.PathSpec]:
    """Get gitignore patterns if available."""
    gitignore = root / ".gitignore"
    if gitignore.exists():
        with open(gitignore) as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f)
    return None


def find_python_files(path: Path, gitignore_spec: Optional[pathspec.PathSpec] = None) -> list[Path]:
    """Find all Python files in path, respecting gitignore."""
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    
    files = []
    for file in path.rglob("*.py"):
        # Skip if gitignored
        if gitignore_spec and gitignore_spec.match_file(file.relative_to(path.parent)):
            continue
        # Skip common directories
        parts = file.parts
        if any(p in {".venv", "venv", ".uv", "__pycache__", ".git", "build", "dist"} for p in parts):
            continue
        files.append(file)
    
    return sorted(files)


def read_code_file(file: Path, max_size: int = MAX_FILE_SIZE) -> Optional[CodeFile]:
    """Read a code file with size limits."""
    try:
        stat = file.stat()
        if stat.st_size > max_size:
            return None
        
        content = file.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1
        
        return CodeFile(
            path=file,
            content=content,
            lines=lines,
            size=stat.st_size
        )
    except Exception:
        return None


def get_project_context(root: Path) -> str:
    """Get project context from common files."""
    context_parts = []
    
    # Check for pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                project = data.get("project", {})
                if project:
                    context_parts.append(f"Project: {project.get('name', 'unknown')}")
                    if deps := project.get("dependencies"):
                        context_parts.append(f"Dependencies: {', '.join(deps[:5])}...")
        except:
            pass
    
    # Check for README
    for readme in ["README.md", "README.rst", "README.txt"]:
        readme_file = root / readme
        if readme_file.exists():
            content = readme_file.read_text(encoding="utf-8", errors="ignore")
            # Get first paragraph
            first_para = content.split("\n\n")[0][:200]
            context_parts.append(f"README excerpt: {first_para}")
            break
    
    return "\n".join(context_parts) if context_parts else ""

def build_review_prompt(files: list[CodeFile], focus: ReviewFocus, context: str) -> str:
    """Build review prompt for the AI."""
    
    focus_instructions = {
        ReviewFocus.GENERAL: """
Provide a balanced code review covering:
- Code quality and readability
- Potential bugs and edge cases  
- Best practices and idioms
- Performance considerations
- Security implications""",
        ReviewFocus.SECURITY: """
Focus on security vulnerabilities:
- Input validation and sanitization
- SQL injection, XSS, command injection risks
- Hardcoded secrets or credentials
- Path traversal and file access issues
- Authentication and authorization flaws
- Unsafe deserialization or eval usage""",
        ReviewFocus.PERFORMANCE: """
Focus on performance optimization:
- Algorithm complexity and efficiency
- Database query optimization (N+1 queries)
- Memory usage and leaks
- Caching opportunities
- Async/concurrent programming issues
- Unnecessary loops or computations""",
        ReviewFocus.TESTING: """
Focus on testability and test coverage:
- Missing tests for critical functions
- Edge cases not covered
- Test structure and organization
- Mock/stub usage
- Test isolation and dependencies
- Assertion quality""",
        ReviewFocus.STYLE: """
Focus on code style and conventions:
- PEP 8 compliance
- Naming conventions
- Documentation and docstrings
- Type hints completeness
- Code organization and structure
- Import organization"""
    }
    
    # Build file list
    file_list = "\n".join([
        f"- {f.path.name} ({f.lines} lines)"
        for f in files[:10]  # List first 10 files
    ])
    if len(files) > 10:
        file_list += f"\n... and {len(files) - 10} more files"
    
    # Combine code from files
    total_lines = sum(f.lines for f in files)
    code_blocks = []
    
    for file in files[:5]:  # Include up to 5 files
        if len(code_blocks) > 0 and sum(len(b) for b in code_blocks) > 10000:
            break  # Limit total size
        
        code_blocks.append(f"\n### File: {file.path}\n```python\n{file.content[:5000]}\n```")
    
    code_section = "\n".join(code_blocks)
    
    prompt = f"""You are an expert Python developer conducting a code review.

{context}

Files being reviewed:
{file_list}

Total lines: {total_lines}

Review Focus:
{focus_instructions[focus]}

Provide a structured review with these sections:
1. **Overview** - Summary of the code and its purpose
2. **Strengths** - What's done well
3. **Issues Found** - Problems discovered (be specific with file names and line numbers)
4. **Recommendations** - Actionable improvements
5. **Priority Actions** - Top 3-5 most important changes

Be constructive, specific, and reference actual code. Use markdown formatting.

Code to review:
{code_section}
"""
    
    return prompt


def format_file_stats(files: list[CodeFile]) -> Table:
    """Create a table of files to review."""
    table = Table(title="Files to Review")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right", style="yellow")
    table.add_column("Size", justify="right", style="dim")
    
    total_lines = 0
    total_size = 0
    
    for file in files:
        table.add_row(
            str(file.path.name),
            str(file.lines),
            f"{file.size:,} bytes"
        )
        total_lines += file.lines
        total_size += file.size
    
    if len(files) > 1:
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_lines}[/bold]",
            f"[bold]{total_size:,} bytes[/bold]",
            style="green"
        )
    
    return table


def check_git_status(path: Path) -> Optional[str]:
    """Check if path is in a git repo and get status."""
    try:
        # Find git root
        result = sp.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path if path.is_dir() else path.parent,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return None
        
        git_root = Path(result.stdout.strip())
        
        # Get brief status
        result = sp.run(
            ["git", "status", "--short"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split("\n")
            return f"Git repository with {len(lines)} uncommitted changes"
        elif result.returncode == 0:
            return "Git repository (clean)"
    except:
        pass
    
    return None


@app.command()
def review(
    path: Path = typer.Argument(..., help="File or directory to review"),
    focus: ReviewFocus = typer.Option(ReviewFocus.GENERAL, "--focus", "-f", help="Review focus area"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    max_files: int = typer.Option(10, "--max-files", help="Maximum files to review"),
    show_code: bool = typer.Option(False, "--show-code", "-s", help="Show code snippets in terminal"),
) -> None:
    """Review Python code with AI assistance."""
    
    # Validate path
    if not path.exists():
        console.print(f"[red]Error:[/red] Path '{path}' does not exist")
        raise typer.Exit(1)
    
    # Find Python files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding Python files...", total=None)
        
        # Get gitignore spec if in git repo
        root = path if path.is_dir() else path.parent
        gitignore_spec = get_gitignore_spec(root)
        
        # Find files
        files = find_python_files(path, gitignore_spec)
        
        if not files:
            console.print("[yellow]No Python files found to review[/yellow]")
            raise typer.Exit(0)
        
        if len(files) > max_files:
            console.print(f"[yellow]Found {len(files)} files, reviewing first {max_files}[/yellow]")
            files = files[:max_files]
        
        # Read files
        progress.update(task, description="Reading files...")
        code_files = []
        skipped = 0
        
        for file in files:
            if code_file := read_code_file(file):
                code_files.append(code_file)
            else:
                skipped += 1
        
        if not code_files:
            console.print("[red]Error:[/red] No files could be read")
            raise typer.Exit(1)
        
        if skipped > 0:
            console.print(f"[dim]Skipped {skipped} file(s) due to size limits[/dim]")
        
        # Get project context
        progress.update(task, description="Analyzing project context...")
        context = get_project_context(root)
        
        # Check git status
        git_status = check_git_status(path)
        if git_status:
            context = f"{context}\n{git_status}" if context else git_status
    
    # Display file stats
    console.print(format_file_stats(code_files))
    
    # Show code preview if requested
    if show_code and code_files:
        lexer = PythonLexer()
        formatter = TerminalFormatter()
        
        for file in code_files[:2]:  # Show first 2 files
            console.print(f"\n[cyan]Preview: {file.path.name}[/cyan]")
            preview = file.content[:500] + ("\n..." if len(file.content) > 500 else "")
            highlighted = highlight(preview, lexer, formatter)
            console.print(Panel(highlighted, border_style="dim"))
    
    # Build prompt
    console.print(f"\n[cyan]Generating {focus.value} review with {model}...[/cyan]")
    prompt = build_review_prompt(code_files, focus, context)
    
    # Get AI model
    try:
        ai_model = llm.get_model(model)
    except llm.UnknownModelError:
        console.print(f"[red]Error:[/red] Unknown model '{model}'")
        console.print("Available models: " + ", ".join([m.model_id for m in llm.get_models()]))
        raise typer.Exit(1)
    
    # Generate review
    try:
        response = ai_model.prompt(prompt)
        review_text = response.text()
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to generate review: {e}")
        raise typer.Exit(1)
    
    # Display review
    console.print(Panel("[bold green]Code Review Complete[/bold green]", border_style="green"))
    console.print(Markdown(review_text))

@app.command()
def quick(
    path: Path = typer.Argument(..., help="File to quickly review"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model"),
) -> None:
    """Quick review of a single file with minimal output."""
    
    if not path.exists() or not path.is_file():
        console.print(f"[red]Error:[/red] '{path}' is not a valid file")
        raise typer.Exit(1)
    
    if path.suffix != ".py":
        console.print("[yellow]Warning:[/yellow] File is not a Python file")
    
    # Read file
    code_file = read_code_file(path)
    if not code_file:
        console.print("[red]Error:[/red] File is too large or cannot be read")
        raise typer.Exit(1)
    
    # Simple prompt for quick review
    prompt = f"""Quickly review this Python code. Focus on:
1. Critical bugs or security issues
2. Major performance problems
3. Serious code quality issues

Be brief and actionable. List only important issues.

File: {path.name}
```python
{code_file.content}
```
"""
    
    console.print(f"[cyan]Quick review of {path.name}...[/cyan]")
    
    try:
        model = llm.get_model(model)
        response = model.prompt(prompt)
        
        console.print(Panel(
            Markdown(response.text()),
            title=f"Quick Review: {path.name}",
            border_style="yellow"
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
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
