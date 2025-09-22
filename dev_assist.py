#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "tomli>=2.0; python_version < '3.11'",
#   "tomli-w>=1.0",
#   "gitpython>=3.1",
# ]
# ///
"""
dev-assist: Smart development assistant for Python projects.

Automates common development workflows:
- Project initialization with modern Python tooling (uv)
- Dependency management and updates
- Git operations and conventional commits
- Test running with coverage
- Pre-commit setup and checks
- Quick fixes for common issues

Usage:
    ./dev_assist.py init my-project  # Create new project
    ./dev_assist.py deps add httpx   # Add dependency
    ./dev_assist.py test             # Run tests with coverage
    ./dev_assist.py commit feat      # Interactive commit helper
"""

from __future__ import annotations

import os
import subprocess as sp
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
from git import Repo
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import tomli_w

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


class CommitType(str, Enum):
    """Conventional commit types."""
    FEAT = "feat"       # New feature
    FIX = "fix"         # Bug fix
    DOCS = "docs"       # Documentation only
    STYLE = "style"     # Code style (formatting, semicolons, etc)
    REFACTOR = "refactor"  # Code refactoring
    PERF = "perf"       # Performance improvements
    TEST = "test"       # Adding tests
    BUILD = "build"     # Build system changes
    CI = "ci"           # CI/CD changes
    CHORE = "chore"     # Other changes


@dataclass
class ProjectInfo:
    """Project information."""
    name: str
    path: Path
    has_git: bool
    has_pyproject: bool
    has_tests: bool
    python_version: str
    dependencies: list[str]
    dev_dependencies: list[str]


class UVManager:
    """Manage uv operations."""
    
    @staticmethod
    def check_installed() -> bool:
        """Check if uv is installed."""
        try:
            sp.run(["uv", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, sp.CalledProcessError):
            return False
    
    @staticmethod
    def init_project(name: str, python_version: Optional[str] = None) -> Path:
        """Initialize a new uv project."""
        cmd = ["uv", "init", name]
        if python_version:
            cmd.extend(["--python", python_version])
        
        sp.run(cmd, check=True)
        return Path(name)
    
    @staticmethod
    def add_dependency(package: str, dev: bool = False, extras: Optional[list[str]] = None) -> None:
        """Add a dependency to the project."""
        cmd = ["uv", "add"]
        
        if dev:
            cmd.append("--dev")
        
        if extras:
            package = f"{package}[{','.join(extras)}]"
        
        cmd.append(package)
        sp.run(cmd, check=True)
    
    @staticmethod
    def remove_dependency(package: str, dev: bool = False) -> None:
        """Remove a dependency."""
        cmd = ["uv", "remove"]
        if dev:
            cmd.append("--dev")
        cmd.append(package)
        sp.run(cmd, check=True)
    
    @staticmethod
    def sync_dependencies() -> None:
        """Sync dependencies from pyproject.toml."""
        sp.run(["uv", "sync"], check=True)
    
    @staticmethod
    def run_command(command: list[str]) -> sp.CompletedProcess:
        """Run a command in the uv environment."""
        return sp.run(["uv", "run"] + command, capture_output=True, text=True)
    
    @staticmethod
    def update_dependencies(packages: Optional[list[str]] = None) -> None:
        """Update dependencies."""
        cmd = ["uv", "lock", "--upgrade"]
        if packages:
            for pkg in packages:
                cmd.extend(["--upgrade-package", pkg])
        sp.run(cmd, check=True)
        sp.run(["uv", "sync"], check=True)


class GitManager:
    """Manage git operations."""
    
    @staticmethod
    def init_repo(path: Path) -> Repo:
        """Initialize a git repository."""
        return Repo.init(path)
    
    @staticmethod
    def get_repo(path: Path = Path.cwd()) -> Optional[Repo]:
        """Get existing repo or None."""
        try:
            return Repo(path, search_parent_directories=True)
        except:
            return None
    
    @staticmethod
    def stage_files(repo: Repo, patterns: list[str]) -> list[str]:
        """Stage files matching patterns."""
        staged = []
        for pattern in patterns:
            if pattern == ".":
                repo.git.add(".")
                staged.append("all files")
            else:
                matching = list(Path(repo.working_dir).glob(pattern))
                if matching:
                    repo.index.add([str(f) for f in matching])
                    staged.extend([str(f) for f in matching])
        return staged
    
    @staticmethod
    def commit(repo: Repo, message: str) -> str:
        """Create a commit."""
        return repo.index.commit(message).hexsha
    
    @staticmethod
    def get_status(repo: Repo) -> dict[str, list[str]]:
        """Get repository status."""
        status = {
            "modified": [],
            "added": [],
            "deleted": [],
            "untracked": []
        }
        
        for item in repo.index.diff(None):
            status["modified"].append(item.a_path)
        
        for item in repo.index.diff("HEAD"):
            if item.change_type == "A":
                status["added"].append(item.a_path)
            elif item.change_type == "D":
                status["deleted"].append(item.a_path)
        
        status["untracked"] = repo.untracked_files
        
        return status


def get_project_info(path: Path = Path.cwd()) -> ProjectInfo:
    """Gather project information."""
    pyproject_path = path / "pyproject.toml"
    
    # Default values
    name = path.name
    dependencies = []
    dev_dependencies = []
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Parse pyproject.toml if it exists
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        
        project = data.get("project", {})
        name = project.get("name", name)
        dependencies = project.get("dependencies", [])
        
        # Get dev dependencies from various sources
        dev_group = data.get("dependency-groups", {}).get("dev", [])
        optional_dev = project.get("optional-dependencies", {}).get("dev", [])
        dev_dependencies = dev_group or optional_dev
        
        # Get Python version requirement
        requires_python = project.get("requires-python", "")
        if requires_python:
            # Simple extraction, e.g., ">=3.11" -> "3.11"
            import re
            match = re.search(r"(\d+\.\d+)", requires_python)
            if match:
                python_version = match.group(1)
    
    return ProjectInfo(
        name=name,
        path=path,
        has_git=GitManager.get_repo(path) is not None,
        has_pyproject=pyproject_path.exists(),
        has_tests=(path / "tests").exists() or (path / "test").exists(),
        python_version=python_version,
        dependencies=dependencies,
        dev_dependencies=dev_dependencies,
    )


def create_default_files(project_path: Path, project_name: str) -> None:
    """Create default project files."""
    
    # Create source directory
    src_dir = project_path / project_name.replace("-", "_")
    src_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    (src_dir / "__init__.py").write_text('"""Package initialization."""\n\n__version__ = "0.1.0"\n')
    
    # Create main.py
    main_content = '''"""Main module."""

def hello(name: str = "World") -> str:
    """Say hello.
    
    Parameters
    ----------
    name : str
        Name to greet.
    
    Returns
    -------
    str
        Greeting message.
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(hello())
'''
    (src_dir / "main.py").write_text(main_content)
    
    # Create tests directory
    tests_dir = project_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    
    # Create test file
    test_content = f'''"""Test suite for {project_name}."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from {project_name.replace("-", "_")}.main import hello


def test_hello():
    """Test hello function."""
    assert hello() == "Hello, World!"
    assert hello("Alice") == "Hello, Alice!"


if __name__ == "__main__":
    test_hello()
    print("All tests passed!")
'''
    (tests_dir / "test_main.py").write_text(test_content)
    
    # Create README
    readme_content = f"""# {project_name}

A Python project managed with uv.

## Installation

```bash
uv sync
```

## Development

Run tests:
```bash
uv run pytest
```

Run with coverage:
```bash
uv run pytest --cov
```

Format code:
```bash
uv run ruff format .
```

Lint code:
```bash
uv run ruff check . --fix
```
"""
    (project_path / "README.md").write_text(readme_content)
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
.uv/

# Testing
.coverage
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project
*.log
*.db
.env
"""
    (project_path / ".gitignore").write_text(gitignore_content)


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    python_version: Optional[str] = typer.Option(None, "--python", "-p", help="Python version (e.g., 3.11)"),
    with_git: bool = typer.Option(True, "--git/--no-git", help="Initialize git repository"),
    dev_packages: list[str] = typer.Option([], "--dev", "-d", help="Dev dependencies to add"),
) -> None:
    """Initialize a new Python project with modern tooling."""
    
    if not UVManager.check_installed():
        console.print("[red]Error:[/red] uv is not installed")
        console.print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        raise typer.Exit(1)
    
    project_path = Path(name)
    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize project with uv
        task = progress.add_task("Initializing project with uv...", total=None)
        project_path = UVManager.init_project(name, python_version)
        
        # Create default files
        progress.update(task, description="Creating project structure...")
        create_default_files(project_path, name)
        
        # Change to project directory
        os.chdir(project_path)
        
        # Add common dev dependencies
        default_dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5", "mypy>=1.10"]
        all_dev = default_dev + dev_packages
        
        for pkg in all_dev:
            progress.update(task, description=f"Adding {pkg}...")
            UVManager.add_dependency(pkg, dev=True)
        
        # Initialize git if requested
        if with_git:
            progress.update(task, description="Initializing git repository...")
            repo = GitManager.init_repo(project_path)
            repo.index.add(".")
            repo.index.commit("Initial commit")
    
    # Display success message
    console.print(Panel.fit(
        f"[green]✓[/green] Project '{name}' created successfully!\n\n"
        f"[cyan]Next steps:[/cyan]\n"
        f"  cd {name}\n"
        f"  uv run pytest       # Run tests\n"
        f"  uv run python -m {name.replace('-', '_')}.main  # Run main",
        title="Project Initialized",
        border_style="green"
    ))


@app.command()
def deps(
    action: str = typer.Argument(..., help="Action: add, remove, update, list"),
    packages: list[str] = typer.Argument(None, help="Package names"),
    dev: bool = typer.Option(False, "--dev", "-d", help="Dev dependency"),
    extras: list[str] = typer.Option([], "--extra", "-e", help="Package extras"),
) -> None:
    """Manage project dependencies."""
    
    if not UVManager.check_installed():
        console.print("[red]Error:[/red] uv is not installed")
        raise typer.Exit(1)
    
    info = get_project_info()
    if not info.has_pyproject:
        console.print("[red]Error:[/red] No pyproject.toml found")
        raise typer.Exit(1)
    
    if action == "add":
        if not packages:
            console.print("[red]Error:[/red] Specify packages to add")
            raise typer.Exit(1)
        
        for pkg in packages:
            console.print(f"Adding {pkg}...")
            UVManager.add_dependency(pkg, dev=dev, extras=extras)
        console.print("[green]✓ Dependencies added[/green]")
    
    elif action == "remove":
        if not packages:
            console.print("[red]Error:[/red] Specify packages to remove")
            raise typer.Exit(1)
        
        for pkg in packages:
            console.print(f"Removing {pkg}...")
            UVManager.remove_dependency(pkg, dev=dev)
        console.print("[green]✓ Dependencies removed[/green]")
    
    elif action == "update":
        console.print("Updating dependencies...")
        UVManager.update_dependencies(packages)
        console.print("[green]✓ Dependencies updated[/green]")
    
    elif action == "list":
        table = Table(title="Project Dependencies")
        table.add_column("Type", style="cyan")
        table.add_column("Package", style="white")
        
        for dep in info.dependencies:
            table.add_row("prod", dep)
        
        for dep in info.dev_dependencies:
            table.add_row("dev", dep)
        
        console.print(table)
    
    else:
        console.print(f"[red]Error:[/red] Unknown action '{action}'")
        console.print("Valid actions: add, remove, update, list")
        raise typer.Exit(1)


@app.command()
def test(
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Run with coverage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Specific test file"),
) -> None:
    """Run tests with optional coverage."""
    
    info = get_project_info()
    if not info.has_tests:
        console.print("[yellow]Warning:[/yellow] No tests directory found")
        if not Confirm.ask("Continue anyway?"):
            raise typer.Exit(0)
    
    cmd = ["pytest"]
    
    if coverage:
        cmd.extend(["--cov", info.name.replace("-", "_"), "--cov-report", "term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    if file:
        cmd.append(file)
    
    console.print(f"[cyan]Running:[/cyan] uv run {' '.join(cmd)}")
    result = UVManager.run_command(cmd)
    
    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr, style="red")
    
    if result.returncode != 0:
        console.print("[red]✗ Tests failed[/red]")
        raise typer.Exit(result.returncode)
    else:
        console.print("[green]✓ All tests passed[/green]")


@app.command()
def commit(
    type: CommitType = typer.Argument(..., help="Commit type"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Commit scope"),
    breaking: bool = typer.Option(False, "--breaking", "-b", help="Breaking change"),
) -> None:
    """Create a conventional commit with interactive prompts."""
    
    repo = GitManager.get_repo()
    if not repo:
        console.print("[red]Error:[/red] Not in a git repository")
        raise typer.Exit(1)
    
    # Check status
    status = GitManager.get_status(repo)
    total_changes = sum(len(files) for files in status.values())
    
    if total_changes == 0:
        console.print("[yellow]No changes to commit[/yellow]")
        raise typer.Exit(0)
    
    # Show status
    console.print("\n[bold]Repository Status:[/bold]")
    for status_type, files in status.items():
        if files:
            console.print(f"  {status_type}: {len(files)} file(s)")
            for f in files[:5]:  # Show first 5
                console.print(f"    • {f}")
            if len(files) > 5:
                console.print(f"    ... and {len(files) - 5} more")
    
    # Interactive staging
    if status["untracked"] or status["modified"]:
        if Confirm.ask("\nStage all changes?", default=True):
            repo.git.add(".")
            console.print("[green]✓ All changes staged[/green]")
        else:
            patterns = Prompt.ask("Enter patterns to stage (comma-separated)", default=".")
            staged = GitManager.stage_files(repo, patterns.split(","))
            console.print(f"[green]✓ Staged: {', '.join(staged)}[/green]")
    
    # Get commit message
    description = Prompt.ask("\nCommit description")
    
    # Build commit message
    commit_msg = f"{type.value}"
    if scope:
        commit_msg += f"({scope})"
    if breaking:
        commit_msg += "!"
    commit_msg += f": {description}"
    
    # Optional body
    if Confirm.ask("\nAdd detailed description?", default=False):
        console.print("Enter commit body (end with empty line):")
        body_lines = []
        while True:
            line = input()
            if not line:
                break
            body_lines.append(line)
        
        if body_lines:
            commit_msg += "\n\n" + "\n".join(body_lines)
    
    if breaking and Confirm.ask("\nAdd breaking change note?", default=True):
        breaking_note = Prompt.ask("Breaking change description")
        commit_msg += f"\n\nBREAKING CHANGE: {breaking_note}"
    
    # Show commit message
    console.print("\n[bold]Commit Message:[/bold]")
    console.print(Panel(commit_msg, border_style="dim"))
    
    # Confirm and commit
    if Confirm.ask("\nCreate commit?", default=True):
        sha = GitManager.commit(repo, commit_msg)
        console.print(f"[green]✓ Committed: {sha[:8]}[/green]")
    else:
        console.print("[yellow]Commit cancelled[/yellow]")


@app.command()
def info() -> None:
    """Show project information."""
    
    info = get_project_info()
    
    console.print(Panel.fit(
        f"[bold]{info.name}[/bold]\n"
        f"Path: {info.path}\n"
        f"Python: {info.python_version}\n"
        f"Git: {'✓' if info.has_git else '✗'}\n"
        f"Tests: {'✓' if info.has_tests else '✗'}\n"
        f"Dependencies: {len(info.dependencies)}\n"
        f"Dev Dependencies: {len(info.dev_dependencies)}",
        title="Project Information",
        border_style="cyan"
    ))


@app.command()
def fix(
    what: str = typer.Argument("all", help="What to fix: format, lint, types, or all"),
) -> None:
    """Quick fixes for common issues."""
    
    info = get_project_info()
    if not info.has_pyproject:
        console.print("[red]Error:[/red] No pyproject.toml found")
        raise typer.Exit(1)
    
    commands = {
        "format": ["ruff", "format", "."],
        "lint": ["ruff", "check", ".", "--fix"],
        "types": ["mypy", "."],
    }
    
    if what == "all":
        to_run = ["format", "lint", "types"]
    elif what in commands:
        to_run = [what]
    else:
        console.print(f"[red]Error:[/red] Unknown fix target '{what}'")
        console.print("Valid options: format, lint, types, all")
        raise typer.Exit(1)
    
    for target in to_run:
        console.print(f"[cyan]Running {target}...[/cyan]")
        result = UVManager.run_command(commands[target])
        
        if result.returncode != 0:
            console.print(f"[yellow]Warning:[/yellow] {target} found issues")
            if result.stdout:
                console.print(result.stdout)
        else:
            console.print(f"[green]✓ {target} completed[/green]")


if __name__ == "__main__":
    app()