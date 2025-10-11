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
Smart development assistant for Python projects.

This module provides a command-line interface for automating common Python
development workflows. It integrates with modern tools like uv for dependency
management, git for version control, and various quality assurance tools.

Features
--------
- Project initialization with modern Python tooling (uv)
- Dependency management and updates
- Git operations and conventional commits
- Test running with coverage analysis
- Pre-commit setup and automated checks
- Quick fixes for common code quality issues

Notes
-----
The tool is designed to work with uv-managed Python projects and follows
modern Python development best practices. All commands are implemented as
Typer applications with Rich-formatted output for better user experience.

Examples
--------
Initialize a new project:
    ./dev_assist.py init my-project

Add a dependency:
    ./dev_assist.py deps add httpx

Run tests with coverage:
    ./dev_assist.py test

Create an interactive conventional commit:
    ./dev_assist.py commit feat
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
    """Enumeration of conventional commit types.

    This enum defines the standard commit types used in conventional commits,
    following the Angular commit message format. Each type indicates the kind
    of change being made to the codebase.

    Notes
    -----
    Conventional commits provide a standardized format for commit messages,
    making it easier to understand the nature of changes and automate
    version management and changelog generation.

    Examples
    --------
    >>> commit_type = CommitType.FEAT
    >>> str(commit_type)
    'feat'

    >>> CommitType.FIX in CommitType
    True
    """
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
    """Container for project metadata and configuration information.

    This dataclass holds information about a Python project's current state,
    including its configuration, dependencies, and development environment setup.
    It's used throughout the dev-assist tool to make decisions about project
    management operations.

    Attributes
    ----------
    name : str
        The project name, typically derived from pyproject.toml or directory name.
    path : Path
        Absolute path to the project root directory.
    has_git : bool
        True if the project is a git repository.
    has_pyproject : bool
        True if pyproject.toml exists in the project root.
    has_tests : bool
        True if a tests or test directory exists.
    python_version : str
        Required Python version (e.g., "3.11"), parsed from pyproject.toml.
    dependencies : list[str]
        List of production dependencies from pyproject.toml.
    dev_dependencies : list[str]
        List of development dependencies from pyproject.toml.

    Notes
    -----
    This information is gathered by scanning the project directory and parsing
    configuration files. It's used to determine what operations are available
    and how to execute them safely.

    Examples
    --------
    >>> info = ProjectInfo(
    ...     name="my-project",
    ...     path=Path("/path/to/project"),
    ...     has_git=True,
    ...     has_pyproject=True,
    ...     has_tests=True,
    ...     python_version="3.11",
    ...     dependencies=["requests"],
    ...     dev_dependencies=["pytest"]
    ... )
    >>> info.name
    'my-project'
    """
    name: str
    path: Path
    has_git: bool
    has_pyproject: bool
    has_tests: bool
    python_version: str
    dependencies: list[str]
    dev_dependencies: list[str]


class UVManager:
    """Static utility class for managing uv package manager operations.

    This class provides a high-level interface to uv commands for project
    initialization, dependency management, and command execution. All methods
    are static and handle subprocess calls to uv with appropriate error handling.

    Notes
    -----
    uv is a fast Python package manager written in Rust. This class abstracts
    common uv operations used in Python project development workflows.

    Examples
    --------
    >>> UVManager.check_installed()
    True

    >>> path = UVManager.init_project("my-project", "3.11")
    >>> str(path)
    'my-project'
    """
    
    @staticmethod
    def check_installed() -> bool:
        """Check if uv package manager is installed and accessible.

        Returns
        -------
        bool
            True if uv is installed and can be executed, False otherwise.

        Examples
        --------
        >>> UVManager.check_installed()
        True
        """
        try:
            sp.run(["uv", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, sp.CalledProcessError):
            return False
    
    @staticmethod
    def init_project(name: str, python_version: Optional[str] = None) -> Path:
        """Initialize a new uv project with optional Python version specification.

        Parameters
        ----------
        name : str
            Name of the project directory to create.
        python_version : str, optional
            Python version to use for the project (e.g., "3.11").

        Returns
        -------
        Path
            Path to the created project directory.

        Raises
        ------
        subprocess.CalledProcessError
            If uv init command fails.

        Examples
        --------
        >>> path = UVManager.init_project("my-app", "3.11")
        >>> path.name
        'my-app'
        """
        cmd = ["uv", "init", name]
        if python_version:
            cmd.extend(["--python", python_version])

        sp.run(cmd, check=True)
        return Path(name)
    
    @staticmethod
    def add_dependency(package: str, dev: bool = False, extras: Optional[list[str]] = None) -> None:
        """Add a dependency to the current uv project.

        Parameters
        ----------
        package : str
            Package name to add, optionally with version specifier.
        dev : bool, default False
            If True, add as a development dependency.
        extras : list[str], optional
            List of package extras to include (e.g., ["test", "docs"]).

        Raises
        ------
        subprocess.CalledProcessError
            If uv add command fails.

        Examples
        --------
        >>> UVManager.add_dependency("requests>=2.0")
        >>> UVManager.add_dependency("pytest", dev=True)
        >>> UVManager.add_dependency("fastapi", extras=["all"])
        """
        cmd = ["uv", "add"]

        if dev:
            cmd.append("--dev")

        if extras:
            package = f"{package}[{','.join(extras)}]"

        cmd.append(package)
        sp.run(cmd, check=True)
    
    @staticmethod
    def remove_dependency(package: str, dev: bool = False) -> None:
        """Remove a dependency from the current uv project.

        Parameters
        ----------
        package : str
            Name of the package to remove.
        dev : bool, default False
            If True, remove from development dependencies.

        Raises
        ------
        subprocess.CalledProcessError
            If uv remove command fails.

        Examples
        --------
        >>> UVManager.remove_dependency("requests")
        >>> UVManager.remove_dependency("pytest", dev=True)
        """
        cmd = ["uv", "remove"]
        if dev:
            cmd.append("--dev")
        cmd.append(package)
        sp.run(cmd, check=True)
    
    @staticmethod
    def sync_dependencies() -> None:
        """Synchronize project dependencies from pyproject.toml.

        Installs or removes packages to match the current pyproject.toml
        dependency specifications.

        Raises
        ------
        subprocess.CalledProcessError
            If uv sync command fails.

        Examples
        --------
        >>> UVManager.sync_dependencies()
        """
        sp.run(["uv", "sync"], check=True)
    
    @staticmethod
    def run_command(command: list[str]) -> sp.CompletedProcess:
        """Execute a command within the uv-managed virtual environment.

        Parameters
        ----------
        command : list[str]
            Command and arguments to execute.

        Returns
        -------
        subprocess.CompletedProcess
            Result of the command execution with stdout, stderr, and returncode.

        Examples
        --------
        >>> result = UVManager.run_command(["python", "--version"])
        >>> result.returncode
        0
        """
        return sp.run(["uv", "run"] + command, capture_output=True, text=True)
    
    @staticmethod
    def update_dependencies(packages: Optional[list[str]] = None) -> None:
        """Update project dependencies to their latest compatible versions.

        Parameters
        ----------
        packages : list[str], optional
            Specific packages to update. If None, updates all dependencies.

        Raises
        ------
        subprocess.CalledProcessError
            If uv lock or sync commands fail.

        Notes
        -----
        This method first updates the lock file with new versions, then syncs
        the environment to install the updated packages.

        Examples
        --------
        >>> UVManager.update_dependencies()  # Update all
        >>> UVManager.update_dependencies(["requests", "click"])  # Update specific
        """
        cmd = ["uv", "lock", "--upgrade"]
        if packages:
            for pkg in packages:
                cmd.extend(["--upgrade-package", pkg])
        sp.run(cmd, check=True)
        sp.run(["uv", "sync"], check=True)


class GitManager:
    """Static utility class for managing git repository operations.

    This class provides a high-level interface to common git operations
    used in development workflows, including repository initialization,
    staging files, committing changes, and status checking.

    Notes
    -----
    Uses the GitPython library to interact with git repositories.
    All operations are performed on the current working directory's repository
    unless a specific path is provided.

    Examples
    --------
    >>> repo = GitManager.init_repo(Path("my-project"))
    >>> GitManager.stage_files(repo, ["*.py"])
    ['file1.py', 'file2.py']
    """
    
    @staticmethod
    def init_repo(path: Path) -> Repo:
        """Initialize a new git repository at the specified path.

        Parameters
        ----------
        path : Path
            Directory path where the git repository should be initialized.

        Returns
        -------
        Repo
            GitPython Repo object for the initialized repository.

        Examples
        --------
        >>> from pathlib import Path
        >>> repo = GitManager.init_repo(Path("my-project"))
        >>> repo.git_dir
        'my-project/.git'
        """
        return Repo.init(path)
    
    @staticmethod
    def get_repo(path: Path = Path.cwd()) -> Optional[Repo]:
        """Get the git repository object for a given path.

        Parameters
        ----------
        path : Path, default Path.cwd()
            Directory path to search for a git repository.

        Returns
        -------
        Repo or None
            GitPython Repo object if a repository is found, None otherwise.

        Notes
        -----
        Searches parent directories if no repository is found in the given path.

        Examples
        --------
        >>> repo = GitManager.get_repo()
        >>> if repo:
        ...     print("In a git repository")
        """
        try:
            return Repo(path, search_parent_directories=True)
        except:
            return None
    
    @staticmethod
    def stage_files(repo: Repo, patterns: list[str]) -> list[str]:
        """Stage files in the repository that match given patterns.

        Parameters
        ----------
        repo : Repo
            GitPython repository object.
        patterns : list[str]
            List of file patterns to stage. Use "." to stage all files.

        Returns
        -------
        list[str]
            List of staged file paths or descriptions.

        Examples
        --------
        >>> staged = GitManager.stage_files(repo, ["*.py", "README.md"])
        >>> print(f"Staged {len(staged)} files")
        """
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
        """Create a commit with the given message.

        Parameters
        ----------
        repo : Repo
            GitPython repository object.
        message : str
            Commit message.

        Returns
        -------
        str
            Hexadecimal SHA hash of the created commit.

        Examples
        --------
        >>> sha = GitManager.commit(repo, "Add new feature")
        >>> print(f"Committed: {sha[:8]}")
        """
        return repo.index.commit(message).hexsha
    
    @staticmethod
    def get_status(repo: Repo) -> dict[str, list[str]]:
        """Get the current status of the git repository.

        Parameters
        ----------
        repo : Repo
            GitPython repository object.

        Returns
        -------
        dict[str, list[str]]
            Dictionary with keys 'modified', 'added', 'deleted', 'untracked'
            containing lists of file paths for each status type.

        Examples
        --------
        >>> status = GitManager.get_status(repo)
        >>> print(f"Modified files: {len(status['modified'])}")
        >>> print(f"Untracked files: {len(status['untracked'])}")
        """
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
    """Gather comprehensive information about a Python project.

    Scans the project directory and parses configuration files to collect
    metadata about the project's current state, dependencies, and setup.

    Parameters
    ----------
    path : Path, default Path.cwd()
        Root directory of the project to analyze.

    Returns
    -------
    ProjectInfo
        Dataclass containing project metadata including name, paths,
        git status, configuration presence, and dependency lists.

    Notes
    -----
    This function reads pyproject.toml to extract project name, dependencies,
    and Python version requirements. It also checks for the presence of
    common project files and directories.

    Examples
    --------
    >>> info = get_project_info()
    >>> print(f"Project: {info.name}")
    >>> print(f"Has tests: {info.has_tests}")
    """
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
    """Create a standard set of default files for a new Python project.

    Generates a basic project structure including source directory, main module,
    test suite, documentation, and common configuration files.

    Parameters
    ----------
    project_path : Path
        Root directory of the project where files will be created.
    project_name : str
        Name of the project, used for module naming and documentation.

    Notes
    -----
    Creates the following structure:
    - Source package directory with __init__.py and main.py
    - tests/ directory with test_main.py
    - README.md with basic project documentation
    - .gitignore with common Python exclusions

    Examples
    --------
    >>> create_default_files(Path("my-app"), "my_app")
    """
    
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
    """Initialize a new Python project with modern tooling and best practices.

    Creates a complete project structure using uv for dependency management,
    sets up testing, linting, and formatting tools, and optionally initializes
    a git repository with an initial commit.

    Parameters
    ----------
    name : str
        Name of the project to create.
    python_version : str, optional
        Python version to use (e.g., "3.11").
    with_git : bool, default True
        Whether to initialize a git repository.
    dev_packages : list[str], default []
        Additional development dependencies to install.

    Raises
    ------
    typer.Exit
        If uv is not installed or project directory already exists.

    Notes
    -----
    This command performs the following steps:
    1. Validates uv installation
    2. Creates project directory with uv init
    3. Generates default project files
    4. Installs common development tools (pytest, ruff, mypy)
    5. Optionally initializes git repository

    Examples
    --------
    Create a basic project:
    >>> init("my-project")

    Create with specific Python version and extra dev tools:
    >>> init("my-project", python_version="3.11", dev_packages=["black", "isort"])
    """
    
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
    """Manage project dependencies using uv package manager.

    Provides a unified interface for adding, removing, updating, and listing
    project dependencies. Supports both production and development dependencies.

    Parameters
    ----------
    action : str
        Action to perform: "add", "remove", "update", or "list".
    packages : list[str], optional
        Package names for add/remove/update actions.
    dev : bool, default False
        Whether to operate on development dependencies.
    extras : list[str], default []
        Package extras to include when adding dependencies.

    Raises
    ------
    typer.Exit
        If uv is not installed, pyproject.toml is missing, or invalid action/packages.

    Notes
    -----
    - Requires pyproject.toml to be present in the current directory
    - Uses uv for all dependency operations
    - Displays dependency list in a formatted table

    Examples
    --------
    Add production dependency:
    >>> deps("add", ["requests"])

    Add development dependency with extras:
    >>> deps("add", ["pytest"], dev=True, extras=["cov"])

    Update all dependencies:
    >>> deps("update")

    List all dependencies:
    >>> deps("list")
    """
    
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
    """Run project tests using pytest with optional coverage reporting.

    Executes the test suite and provides formatted output. Supports running
    all tests or specific test files, with optional coverage analysis.

    Parameters
    ----------
    coverage : bool, default True
        Whether to run tests with coverage analysis.
    verbose : bool, default False
        Enable verbose pytest output.
    file : str, optional
        Path to specific test file to run.

    Raises
    ------
    typer.Exit
        If no tests directory is found or tests fail.

    Notes
    -----
    - Looks for tests/ or test/ directory
    - Uses pytest as the test runner
    - Coverage reports show missing lines when enabled

    Examples
    --------
    Run all tests with coverage:
    >>> test()

    Run specific test file verbosely:
    >>> test(file="tests/test_main.py", verbose=True)

    Run tests without coverage:
    >>> test(coverage=False)
    """
    
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
    """Create a conventional commit with interactive prompts and validation.

    Guides the user through creating a properly formatted conventional commit
    message. Shows repository status, allows selective staging, and supports
    detailed commit messages with breaking change notes.

    Parameters
    ----------
    type : CommitType
        The type of change being committed (feat, fix, docs, etc.).
    scope : str, optional
        Optional scope to group related commits (e.g., "auth", "api").
    breaking : bool, default False
        Whether this commit introduces breaking changes.

    Raises
    ------
    typer.Exit
        If not in a git repository or no changes to commit.

    Notes
    -----
    Follows the conventional commits specification. The commit message format is:
    <type>(<scope>)!: <description>

    Optional body and breaking change notes can be added interactively.

    Examples
    --------
    Create a feature commit:
    >>> commit(CommitType.FEAT, scope="auth")

    Create a breaking fix:
    >>> commit(CommitType.FIX, breaking=True)
    """
    
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
    """Display comprehensive information about the current project.

    Gathers and presents project metadata in a formatted panel, including
    project name, paths, Python version, git status, and dependency counts.

    Notes
    -----
    Uses get_project_info() to collect current project state.
    Displays information in a Rich-formatted panel for easy reading.

    Examples
    --------
    >>> info()  # Shows project info in terminal
    """
    
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
    """Apply quick fixes for common code quality issues.

    Runs automated tools to format code, fix linting issues, and check types.
    Supports running individual tools or all of them in sequence.

    Parameters
    ----------
    what : str, default "all"
        Which fixes to apply: "format", "lint", "types", or "all".

    Raises
    ------
    typer.Exit
        If pyproject.toml is missing or invalid fix target specified.

    Notes
    -----
    Uses the following tools:
    - format: ruff format (code formatting)
    - lint: ruff check --fix (linting and auto-fixes)
    - types: mypy (static type checking)

    All tools are run via uv to ensure proper environment isolation.

    Examples
    --------
    Fix all issues:
    >>> fix()

    Format code only:
    >>> fix("format")

    Check types only:
    >>> fix("types")
    """
    
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