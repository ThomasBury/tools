#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "ruff>=0.5.0",
#   "mypy>=1.10",
#   "pytest>=8.0",
#   "pytest-cov>=5.0",
#   "pytest-xdist>=3.6",
#   "pip-audit>=2.7",
# ]
# ///
"""
Fast, FOSS quality gate for Python repositories.

This module provides a command-line tool to perform various quality checks on Python projects,
including formatting, linting, type-checking, testing, and dependency auditing.
It uses Ruff for formatting and linting, mypy for type-checking, pytest for testing,
and pip-audit for dependency auditing. If no pyproject.toml is present, it creates a minimal one
configured for these tools with NumPy docstring style.
"""

from __future__ import annotations

import subprocess as sp
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()
ROOT = Path.cwd()

# --- Configuration ---


def get_pyproject_content() -> str:
    """
    Generate pyproject.toml content with the current Python version.

    Returns
    -------
    str
        The content of the pyproject.toml file as a string.
    """
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_version_short = f"py{sys.version_info.major}{sys.version_info.minor}"

    return f"""\
[tool.ruff]
target-version = "{py_version_short}"
line-length = 100
extend-exclude = ["build", ".venv", ".uv", ".mypy_cache", ".pytest_cache"]

[tool.ruff.lint]
select = ["E","F","I","UP","B","N","D","C4","T20","PGH","PIE","PERF"]
# Docstrings: require NumPy style; relax a couple of noisy rules if you like.
ignore = ["D104","D105","D107"]
fixable = ["ALL"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

[tool.mypy]
python_version = "{py_version}"
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_any_generics = true
no_implicit_optional = true
strict_equality = true
pretty = true
# Loosen this if you vendor libs without type hints:
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-q"
"""


# --- Core Logic ---


def run(cmd: list[str]) -> int:
    """
    Execute a command and return its exit code.

    Parameters
    ----------
    cmd : list[str]
        The command to execute as a list of strings.

    Returns
    -------
    int
        The exit code of the command.
    """
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return sp.call(cmd)


def ensure_pyproject():
    """
    Create a pyproject.toml file if it does not already exist.

    This function checks for the presence of a pyproject.toml file in the current working directory.
    If it does not exist, it generates a minimal configuration file using get_pyproject_content()
    and writes it to disk.
    """
    pj = ROOT / "pyproject.toml"
    if pj.exists():
        return
    console.print(
        "[yellow]No pyproject.toml found; creating a minimal one for Ruff/Mypy/PyTest.[/yellow]"
    )
    pj.write_text(get_pyproject_content(), encoding="utf-8")


# --- Commands ---


@app.command()
def init():
    """
    Create a minimal pyproject.toml if missing.

    This command ensures that a pyproject.toml file exists in the current directory.
    If not present, it generates and writes a default configuration file.
    """
    ensure_pyproject()
    console.print("[green]Initialized pyproject.toml[/green]")


@app.command()
def format():
    """
    Format code with Ruff formatter.

    This command runs Ruff's formatter on the current directory to standardize code formatting.
    """
    ensure_pyproject()
    sys.exit(run([sys.executable, "-m", "ruff", "format", "."]))


@app.command()
def lint(
    file: str = typer.Option(
        None, "--file", "-f", help="Path to a single file to lint and fix."
    )
):
    """
    Lint and autofix code with Ruff.

    This command runs Ruff's linter and fixer on the specified file or the entire directory.
    It enforces NumPy docstring style and can automatically fix many issues.

    Parameters
    ----------
    file : str, optional
        Path to a single file to lint and fix. If not provided, lints the entire directory.
    """
    ensure_pyproject()
    target = file if file else "."
    sys.exit(run([sys.executable, "-m", "ruff", "check", "--fix", target]))


@app.command()
def types():
    """
    Type-check code with mypy.

    This command runs mypy type checker on the current directory to verify type annotations.
    """
    ensure_pyproject()
    sys.exit(run([sys.executable, "-m", "mypy", "."]))


@app.command()
def test():
    """
    Run tests with pytest in parallel.

    This command executes pytest with parallel execution on all available cores using pytest-xdist.
    """
    ensure_pyproject()
    # -n auto uses all physical cores; requires pytest-xdist
    code = run([sys.executable, "-m", "pytest", "-n", "auto"])
    sys.exit(code)


@app.command()
def audit():
    """
    Audit dependencies for security vulnerabilities.

    This command runs pip-audit to check for known security vulnerabilities in project dependencies.
    """
    sys.exit(run([sys.executable, "-m", "pip_audit"]))


@app.command()
def all():
    """
    Run all quality checks in sequence.

    This command executes all quality checks in order: formatting, linting, type-checking, testing, and auditing.
    If any step fails, the process stops and exits with the failure code.
    """
    ensure_pyproject()
    steps = {
        "format": [sys.executable, "-m", "ruff", "format", "."],
        "lint": [sys.executable, "-m", "ruff", "check", "--fix", "."],
        "types": [sys.executable, "-m", "mypy", "."],
        "test": [sys.executable, "-m", "pytest", "-n", "auto"],
        "audit": [sys.executable, "-m", "pip_audit"],
    }

    for name, command in steps.items():
        console.print(f"\n[bold cyan]Running: {name}[/bold cyan]")
        exit_code = run(command)
        if exit_code != 0:
            console.print(
                f"\n[bold red]'{name}' step failed with exit code {exit_code}. Aborting.[/bold red]"
            )
            sys.exit(exit_code)

    console.print("\n[bold green]✅ All checks passed.[/bold green]")


if __name__ == "__main__":
    # Allow `lintfix.py -f <file>` as a shortcut for `lintfix.py lint -f <file>`
    args = sys.argv[1:]
    commands = {cmd.name for cmd in app.registered_commands}
    is_cmd_present = any(arg in commands for arg in args)
    is_f_present = (
        "-f" in args
        or "--file" in args
        or any(a.startswith("--file=") for a in args)
    )

    if not is_cmd_present and is_f_present:
        # If no command is given but -f/--file is, insert 'lint' command
        sys.argv.insert(1, "lint")

    app()