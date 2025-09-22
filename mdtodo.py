#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "pyyaml>=6.0"
# ]
# ///
"""
Scan a repo for TODOs and a YAML task file; pretty-print what's due.

Fully local, no external services.

Examples
--------
Scan for TODOs in the current repository:

    $ uv run mdtodo.py todo

Show tasks from a YAML file:

    $ uv run mdtodo.py tasks --file tasks.yaml

You can also specify a different path or YAML file as needed.
"""

import re
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

# Regex to match TODO comments in Python files
TODOS = re.compile(r"#\s*TODO[:\s](.*)", re.IGNORECASE)


def repo_root(start: Path) -> Path:
    """
    Find the root of the repository by searching for a .git directory.

    Parameters
    ----------
    start : Path
        The starting path to search from.

    Returns
    -------
    Path
        The root path of the repository, or the resolved start path if .git is not found.
    """
    p: Path = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


@app.command()
def todo(path: str = ".") -> None:
    """
    Scan Python files in the repo for TODO comments and display them in a table.

    Parameters
    ----------
    path : str, optional
        The path to start searching from (default is current directory).

    Returns
    -------
    None
    """
    root: Path = repo_root(Path(path))
    table: Table = Table("File", "Line", "Item")
    # Recursively search for Python files and extract TODOs
    for file in root.rglob("*.py"):
        try:
            for i, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                m: re.Match[str] | None = TODOS.search(line)
                if m:
                    table.add_row(str(file.relative_to(root)), str(i), m.group(1).strip())
        except Exception:
            # Ignore files that can't be read
            pass
    console.print(table)


@app.command()
def tasks(file: str = "tasks.yaml") -> None:
    """
    Read a YAML file of tasks and display them in a table.

    Parameters
    ----------
    file : str, optional
        The YAML file to read tasks from (default is 'tasks.yaml').

    Returns
    -------
    None

    Raises
    ------
    typer.Exit
        If the specified file does not exist.
    """
    p: Path = Path(file)
    if not p.exists():
        typer.echo(f"No {file} found", err=True)
        raise typer.Exit(1)
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    table: Table = Table("Task", "Due", "Status")
    # Display each task in the table
    for t in data.get("tasks", []):
        table.add_row(t.get("title", "?"), t.get("due", "-"), t.get("status", "todo"))
    console.print(table)


if __name__ == "__main__":
    app()
