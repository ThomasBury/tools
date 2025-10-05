#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "mkdocs>=1.6",
#   "mkdocs-material>=9.5",
#   "mkdocstrings[python]>=0.25",
# ]
# ///
"""
docs-serve: bootstrap and serve MkDocs (Material) with mkdocstrings (NumPy style).

- Creates mkdocs.yml, docs/index.md, docs/api.md if missing.
- Serves locally or builds static site.
"""

from __future__ import annotations

import subprocess as sp
import sys
import tomllib
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()
ROOT = Path.cwd()

def run(cmd: list[str]) -> int:
    """Executes a command, prints it, and returns its exit code."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        completed = sp.run(cmd, check=False)
    except FileNotFoundError:
        console.print(
            "[bold red]Command not found.[/bold red] "
            "Ensure dependencies like mkdocs are installed."
        )
        return 127
    except OSError as exc:
        console.print(f"[bold red]Failed to execute command: {exc}[/bold red]")
        return 1
    return completed.returncode

def guess_pkg_name() -> str | None:
    """Tries to guess the package name from pyproject.toml or src/ layout."""
    # Attempt to read from pyproject.toml (PEP 621)
    pyproject_path = ROOT / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            console.print("[yellow]Warning: Could not parse pyproject.toml.[/yellow]")
        else:
            project = data.get("project") or {}
            name = project.get("name")
            if not name:
                tool = data.get("tool") or {}
                poetry = tool.get("poetry") or {}
                if poetry:
                    name = poetry.get("name")
                if not name:
                    hatch = tool.get("hatch") or {}
                    metadata = hatch.get("metadata") or {}
                    name = metadata.get("name")
            if name:
                return name.replace("-", "_")

    # Fallback to src/ layout
    src_dir = ROOT / "src"
    if src_dir.is_dir():
        for path in src_dir.iterdir():
            if path.is_dir() and (path / "__init__.py").exists():
                return path.name

    # Fallback to flat layout
    for path in ROOT.iterdir():
        if path.is_dir() and (path / "__init__.py").exists() and path.name != "docs":
            return path.name

    return None

def ensure_scaffold(explicit_pkg: str | None = None) -> str:
    """Ensures the necessary MkDocs files and directories exist."""
    pkg_name = explicit_pkg or guess_pkg_name()
    if not pkg_name:
        pkg_name = "your_package"
        console.print(
            "[bold yellow]Warning: Could not auto-detect package name.[/bold yellow]\n"
            f"Using placeholder '{pkg_name}'. You may need to edit docs/api.md manually."
        )

    # Ensure docs directory exists
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Scaffold index.md
    index_md = docs_dir / "index.md"
    if not index_md.exists():
        console.print(f"[green]Creating missing file:[/green] {index_md}")
        index_md.write_text("# Project Documentation\n\nWelcome!\n", encoding="utf-8")

    # Scaffold api.md
    api_md = docs_dir / "api.md"
    if not api_md.exists():
        console.print(f"[green]Creating missing file:[/green] {api_md}")
        api_content = f"::: {pkg_name}\n    handler: python\n"
        api_md.write_text(f"# API Reference\n\n{api_content}", encoding="utf-8")

    # Scaffold mkdocs.yml
    mkdocs_yml = ROOT / "mkdocs.yml"
    if not mkdocs_yml.exists():
        console.print(f"[green]Creating missing file:[/green] {mkdocs_yml}")
        mkdocs_yml_content = f"""
site_name: {pkg_name}
theme:
  name: material
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: false
            separate_signature: true
            members_order: source
nav:
  - Home: index.md
  - API: api.md
"""
        mkdocs_yml.write_text(mkdocs_yml_content.strip() + "\n", encoding="utf-8")
    return pkg_name

@app.command()
def build(
    package: str | None = typer.Option(
        None,
        "--package",
        help="Override the detected package/module name.",
    )
):
    """Build static docs site into ./site."""
    ensure_scaffold(package)
    console.print("\n[bold cyan]Building static documentation...[/bold cyan]")
    exit_code = run([sys.executable, "-m", "mkdocs", "build", "--clean"])
    if exit_code == 0:
        console.print("\n[bold green]✅ Build successful. Site generated in ./site folder.[/bold green]")
    else:
        console.print(f"\n[bold red]❌ Build failed with exit code {exit_code}.[/bold red]")
    sys.exit(exit_code)

@app.command(no_args_is_help=False)
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve documentation on."),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface to bind the server.",
    ),
    package: str | None = typer.Option(
        None,
        "--package",
        help="Override the detected package/module name.",
    ),
):
    """Serve docs with live reload (default: 127.0.0.1:8000)."""
    ensure_scaffold(package)
    console.print("\n[bold cyan]Starting live-reload server...[/bold cyan]")
    cmd = [sys.executable, "-m", "mkdocs", "serve", "-a", f"{host}:{port}"]
    exit_code = run(cmd)
    sys.exit(exit_code)

if __name__ == "__main__":
    app()
