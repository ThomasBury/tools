#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
#   "llm-gemini>=0.24",
# ]
# ///
"""
release-notes.py: A micro-agent to generate release notes from git history.
"""

import os
import subprocess
from typing import Optional
import typer
from rich.console import Console
from rich.markdown import Markdown
import llm

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_MODEL = os.environ.get("RELEASE_NOTES_MODEL", "gemini-1.5-flash-latest")

def get_git_history(from_tag: Optional[str] = None) -> str:
    """Returns git log history."""
    if from_tag is None:
        try:
            # Get the most recent tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                from_tag = result.stdout.strip()
            else:
                console.print("[yellow]No git tags found. Generating release notes from all commits.[/yellow]")
        except FileNotFoundError:
            console.print("[red]Error: git not found. Is it installed and in your PATH?[/red]")
            raise typer.Exit(1)

    if from_tag:
        log_range = f"{from_tag}..HEAD"
        console.print(f"[cyan]Analyzing commits since tag '{from_tag}'...[/cyan]")
    else:
        log_range = "HEAD"
        console.print("[cyan]Analyzing all commits...[/cyan]")

    try:
        # Get commit subjects since the tag
        result = subprocess.run(
            ["git", "log", log_range, "--pretty=format:%s"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error getting git log:[/red]\n{e.stderr}")
        raise typer.Exit(1)

def generate_release_notes(history: str, model_name: str) -> str:
    """Generates release notes using an LLM."""
    prompt = f"""
You are an expert technical writer. Based on the following list of git commit subjects, generate markdown-formatted release notes.

Rules:
- Group the commits into logical categories like "Features", "Bug Fixes", "Documentation", and "Refactoring".
- For each commit, create a concise, user-friendly list item.
- If a commit subject doesn't seem to fit any category, you can place it under a "Miscellaneous" section.
- The output should be only the markdown release notes, with no extra text or explanation.

Commit Subjects:
{history}
"""
    try:
        model = llm.get_model(model_name)
        response = model.prompt(prompt)
        return response.text().strip()
    except Exception as e:
        console.print(f"[red]Error generating release notes:[/red] {e}")
        raise typer.Exit(1)

@app.command()
def main(
    from_tag: Optional[str] = typer.Option(None, "--from", help="Generate release notes for commits since this tag."),
    output: Optional[typer.FileTextWrite] = typer.Option(None, "--output", "-o", help="Output file to write release notes to."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
):
    """Generates release notes from git commit history."""
    history = get_git_history(from_tag)

    if not history.strip():
        console.print("[yellow]No new commits found. Nothing to generate.[/yellow]")
        raise typer.Exit()

    console.print("[cyan]Generating release notes...[/cyan]")
    notes = generate_release_notes(history, model)

    console.print("\n--- Release Notes ---\n")
    console.print(Markdown(notes))
    console.print("\n--- End Release Notes ---\n")

    if output:
        try:
            output.write(notes)
            console.print(f"[green]✓ Release notes written to {output.name}[/green]")
        except Exception as e:
            console.print(f"[red]Error writing to output file:[/red] {e}")
            raise typer.Exit(1)

if __name__ == "__main__":
    app()
