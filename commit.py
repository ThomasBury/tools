#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "textual>=0.58",
#   "llm>=0.26",
#   "llm-gemini>=0.24",
# ]
# ///
"""
commit.py: A micro-agent to generate conventional commit messages.

Usage:
    uv run commit.py

Environment:
    COMMIT_MODEL (optional): Override the default LLM model name.

Interactions:
    - Shows the generated commit message in the terminal.
    - Offers a Textual-powered editor when you choose to edit the message.
"""

import os
import subprocess
import typer
from rich.console import Console
from rich.panel import Panel
import llm
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, TextArea

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_MODEL = os.environ.get("COMMIT_MODEL", "gemini-2.5-flash-lite")


class CommitMessageEditor(App[str | None]):
    """Simple Textual editor for editing commit messages."""

    CSS = """
    Screen {
        layout: vertical;
        padding: 1 2;
        background: $background;
    }

    #instructions {
        margin-bottom: 1;
        text-style: bold;
    }

    TextArea#editor {
        border: tall green;
    }
    """

    BINDINGS = [
        ("ctrl+s", "save", "Save and exit"),
        ("escape", "cancel", "Cancel"),
        ("ctrl+q", "cancel", "Cancel"),
    ]

    def __init__(self, initial_message: str) -> None:
        super().__init__()
        self._initial_message = initial_message

    def compose(self) -> ComposeResult:
        yield Static(
            "Edit the commit message. Press Ctrl+S to save or Esc to cancel.",
            id="instructions",
        )
        yield TextArea(self._initial_message, id="editor", language="markdown")
        yield Footer()

    def on_mount(self) -> None:
        editor = self.query_one(TextArea)
        editor.focus()
        editor.cursor_location = (0, 0)

    def action_save(self) -> None:
        message = self.query_one(TextArea).text
        self.exit(result=message)

    def action_cancel(self) -> None:
        self.exit(result=None)

def get_staged_diff():
    """Returns the staged git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        console.print("[red]Error: git not found. Is it installed and in your PATH?[/red]")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        # A non-zero exit code from `git diff` isn't always an error we should stop for.
        # It can indicate differences, which is what we want.
        # We'll only raise an error if stderr is not empty.
        if e.stderr:
            console.print(f"[red]Error getting git diff:[/red]\n{e.stderr}")
            raise typer.Exit(1)
        return e.stdout


def generate_commit_message(diff: str, model_name: str) -> str:
    """Generates a commit message using an LLM."""
    prompt = f"""
You are an expert programmer tasked with writing a conventional commit message.
Based on the following git diff, generate a commit message.

Rules:
- The message must follow the Conventional Commits specification.
- The subject line must be 50 characters or less.
- The subject line should be followed by a blank line.
- The body should explain the 'why' of the change, not the 'how'.
- The output should be only the commit message, with no extra text or explanation.

Git Diff:
```diff
{diff}
```
"""
    try:
        model = llm.get_model(model_name)
        response = model.prompt(prompt)
        return response.text().strip()
    except Exception as e:
        console.print(f"[red]Error generating commit message:[/red] {e}")
        raise typer.Exit(1)


def edit_commit_message(initial_message: str) -> str | None:
    """Launches the Textual editor to edit a commit message."""
    try:
        return CommitMessageEditor(initial_message).run(return_result=True)
    except Exception as exc:
        console.print(
            f"[red]Textual editor failed, falling back to $EDITOR:[/red] {exc}"
        )
        return typer.edit(initial_message)

def run_git_commit(message: str):
    """Runs git commit with the given message."""
    try:
        subprocess.run(["git", "commit", "-m", message], check=True)
        console.print("[green]✓ Commit successful![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error committing:[/red]\n{e.stderr}")
        raise typer.Exit(1)

@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation and commit directly"),
):
    """Generates a conventional commit message for staged changes."""
    console.print("[cyan]Analyzing staged changes...[/cyan]")
    diff = get_staged_diff()

    if not diff:
        console.print("[yellow]No staged changes found. Nothing to commit.[/yellow]")
        raise typer.Exit()

    console.print("[cyan]Generating commit message...[/cyan]")
    message = generate_commit_message(diff, model)

    console.print(Panel(message, title="Generated Commit Message", border_style="green"))

    if yes:
        run_git_commit(message)
        raise typer.Exit()

    while True:
        action = console.input("Commit with this message? [Y/n/e(dit)]: ").lower().strip()
        if action in ("y", ""):
            run_git_commit(message)
            break
        elif action == "n":
            console.print("[yellow]Commit aborted.[/yellow]")
            break
        elif action == "e":
            edited_message = edit_commit_message(message)
            if edited_message:
                run_git_commit(edited_message.strip())
            else:
                console.print(
                    "[yellow]No changes supplied from editor. Commit aborted.[/yellow]"
                )
            break
        else:
            console.print("[red]Invalid option. Please enter 'y', 'n', or 'e'.[/red]")

if __name__ == "__main__":
    app()
