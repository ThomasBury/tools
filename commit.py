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
    - Presents a Textual interface that shows staged changes, guidance, and a side-by-side commit message editor.
    - Use the built-in editor or press Ctrl+O to hand off editing to your system editor.
"""

import os
import subprocess
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel
import llm
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, TextArea

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_MODEL = os.environ.get("COMMIT_MODEL", "gemini-2.5-flash-lite")


class CommitWorkflowApp(App[str | None]):
    """Textual-powered commit assistant with diff viewer and editor."""

    TITLE = "Commit Assistant"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface-darken-2;
    }

    Header {
        background: $primary;
        color: $text;
    }

    #layout {
        height: 1fr;
        margin: 1;
    }

    #info-panel {
        layout: vertical;
        width: 2fr;
        padding: 1;
        background: $surface;
        border: tall $primary;
    }

    #editor-panel {
        layout: vertical;
        width: 1fr;
        padding: 1;
        background: $surface;
        border: tall $accent;
    }

    TextArea#diff-view {
        height: 1fr;
        border: tall $primary;
        background: $boost;
        color: $text;
        padding: 1;
    }

    TextArea#diff-view .textarea--cursor {
        visibility: hidden;
    }

    TextArea#commit-editor {
        height: 1fr;
        border: tall $primary;
        background: $boost;
        color: $text;
    }

    #status {
        padding-top: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+s", "commit", "Commit"),
        ("ctrl+r", "regenerate", "Regenerate"),
        ("ctrl+o", "system_editor", "Open system editor"),
        ("escape", "cancel", "Cancel"),
        ("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        staged_diff: str,
        initial_message: str,
        model_name: str,
        regenerate_message: Callable[[], str],
    ) -> None:
        super().__init__()
        self._diff = staged_diff
        self._initial_message = initial_message
        self._model_name = model_name
        self._regenerate = regenerate_message

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="layout"):
            with Vertical(id="info-panel"):
                yield Static(
                    "[bold]Staged Changes[/bold]\n"
                    f"Model: [cyan]{self._model_name}[/cyan]\n"
                    f"Diff lines: [cyan]{len(self._diff.splitlines())}[/cyan]\n\n"
                    "[green]Ctrl+S[/green] commit · [yellow]Ctrl+R[/yellow] regenerate · "
                    "[green]Ctrl+O[/green] system editor · [red]Esc[/red] cancel",
                    markup=True,
                )
                yield TextArea(
                    self._diff or "No staged changes detected.",
                    id="diff-view",
                    language="diff",
                    read_only=True,
                )
            with Vertical(id="editor-panel"):
                yield Static("[bold]Commit Message[/bold]", markup=True)
                yield TextArea(
                    self._initial_message,
                    id="commit-editor",
                    language="markdown",
                    placeholder="Write your commit message...",
                )
                yield Static("Ready.", id="status")
        yield Footer()

    def on_mount(self) -> None:
        editor = self.query_one("#commit-editor", TextArea)
        editor.focus()

    def action_commit(self) -> None:
        message = self.query_one("#commit-editor", TextArea).text.strip()
        if not message:
            self._set_status("Commit message cannot be empty.", tone="error")
            return
        self.exit(result=message)

    def action_cancel(self) -> None:
        self._set_status("Canceled by user.", tone="warning")
        self.exit(result=None)

    def action_regenerate(self) -> None:
        try:
            regenerated = self._regenerate().strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._set_status(f"Regeneration failed: {exc}", tone="error")
            return
        editor = self.query_one("#commit-editor", TextArea)
        editor.text = regenerated
        editor.cursor_location = (0, 0)
        self._set_status("Commit message regenerated.", tone="success")

    def action_system_editor(self) -> None:
        editor = self.query_one("#commit-editor", TextArea)
        edited = typer.edit(editor.text)
        if edited is None:
            self._set_status("System editor closed without changes.", tone="warning")
            return
        editor.text = edited.strip()
        editor.cursor_location = (0, 0)
        self._set_status("Loaded message from system editor.", tone="success")

    def _set_status(self, message: str, tone: str = "info") -> None:
        palette = {
            "info": "white",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = palette.get(tone, "white")
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"[{color}]{message}[/{color}]")

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


def launch_commit_ui(diff: str, initial_message: str, model_name: str) -> str | None:
    """Launches the Textual workflow for reviewing and editing a commit message."""

    def regenerate() -> str:
        return generate_commit_message(diff, model_name)

    try:
        app = CommitWorkflowApp(diff, initial_message, model_name, regenerate)
        return app.run()
    except Exception as exc:
        console.print(
            f"[red]Textual interface failed, falling back to $EDITOR:[/red] {exc}"
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

    if yes:
        console.print(Panel(message, title="Generated Commit Message", border_style="green"))
        run_git_commit(message)
        raise typer.Exit()

    edited_message = launch_commit_ui(diff, message, model)
    if edited_message is None:
        console.print("[yellow]Commit aborted.[/yellow]")
        raise typer.Exit()

    final_message = edited_message.strip()
    if not final_message:
        console.print("[yellow]Commit message left empty. Commit aborted.[/yellow]")
        raise typer.Exit()

    console.print(Panel(final_message, title="Commit Message", border_style="green"))
    run_git_commit(final_message)

if __name__ == "__main__":
    app()
