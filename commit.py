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
commit.py: A micro-agent to analyze your working tree, draft commit plans, and
help you land conventional commits.

Usage:
    uv run commit.py

Environment:
    COMMIT_MODEL (optional): Override the default LLM model name.

Interactions:
    - Presents a Textual interface with repo status, AI-generated commit plans,
      and a side-by-side commit message editor.
    - Press Ctrl+A to apply the AI-generated multi-commit plan.
    - Reassign files between commits and press Ctrl+G to regenerate messages.
    - Use the built-in editor or press Ctrl+O to hand off editing to your system editor.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
import llm
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, TextArea, ListView, ListItem, Checkbox

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_MODEL = os.environ.get("COMMIT_MODEL", "gemini-2.5-flash-lite")


@dataclass
class RepoStatus:
    """Snapshot of repository state relevant to committing."""

    branch: str
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    summary: str

    @property
    def has_changes(self) -> bool:
        return bool(self.staged or self.unstaged or self.untracked)


@dataclass
class CommitSuggestion:
    """LLM-generated commit suggestion."""

    title: str
    body: str
    paths: list[str]

    @property
    def message(self) -> str:
        body = self.body.strip()
        return f"{self.title.strip()}\n\n{body}" if body else self.title.strip()


class CommitWorkflowApp(App[Any | None]):
    """Textual-powered commit assistant with editable commit plans."""

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

    #info-panel > .spacer,
    #editor-panel > .spacer {
        height: 1;
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

    .spacer {
        height: 1;
    }
    """

    BINDINGS = [
        ("ctrl+s", "commit", "Commit"),
        ("ctrl+r", "regenerate", "Regenerate selected"),
        ("ctrl+g", "regenerate_plan", "Regenerate plan"),
        ("ctrl+o", "system_editor", "Open system editor"),
        ("ctrl+a", "apply_plan", "Apply commit plan"),
        ("escape", "cancel", "Cancel"),
        ("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        diff_text: str,
        initial_message: str,
        model_name: str,
        regenerate_message: Callable[[], str],
        repo_status: RepoStatus,
        commit_plan: list[CommitSuggestion],
        allow_manual_commit: bool,
    ) -> None:
        super().__init__()
        self._full_diff = diff_text
        self._initial_message = initial_message
        self._model_name = model_name
        self._regenerate_default = regenerate_message
        self._status = repo_status
        self._plan = commit_plan or []
        self._allow_manual_commit = allow_manual_commit

        # Build file inventory
        self._file_status: dict[str, str] = {}
        self._all_files: list[str] = []
        file_sources = [
            (repo_status.staged, "staged"),
            (repo_status.unstaged, "unstaged"),
            (repo_status.untracked, "untracked"),
        ]
        for files, status in file_sources:
            for path in files:
                if path not in self._file_status:
                    self._file_status[path] = status
                    self._all_files.append(path)

        self._assignment: dict[str, Optional[int]] = {
            path: None for path in self._all_files
        }

        for index, suggestion in enumerate(self._plan):
            normalized = []
            for path in suggestion.paths:
                if path in self._assignment:
                    self._assignment[path] = index
                    normalized.append(path)
            suggestion.paths = normalized

        # Ensure plan paths reflect assignments and at least one commit exists.
        self._sync_plan_paths()
        if not self._plan and self._all_files:
            title, body = split_commit_message(initial_message)
            default_paths = [path for path in self._all_files]
            self._plan = [CommitSuggestion(title=title, body=body, paths=default_paths)]
            for path in default_paths:
                self._assignment[path] = 0

        self._message_fresh = [True] * len(self._plan)
        self._selected_index = 0 if self._plan else None

        self._file_checkboxes: dict[str, Checkbox] = {}
        self._checkbox_to_path: dict[Checkbox, str] = {}
        self._updating_checkboxes = False
        self._updating_message = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="layout"):
            with Vertical(id="info-panel"):
                self.summary_panel = Static("", markup=True)
                yield self.summary_panel
                yield Static(classes="spacer")

                instructions = (
                    "[bold]Shortcuts[/bold]\n"
                    "• [green]Enter[/green] select commit\n"
                    "• [green]Space[/green] toggle file in commit\n"
                    "• [green]Ctrl+R[/green] regenerate message\n"
                    "• [green]Ctrl+G[/green] regenerate plan\n"
                    "• [green]Ctrl+A[/green] apply plan\n"
                    "• [green]Ctrl+O[/green] open system editor\n"
                    "• [green]Ctrl+S[/green] commit staged changes\n"
                    "• [red]Esc[/red] cancel"
                )
                yield Static(instructions, markup=True)
                yield Static(classes="spacer")

                yield Static("[bold]Commit Plan[/bold]", markup=True)
                self.commit_list = ListView(*self._build_commit_items(), id="commit-list")
                yield self.commit_list
                yield Static(classes="spacer")

                yield Static("[bold]Files for Selected Commit[/bold]", markup=True)
                file_items = []
                for path in self._all_files:
                    label = self._format_file_label(path)
                    checkbox = Checkbox(label=label, value=False)
                    self._file_checkboxes[path] = checkbox
                    self._checkbox_to_path[checkbox] = path
                    file_items.append(ListItem(checkbox))
                self.file_list = ListView(*file_items, id="file-list")
                yield self.file_list

            with Vertical(id="editor-panel"):
                yield Static("[bold]Commit Diff[/bold]", markup=True)
                self.diff_view = TextArea(
                    self._full_diff,
                    id="diff-view",
                    language="diff",
                    read_only=True,
                )
                yield self.diff_view
                yield Static(classes="spacer")

                yield Static("[bold]Commit Message[/bold]", markup=True)
                self.message_editor = TextArea(
                    self._initial_message,
                    id="commit-editor",
                    language="markdown",
                    placeholder="Write your commit message...",
                )
                yield self.message_editor

                self.status_label = Static("Ready.", id="status")
                yield self.status_label

        yield Footer()

    def on_mount(self) -> None:
        if self._plan:
            self.commit_list.index = 0
        self._update_all_views()
        if self._allow_manual_commit:
            self.message_editor.focus()
        else:
            self.commit_list.focus()

    def action_commit(self) -> None:
        if not self._allow_manual_commit:
            self._set_status(
                "No staged changes. Use Ctrl+A to apply the commit plan instead.",
                tone="warning",
            )
            return
        message = self.message_editor.text.strip()
        if not message:
            self._set_status("Commit message cannot be empty.", tone="error")
            return
        self.exit(result=message)

    def action_cancel(self) -> None:
        self._set_status("Canceled by user.", tone="warning")
        self.exit(result=None)

    def action_regenerate(self) -> None:
        if self._plan and self._selected_index is not None:
            self._set_status("Regenerating commit message...", tone="info")
            suggestion = regenerate_commit_plan_messages(
                [self._plan[self._selected_index]],
                self._status,
                self._model_name,
            )[0]
            self._plan[self._selected_index] = suggestion
            self._message_fresh[self._selected_index] = True
            self._update_message_editor()
            self._update_commit_list()
            self._set_status("Commit message regenerated.", tone="success")
        else:
            try:
                regenerated = self._regenerate_default().strip()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._set_status(f"Regeneration failed: {exc}", tone="error")
                return
            self.message_editor.text = regenerated
            self.message_editor.cursor_location = (0, 0)
            self._set_status("Commit message regenerated.", tone="success")

    def action_regenerate_plan(self) -> None:
        if not self._plan:
            self._set_status("No commit plan available to regenerate.", tone="warning")
            return
        self._set_status("Regenerating plan messages...", tone="info")
        self._plan = regenerate_commit_plan_messages(self._plan, self._status, self._model_name)
        self._message_fresh = [True] * len(self._plan)
        self._update_commit_list()
        self._update_message_editor()
        self._set_status("Plan messages regenerated.", tone="success")

    def action_system_editor(self) -> None:
        edited = typer.edit(self.message_editor.text)
        if edited is None:
            self._set_status("System editor closed without changes.", tone="warning")
            return
        self.message_editor.text = edited.strip()
        self.message_editor.cursor_location = (0, 0)
        self._apply_message_editor_changes()
        self._set_status("Loaded message from system editor.", tone="success")

    def action_apply_plan(self) -> None:
        if not self._plan:
            self._set_status("No commit plan available to apply.", tone="warning")
            return
        plan_payload = [
            {"title": item.title, "body": item.body, "paths": list(item.paths)}
            for item in self._plan
        ]
        self.exit(
            {
                "apply_plan": True,
                "plan": plan_payload,
                "messages_stale": any(not fresh for fresh in self._message_fresh),
                "unassigned": self._get_unassigned_files(),
            }
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view is self.commit_list:
            self._selected_index = event.index
            self._update_all_views()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._updating_checkboxes:
            return
        checkbox = event.checkbox
        path = self._checkbox_to_path.get(checkbox)
        if path is None or self._selected_index is None:
            return
        previous = self._assignment.get(path)
        if event.value:
            if previous == self._selected_index:
                return
            self._assignment[path] = self._selected_index
            if previous is not None and 0 <= previous < len(self._message_fresh):
                self._message_fresh[previous] = False
        else:
            if previous != self._selected_index:
                return
            self._assignment[path] = None
        self._message_fresh[self._selected_index] = False
        self._sync_plan_paths()
        self._update_commit_list()
        self._update_file_checkboxes()
        self._update_diff_view()
        self._update_status_view()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self.message_editor and not self._updating_message:
            self._apply_message_editor_changes()

    def _apply_message_editor_changes(self) -> None:
        if self._selected_index is None or self._selected_index >= len(self._plan):
            return
        title, body = split_commit_message(self.message_editor.text)
        self._plan[self._selected_index].title = title
        self._plan[self._selected_index].body = body
        self._message_fresh[self._selected_index] = True
        self._update_commit_list()

    def _build_commit_items(self) -> list[ListItem]:
        items: list[ListItem] = []
        if not self._plan:
            placeholder = Static("No commit plan available.")
            items.append(ListItem(placeholder))
        else:
            for idx in range(len(self._plan)):
                items.append(ListItem(Static(self._format_commit_label(idx), markup=True)))
        return items

    def _format_commit_label(self, index: int) -> str:
        if index >= len(self._plan):
            return ""
        suggestion = self._plan[index]
        file_count = len(suggestion.paths)
        freshness = "[green]✓[/green]" if self._message_fresh[index] else "[yellow]![/yellow]"
        return (
            f"{freshness} [bold]{index + 1}. {escape(suggestion.title)}[/bold] "
            f"([cyan]{file_count} file{'s' if file_count != 1 else ''}[/cyan])"
        )

    def _format_file_label(self, path: str) -> str:
        status = self._file_status.get(path, "")
        color = {
            "staged": "green",
            "unstaged": "yellow",
            "untracked": "magenta",
        }.get(status, "white")
        return f"[{color}]{escape(path)}[/{color}]"

    def _get_unassigned_files(self) -> list[str]:
        return [path for path, assignment in self._assignment.items() if assignment is None]

    def _sync_plan_paths(self) -> None:
        for index in range(len(self._plan)):
            paths = [path for path in self._all_files if self._assignment.get(path) == index]
            self._plan[index].paths = paths

    def _update_commit_list(self) -> None:
        if not self._plan:
            return
        for index, item in enumerate(self.commit_list.children):
            if isinstance(item, ListItem):
                static = item.query_one(Static)
                static.update(self._format_commit_label(index))

    def _update_file_checkboxes(self) -> None:
        if self._selected_index is None:
            selected = None
        else:
            selected = self._selected_index
        self._updating_checkboxes = True
        try:
            for path, checkbox in self._file_checkboxes.items():
                checkbox.value = self._assignment.get(path) == selected
        finally:
            self._updating_checkboxes = False

    def _update_message_editor(self) -> None:
        if self._selected_index is None or self._selected_index >= len(self._plan):
            return
        message = self._plan[self._selected_index].message
        self._updating_message = True
        try:
            self.message_editor.text = message
            self.message_editor.cursor_location = (0, 0)
        finally:
            self._updating_message = False

    def _update_diff_view(self) -> None:
        if self._selected_index is None or self._selected_index >= len(self._plan):
            self.diff_view.text = self._full_diff
            return
        paths = self._plan[self._selected_index].paths
        diff = collect_diff_for_paths(paths, self._status)
        if diff.strip():
            self.diff_view.text = diff
        else:
            self.diff_view.text = self._full_diff or "(no diff available)"

    def _update_status_view(self) -> None:
        assigned_counts = sum(len(item.paths) for item in self._plan)
        unassigned = len(self._get_unassigned_files())
        lines = [
            "[bold]Repository[/bold]",
            f"• Branch: [cyan]{escape(self._status.branch)}[/cyan]",
            f"• Staged files: [green]{len(self._status.staged)}[/green]",
            f"• Unstaged files: [yellow]{len(self._status.unstaged)}[/yellow]",
            f"• Untracked files: [magenta]{len(self._status.untracked)}[/magenta]",
        ]
        if self._status.summary:
            lines.append(escape(self._status.summary))
        lines.extend(
            [
                "",
                f"Plan commits: [cyan]{len(self._plan)}[/cyan]",
                f"Files in plan: [cyan]{assigned_counts}[/cyan]",
                f"Unassigned files: [yellow]{unassigned}[/yellow]",
            ]
        )
        if any(not fresh for fresh in self._message_fresh):
            lines.append("[yellow]Commit messages need regeneration (Ctrl+G).[/yellow]")
        self.summary_panel.update("\n".join(lines))

    def _update_all_views(self) -> None:
        self._update_commit_list()
        self._update_file_checkboxes()
        self._update_message_editor()
        self._update_diff_view()
        self._update_status_view()

    def _set_status(self, message: str, tone: str = "info") -> None:
        palette = {
            "info": "white",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = palette.get(tone, "white")
        self.status_label.update(f"[{color}]{message}[/{color}]")

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


def get_full_diff() -> str:
    """Return diff between HEAD and working tree (staged + unstaged)."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as exc:
        return exc.stdout or ""
    except FileNotFoundError:
        console.print("[red]Error: git not found. Is it installed and in your PATH?[/red]")
        raise typer.Exit(1)


def get_repo_status() -> RepoStatus:
    """Gather staged, unstaged, and untracked files for display and planning."""

    def run_git(args: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return None
        except FileNotFoundError:
            console.print(
                "[red]Error: git not found. Is it installed and in your PATH?[/red]"
            )
            raise typer.Exit(1)

    branch = (run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD").strip()
    staged_output = run_git(["diff", "--name-only", "--staged"]) or ""
    staged = [line.strip() for line in staged_output.splitlines() if line.strip()]

    unstaged_output = run_git(["diff", "--name-only"]) or ""
    unstaged = [line.strip() for line in unstaged_output.splitlines() if line.strip()]

    untracked_output = run_git(["ls-files", "--others", "--exclude-standard"]) or ""
    untracked = [line.strip() for line in untracked_output.splitlines() if line.strip()]

    status_summary = run_git(["status", "--short", "--branch"]) or ""

    return RepoStatus(
        branch=branch,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        summary=status_summary.strip(),
    )


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


def generate_commit_plan(
    status: RepoStatus,
    diff: str,
    model_name: str,
) -> list[CommitSuggestion]:
    """Ask the LLM to propose a sequence of commits for the current tree."""

    if not status.has_changes:
        return []

    staged_list = "\n".join(f"- {path}" for path in status.staged) or "(none)"
    unstaged_list = "\n".join(f"- {path}" for path in status.unstaged) or "(none)"
    untracked_list = "\n".join(f"- {path}" for path in status.untracked) or "(none)"

    prompt = f"""
You are an expert developer helping to structure a set of git commits.

Here is the current repository state on branch {status.branch!r}:
- Staged files:\n{staged_list}
- Unstaged files:\n{unstaged_list}
- Untracked files:\n{untracked_list}

Create a plan that groups the work into one or more conventional commits. For
each commit provide:
- "title": a conventional commit subject line (<= 50 chars)
- "body": optional body text explaining the why (may be empty)
- "paths": an array of file paths pulled from the lists above that belong in
  the commit. Use file or directory names exactly as provided. Do not invent
  new paths. Each file should appear in at most one commit.

Output valid JSON consisting of an array of commit objects in the order they
should be created. Do not wrap the JSON in code fences or add commentary.

If no meaningful commits are possible, return an empty JSON array [].

Relevant diff (truncated if necessary):
```diff
{diff[:15000]}
```
"""

    try:
        model = llm.get_model(model_name)
        response = model.prompt(prompt)
        raw = response.text().strip()
    except Exception as exc:  # pragma: no cover - LLM failures
        console.print(f"[red]Error generating commit plan:[/red] {exc}")
        return []

    json_text = raw
    if "[" in raw and "]" in raw:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        json_text = raw[start:end]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        console.print("[yellow]Warning: Unable to parse commit plan JSON.[/yellow]")
        return []

    suggestions: list[CommitSuggestion] = []
    if not isinstance(data, list):
        return suggestions

    allowed_paths = set(status.staged + status.unstaged + status.untracked)

    for item in data[:5]:  # limit to top 5 suggestions
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        body = item.get("body", "")
        paths = item.get("paths", [])
        if not title:
            continue
        if not isinstance(paths, list):
            paths = []
        filtered_paths = [str(p).strip() for p in paths if str(p).strip() in allowed_paths]
        suggestions.append(
            CommitSuggestion(title=str(title), body=str(body or ""), paths=filtered_paths)
        )

    return suggestions


def split_commit_message(message: str) -> tuple[str, str]:
    """Split a full commit message into title and body."""

    lines = [line.rstrip() for line in message.strip().splitlines() if line.strip() or line == ""]
    if not lines:
        return "chore: update", ""
    title = lines[0][:72]
    body = "\n".join(lines[1:]).strip()
    return title, body


def collect_diff_for_paths(paths: list[str], status: RepoStatus) -> str:
    """Collect combined diff for the specified paths."""

    if not paths:
        return ""

    staged_set = set(status.staged)
    unstaged_set = set(status.unstaged)
    untracked_set = set(status.untracked)

    diff_parts: list[str] = []

    staged_paths = [path for path in paths if path in staged_set]
    if staged_paths:
        diff_parts.append(run_git_diff_command(["git", "diff", "--staged", "--", *staged_paths]))

    working_paths = [path for path in paths if path in unstaged_set or path in staged_set]
    if working_paths:
        diff_parts.append(run_git_diff_command(["git", "diff", "--", *working_paths]))

    for path in paths:
        if path in untracked_set:
            diff_parts.append(run_git_diff_command(["git", "diff", "--no-index", "/dev/null", path]))

    combined = "\n".join(part for part in diff_parts if part)
    return combined


def run_git_diff_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            return result.stdout
        if result.stderr and "fatal" in result.stderr.lower():
            return ""
        return result.stdout or ""
    except FileNotFoundError:
        console.print("[red]Error: git not found while collecting diff.[/red]")
        raise typer.Exit(1)


def regenerate_commit_plan_messages(
    plan: list[CommitSuggestion],
    status: RepoStatus,
    model_name: str,
) -> list[CommitSuggestion]:
    """Regenerate commit messages for the current plan assignments."""

    refreshed: list[CommitSuggestion] = []
    for suggestion in plan:
        diff = collect_diff_for_paths(suggestion.paths, status)
        if diff.strip():
            message = generate_commit_message(diff, model_name)
            title, body = split_commit_message(message)
        else:
            title, body = suggestion.title, suggestion.body
        refreshed.append(CommitSuggestion(title=title, body=body, paths=list(suggestion.paths)))
    return refreshed


def launch_commit_ui(
    diff: str,
    initial_message: str,
    model_name: str,
    status: RepoStatus,
    plan: list[CommitSuggestion],
    allow_manual_commit: bool,
) -> Any | None:
    """Launches the Textual workflow for reviewing and editing a commit message."""

    def regenerate() -> str:
        return generate_commit_message(diff, model_name)

    try:
        app = CommitWorkflowApp(
            diff,
            initial_message,
            model_name,
            regenerate,
            status,
            plan,
            allow_manual_commit,
        )
        return app.run()
    except Exception as exc:
        console.print(
            f"[red]Textual interface failed, falling back to $EDITOR:[/red] {exc}"
        )
        return typer.edit(initial_message)


def apply_commit_plan(plan: list[CommitSuggestion]) -> None:
    """Apply the generated commit plan by staging files and committing."""

    if not plan:
        console.print("[yellow]No commit plan to apply.[/yellow]")
        return

    for suggestion in plan:
        if not suggestion.paths:
            console.print(
                f"[yellow]Skipping '{suggestion.title}' because it has no files.[/yellow]"
            )
            continue

        try:
            subprocess.run(["git", "reset"], check=True)
        except subprocess.CalledProcessError as exc:
            console.print(
                "[red]Failed to reset staging area before applying plan:[/red]\n"
                f"{exc.stderr or exc.stdout}"
            )
            raise typer.Exit(1)

        console.print(
            Panel(
                "\n".join(
                    [
                        f"[bold]{escape(suggestion.title)}[/bold]",
                        escape(suggestion.body) or "(no body)",
                        "",
                        "Files:" if suggestion.paths else "",
                        "\n".join(f"• {escape(path)}" for path in suggestion.paths),
                    ]
                ),
                title="Applying Commit",
                border_style="cyan",
            )
        )

        try:
            subprocess.run(["git", "add", "--", *suggestion.paths], check=True)
        except subprocess.CalledProcessError as exc:
            console.print(
                f"[red]Failed to stage files for commit '{suggestion.title}':[/red]\n{exc.stderr}"
            )
            raise typer.Exit(1)

        run_git_commit(suggestion.message)
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
    """Analyze changes, generate commit plans, and help author commits."""

    console.print("[cyan]Inspecting repository status...[/cyan]")
    status = get_repo_status()

    if not status.has_changes:
        console.print("[yellow]No changes detected. Nothing to commit.[/yellow]")
        raise typer.Exit()

    staged_diff = get_staged_diff()
    full_diff = get_full_diff()
    diff_for_ui = staged_diff or full_diff or "No diff available."

    console.print("[cyan]Drafting commit plan...[/cyan]")
    plan = generate_commit_plan(status, full_diff or staged_diff, model)

    allow_manual_commit = bool(status.staged and staged_diff.strip())

    if allow_manual_commit and staged_diff.strip():
        console.print("[cyan]Generating commit message for staged changes...[/cyan]")
        message = generate_commit_message(staged_diff, model)
    elif plan:
        message = plan[0].message
    else:
        message = "chore: describe changes"

    if not plan:
        all_paths_ordered = list(dict.fromkeys(status.staged + status.unstaged + status.untracked))
        if all_paths_ordered:
            title, body = split_commit_message(message)
            plan = [CommitSuggestion(title=title, body=body, paths=all_paths_ordered)]

    if yes:
        if allow_manual_commit:
            console.print(Panel(message, title="Generated Commit Message", border_style="green"))
            run_git_commit(message)
        elif plan:
            console.print("[cyan]Applying AI-generated commit plan...[/cyan]")
            apply_commit_plan(plan)
        else:
            console.print("[yellow]No staged changes or commit plan available. Nothing to do.[/yellow]")
        raise typer.Exit()

    result = launch_commit_ui(
        diff_for_ui,
        message,
        model,
        status,
        plan,
        allow_manual_commit,
    )

    if isinstance(result, dict) and result.get("apply_plan"):
        plan_payload = result.get("plan", [])
        adjusted_plan = [
            CommitSuggestion(
                title=item.get("title", "chore: update"),
                body=item.get("body", ""),
                paths=list(dict.fromkeys(item.get("paths", []))),
            )
            for item in plan_payload
        ]

        unassigned = result.get("unassigned", [])
        if unassigned:
            console.print("[yellow]The following files were left unassigned:[/yellow]")
            for path in unassigned:
                console.print(f" - {path}")

        if result.get("messages_stale"):
            console.print("[cyan]Regenerating commit messages for updated plan...[/cyan]")
            status = get_repo_status()
            adjusted_plan = regenerate_commit_plan_messages(adjusted_plan, status, model)

        if not adjusted_plan:
            console.print("[yellow]No commits to apply after adjustments.[/yellow]")
            raise typer.Exit()

        confirm = typer.confirm(
            "Apply the commit plan and create commits now?",
            default=True,
        )
        if confirm:
            apply_commit_plan(adjusted_plan)
        else:
            console.print("[yellow]Commit plan application cancelled.[/yellow]")
        raise typer.Exit()

    if result is None:
        console.print("[yellow]Commit aborted.[/yellow]")
        raise typer.Exit()

    final_message = str(result).strip()
    if not final_message:
        console.print("[yellow]Commit message left empty. Commit aborted.[/yellow]")
        raise typer.Exit()

    console.print(Panel(final_message, title="Commit Message", border_style="green"))
    run_git_commit(final_message)

if __name__ == "__main__":
    app()
