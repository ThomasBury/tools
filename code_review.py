#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
#   "llm-gemini>=0.24",
#   "pygments>=2.17",
#   "pathspec>=0.12",
# ]
# ///
"""
Lightweight AI-powered code reviewer.

This module provides a command-line tool for reviewing Python code using AI models.
It analyzes Python files for bugs, style, security, best practices, and performance,
with support for different review focuses. The tool is context-aware, respects
.gitignore patterns, and provides beautiful terminal output with syntax highlighting.

Features
--------
- AI-powered code reviews using multiple LLM models via the llm library
- Multiple review focuses: general, security, performance, testing, style
- Context-aware reviews using project structure (pyproject.toml, README)
- Automatic detection and exclusion of files via .gitignore
- Syntax-highlighted code display in terminal
- Support for reviewing single files or entire directories
- Configurable file size limits and maximum files to review

Examples
--------
Review a single Python file:

>>> ./code_review.py review main.py

Review all Python files in a directory:

>>> ./code_review.py review src/

Perform a security-focused review:

>>> ./code_review.py review . --focus security

Quick review of a single file with minimal output:

>>> ./code_review.py quick utils.py

List available AI models:

>>> ./code_review.py models
"""

from __future__ import annotations

import os
import re
import subprocess as sp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

import llm
import pathspec
import typer
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Defaults
DEFAULT_MODEL = os.environ.get("CODE_REVIEW_MODEL", "gemini-2.5-flash-lite")
MAX_FILE_SIZE = int(os.environ.get("CODE_REVIEW_MAX_SIZE", "100000"))  # 100KB
MAX_PROMPT_FILES = int(os.environ.get("CODE_REVIEW_MAX_PROMPT_FILES", "25"))
MAX_PROMPT_BYTES = int(os.environ.get("CODE_REVIEW_MAX_PROMPT_BYTES", "200000"))
PROMPT_SUMMARY_LIMIT = 10
REVIEW_OUTPUT_GUIDANCE_TEMPLATE = """
You are an expert code reviewer. Respond in a **neutral, professional tone** with **zero conversational filler or flattery**.
Don't soften critiques—be direct and specific. Use bullet points and numbered lists for clarity.
Expose blind spots and hidden risks. Prioritize correctness, security, performance, and maintainability.
If the reasonning is weak or uncertain, explicitly state this in your findings.

**STRICTLY FOLLOW THIS STRUCTURE AND FORMATTING:**

---

## Summary
Provide **exactly two concise sentences**:
1.  Overall code health and quality.
2.  The most notable risks or necessary improvements.

---

## Grade
Provide **one letter grade** (A, B, C, D, F) and **one single-sentence justification**.

---

## Findings
Organize findings under these severity headings **in this exact order**: `High`, `Medium`, `Low`.

Under each severity, use **numbered entries** starting `F1`, `F2`, etc.
Each entry **MUST** follow this format:
`F#. [Category] Short title — explanation referencing specific files/lines (path.py:123).`

**If a severity level has NO findings, write ONLY: `None`**

### High
(Findings here or `None`)

### Medium
(Findings here or `None`)

### Low
(Findings here or `None`)

---

## YAML Findings List (STRICTLY VALID YAML)

Output a **strictly valid YAML list** of all findings. Use double quotes for all string values.

**YAML Schema:**
```yaml
- id: "Fx"
  title: "Short title"
  severity: "High" | "Medium" | "Low"
  category: "Syntax" | "Correctness" | "Security" | "Performance" | "Architecture" | "Maintainability" | "Readability" | "Typing" | "Testing" | "Documentation"
  files:
    - "path.py:line" # Must be non-empty; use ["unknown:0"] if exact file/line is unavailable.
  action: "One-sentence remediation step"
````

**If there are NO findings in total across ALL severities, output ONLY: `[]`**

-----

## Recommendations

Provide a bulleted list mapping **recommended actions** to their finding IDs (e.g., `Fix input validation (F2)`). **Only focus on actions that significantly improve safety, performance, correctness, or maintainability.**

"""


def build_output_guidance(metadata_block: str) -> str:
    """Return the structured output instructions with embedded metadata."""

    return REVIEW_OUTPUT_GUIDANCE_TEMPLATE.format(metadata=metadata_block)


def generate_metadata_entries(
    focus_label: str,
    model_name: str,
    scope: Path,
    git_status: str | None = None,
    git_branch: str | None = None,
    git_commit: str | None = None,
) -> str:
    """Create metadata bullet points for the AI response."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entries = [
        f"- Timestamp (UTC): {timestamp}",
        f"- Focus: {focus_label}",
        f"- Model: {model_name}",
        f"- Scope Root: {scope}",
    ]
    if git_branch:
        entries.append(f"- Git Branch: {git_branch}")
    if git_commit:
        entries.append(f"- Git Commit: {git_commit}")
    if git_status:
        entries.append(f"- Git Status: {git_status}")
    return "\n".join(entries)

class ReviewFocus(str, Enum):
    """Types of review focus.

    An enumeration representing different categories or areas of focus for a code review.
    This allows for categorization and filtering of reviews based on their primary objective.

    Attributes
    ----------
    GENERAL : str
        Represents a general code review covering multiple aspects.
    SECURITY : str
        Represents a review specifically focused on security vulnerabilities and best practices.
    PERFORMANCE : str
        Represents a review focused on code performance, efficiency, and resource utilization.
    TESTING : str
        Represents a review primarily concerned with the quality, coverage, and effectiveness of tests.
    STYLE : str
        Represents a review focused on code style, readability, and adherence to conventions.
    """
    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    STYLE = "style"

@dataclass(slots=True)
class CodeFile:
    """Represents a code file to review.

    This class encapsulates the information about a code file, including its path,
    content, the number of lines, and its size in bytes.

    Attributes
    ----------
    path : Path
        The absolute path to the code file.
    content : str
        The entire content of the code file as a string.
    lines : int
        The total number of lines in the code file.
    size : int
        The size of the code file in bytes.
    """
    path: Path
    content: str
    size: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived attributes after initialization."""
        self.size = len(self.content.encode('utf-8'))

    @property
    def lines(self) -> int:
        return len(self.content.splitlines())

    def get_lines(self) -> list[str]:
        """Return all lines of the code file.

        Returns
        -------
        list[str]
            A list of strings, where each string is a line from the code file.
        """
        return self.content.splitlines()

    def get_line(self, line_num: int) -> str | None:
        """Return a specific line from the code file.

        Parameters
        ----------
        line_num : int
            The 1-based line number to retrieve.

        Returns
        -------
        str | None
            The content of the specified line, or None if the line number is out of bounds.
        """
        lines = self.get_lines()
        if 1 <= line_num <= len(lines):
            return lines[line_num - 1]
        return None

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the CodeFile.

        Returns
        -------
        str
            A string representing the CodeFile, including its path and line count.
        """
        return f"CodeFile(path='{self.path}', lines={self.lines})"


@dataclass(slots=True)
class SecretPattern:
    """Compiled regular expressions used to redact sensitive literals.

    Attributes
    ----------
    label : str
        Human-readable name used in user-facing warnings.
    regex : re.Pattern[str]
        Compiled pattern that matches a specific credential shape.
    replacement : Callable[[re.Match[str]], str]
        Callback that produces the sanitized text for each match.
    """

    label: str
    regex: re.Pattern[str]
    replacement: Callable[[re.Match[str]], str]


def _redact_assignment_value(match: re.Match[str]) -> str:
    """Redact assignment-style literals while preserving syntax.

    Parameters
    ----------
    match : re.Match[str]
        Match object that exposes the assignment prefix and raw value.

    Returns
    -------
    str
        Replacement text that retains the prefix while masking the value.
    """

    prefix = match.group("prefix")
    value = match.group("value")
    if not value:
        return prefix + "'[REDACTED]'"
    if value[0] in {'"', "'"}:
        quote = value[0]
        return f"{prefix}{quote}[REDACTED]{quote}"
    return f"{prefix}[REDACTED]"


def _redact_generic_literal(match: re.Match[str]) -> str:
    """Mask generic quoted credential literals.

    Parameters
    ----------
    match : re.Match[str]
        Match object with captured prefix and surrounding quotes.

    Returns
    -------
    str
        Sanitized literal that keeps delimiters but removes the secret.
    """

    prefix = match.group("prefix")
    quote = match.group("quote")
    return f"{prefix}{quote}[REDACTED]{quote}"


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern(
        "AWS access key",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        lambda _match: "[REDACTED_AWS_ACCESS_KEY]",
    ),
    SecretPattern(
        "AWS secret key",
        re.compile(
            r"(?i)(?P<prefix>aws[_-]?secret[_-]?access[_-]?key\s*(?:=|:)\s*)(?P<value>['\"][A-Za-z0-9/+=]{40}['\"]|[A-Za-z0-9/+=]{40})"
        ),
        _redact_assignment_value,
    ),
    SecretPattern(
        "Generic credential literal",
        re.compile(
            r"(?i)(?P<prefix>\b(?:api|auth|client|access|secret|token|password)[\w-]*\b\s*(?:=|:)\s*)(?P<quote>['\"])(?P<value>[^'\"]{12,})(?P=quote)"
        ),
        _redact_generic_literal,
    ),
    SecretPattern(
        "Bearer token",
        re.compile(r"(?i)(?P<prefix>bearer\s+)(?P<value>[A-Za-z0-9\-_.]{20,})"),
        lambda match: f"{match.group('prefix')}[REDACTED_BEARER_TOKEN]",
    ),
)


def redact_hardcoded_secrets(content: str) -> tuple[str, list[str]]:
    """Redact obvious secret literals before sending code to an LLM.

    Parameters
    ----------
    content : str
        Raw code content that might contain embedded credentials.

    Returns
    -------
    tuple[str, list[str]]
        Two-tuple of sanitized content and a list of secret labels found.
    """

    sanitized = content
    labels: list[str] = []

    for pattern in SECRET_PATTERNS:
        found = False

        def _replacer(match: re.Match[str]) -> str:
            nonlocal found
            found = True
            return pattern.replacement(match)

        sanitized = pattern.regex.sub(_replacer, sanitized)
        if found:
            labels.append(pattern.label)

    return sanitized, labels


def sanitize_prompt_contents(files: list[CodeFile]) -> tuple[dict[Path, str], list[str]]:
    """Return sanitized contents and warnings for a batch of files.

    Parameters
    ----------
    files : list[CodeFile]
        Files whose contents should be scrubbed for credentials.

    Returns
    -------
    tuple[dict[Path, str], list[str]]
        Mapping of file paths to sanitized text and warning messages.
    """

    sanitized_map: dict[Path, str] = {}
    warnings: list[str] = []

    for file in files:
        sanitized_text, labels = redact_hardcoded_secrets(file.content)
        sanitized_map[file.path] = sanitized_text
        if labels:
            joined = ", ".join(labels)
            warnings.append(f"{file.path}: {joined}")

    return sanitized_map, warnings


def get_repo_root(path: Path) -> Path | None:
    """Return the Git repository root containing *path*, if any."""

    try:
        result = sp.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path if path.is_dir() else path.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError):
        return None
    except sp.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None
    
    output = result.stdout.strip()
    if not output:
        return None

    return Path(output)


def get_gitignore_spec(root: Path) -> pathspec.PathSpec | None:
    """Get gitignore patterns if available.

    Searches for a `.gitignore` file in the provided `root` directory. If found,
    it parses the file to create a `pathspec.PathSpec` object representing the
    gitignore rules.

    Parameters
    ----------
    root : Path
        The root directory to search for a `.gitignore` file.

    Returns
    -------
    pathspec.PathSpec | None
        A `pathspec.PathSpec` object containing the parsed gitignore rules if a
        `.gitignore` file is found. Returns `None` if no `.gitignore` file
        exists in the specified `root` directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming a .gitignore file exists with content:
    >>> # "*.pyc"
    >>> # "build/"
    >>> root_dir = Path(".")
    >>> spec = get_gitignore_spec(root_dir)
    >>> if spec:
    ...     print(spec.match_file("my_module.pyc"))
    ...     print(spec.match_file("build/output.log"))
    True
    True

    >>> # Assuming no .gitignore file exists
    >>> non_existent_root = Path("/tmp/non_existent_dir")
    >>> spec_none = get_gitignore_spec(non_existent_root)
    >>> print(spec_none)
    None
    """
    gitignore: Path = root / ".gitignore"
    if gitignore.exists():
        with open(gitignore) as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f)
    return None


def find_python_files(
    path: Path,
    gitignore_spec: pathspec.PathSpec | None = None,
    gitignore_base: Path | None = None,
    limit: int | None = None,
) -> tuple[list[Path], bool]:
    """Find Python files in ``path`` honoring gitignore rules and an optional limit.

    This function recursively searches for all files with the '.py' extension
    within the given `path`. It also filters out files that are ignored by
    a provided `gitignore_spec` and common development directories like
    '.venv', 'venv', '.uv', '__pycache__', '.git', 'build', and 'dist'.

    Parameters
    ----------
    path : Path
        The directory or file path to search within.
    gitignore_spec : pathspec.PathSpec | None, optional
        A PathSpec object representing the gitignore rules. If provided,
        files matching these rules will be excluded. By default, None.
    gitignore_base : Path | None, optional
        Directory used as the reference point for gitignore matching. Defaults
        to ``path`` (or its parent when ``path`` is a file).
    limit : int | None, optional
        Maximum number of files to return. When provided, the traversal stops as
        soon as ``limit + 1`` files are found, so the command remains fast in
        large repositories while still reporting when additional files were
        skipped.

    Returns
    -------
    tuple[list[Path], bool]
        Sorted list of paths and a boolean indicating whether additional files
        were skipped due to ``limit``.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming a directory structure like:
    >>> # /my_project
    >>> # ├── main.py
    >>> # ├── utils.py
    >>> # ├── data
    >>> # │   └── raw.csv
    >>> # ├── .venv
    >>> # │   └── ...
    >>> # └── .gitignore
    >>>
    >>> # Create dummy files and directories for testing
    >>> Path("my_project").mkdir(exist_ok=True)
    >>> Path("my_project/main.py").touch()
    >>> Path("my_project/utils.py").touch()
    >>> Path("my_project/data").mkdir(exist_ok=True)
    >>> Path("my_project/data/raw.csv").touch()
    >>> Path("my_project/.venv").mkdir(exist_ok=True)
    >>> Path("my_project/.gitignore").write_text("*.csv\\n")
    >>>
    >>> # Example 1: Basic usage without gitignore
    >>> python_files_no_ignore, _ = find_python_files(Path("my_project"))
    >>> print([str(p.relative_to("my_project")) for p in python_files_no_ignore])
    ['main.py', 'utils.py']
    >>>
    >>> # Example 2: Usage with gitignore
    >>> gitignore_content = Path("my_project/.gitignore").read_text()
    >>> spec = pathspec.PathSpec.from_lines(pathspec. patrones.GitPattern, gitignore_content.splitlines())
    >>> python_files_with_ignore, _ = find_python_files(Path("my_project"), gitignore_spec=spec)
    >>> print([str(p.relative_to("my_project")) for p in python_files_with_ignore])
    ['main.py', 'utils.py'] # Note: data/raw.csv is not a python file and .venv is excluded by default
    >>>
    >>> # Example 3: Searching a single python file
    >>> single_file_result, _ = find_python_files(Path("my_project/main.py"))
    >>> print([str(p.relative_to("my_project")) for p in single_file_result])
    ['main.py']
    >>>
    >>> # Example 4: Searching a non-python file
    >>> non_python_file_result, _ = find_python_files(Path("my_project/data/raw.csv"))
    >>> print(non_python_file_result)
    []
    >>>
    >>> # Clean up dummy files and directories
    >>> import shutil
    >>> shutil.rmtree("my_project")
    """
    if path.is_file():
        return ([path] if path.suffix == ".py" else []), False

    files: list[Path] = []
    truncated = False
    search_root = path
    ignore_base = gitignore_base or (path if path.is_dir() else path.parent)

    for file in search_root.rglob("*.py"):
        relative_file_path = file.relative_to(search_root)

        if gitignore_spec:
            try:
                ignore_relative = file.relative_to(ignore_base).as_posix()
            except ValueError:
                ignore_relative = relative_file_path.as_posix()

            if gitignore_spec.match_file(ignore_relative):
                continue

        parts = relative_file_path.parts
        if any(p in {".venv", "venv", ".uv", "__pycache__", ".git", "build", "dist"} for p in parts):
            continue

        files.append(file)
        if limit and len(files) > limit:
            truncated = True
            break

    if truncated and limit:
        files = files[:limit]

    return sorted(files), truncated


def path_for_display(file_path: Path, base: Path | None) -> str:
    """Return a human-friendly path relative to ``base`` when possible."""

    if base:
        try:
            return file_path.relative_to(base).as_posix()
        except ValueError:
            pass
    return file_path.as_posix()


def apply_path_filters(
    files: list[Path],
    include_patterns: list[str],
    exclude_patterns: list[str],
    base: Path | None,
) -> tuple[list[Path], list[str], list[str]]:
    """Filter files using optional glob-based include/exclude patterns."""

    include_patterns = [pattern for pattern in include_patterns if pattern]
    exclude_patterns = [pattern for pattern in exclude_patterns if pattern]

    filtered: list[Path] = []
    include_skips: list[str] = []
    exclude_skips: list[str] = []

    for file in files:
        rel_path = path_for_display(file, base)

        if exclude_patterns and any(fnmatch(rel_path, pattern) for pattern in exclude_patterns):
            exclude_skips.append(rel_path)
            continue

        if include_patterns and not any(fnmatch(rel_path, pattern) for pattern in include_patterns):
            include_skips.append(rel_path)
            continue

        filtered.append(file)

    return filtered, include_skips, exclude_skips


def resolve_llm_model(model_id: str) -> llm.Model:
    """Return an installed LLM model or exit with an actionable error."""

    try:
        return llm.get_model(model_id)
    except Exception as exc:
        console.print(f"[red]Unable to load LLM model '{model_id}'.[/red]")

        available_models = [model.model_id for model in llm.get_models()]
        if available_models:
            console.print("[yellow]Available models:[/yellow] " + ", ".join(available_models[:5]))
            if len(available_models) > 5:
                console.print(f"...and {len(available_models) - 5} more")
        else:
            console.print(
                "[yellow]No LLM plugins appear to be installed.[/yellow] "
                "Install one with `uv tool install llm-gemini` or another provider plugin."
            )

        console.print(f"[red]Underlying error:[/red] {exc}")
        raise typer.Exit(1)


def read_code_file(file: Path, max_size: int = MAX_FILE_SIZE) -> CodeFile | None:
    """Read a code file with size limits.

    Reads the content of a given code file, checks if its size exceeds a
    specified maximum, and returns a CodeFile object if within limits.
    Emits warnings for common issues (missing files, permissions, encoding)
    so the user knows why a file was skipped.

    Parameters
    ----------
    file : Path
        The path to the code file to be read.
    max_size : int, optional
        The maximum allowed size of the file in bytes.
        Defaults to MAX_FILE_SIZE.

    Returns
    -------
    CodeFile | None
        A CodeFile object containing the file's path, content, line count,
        and size if the file is read successfully and within the size limit.
        Returns None if the file cannot be read or exceeds ``max_size``.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming a dummy file 'my_code.py' exists with content and size
    >>> # For testing, we'll mock the file operations.
    >>> class MockFile:
    ...     def __init__(self, content: str, size: int):
    ...         self._content = content
    ...         self._size = size
    ...     def stat(self):
    ...         class Stat:
    ...             def __init__(self, size: int):
    ...                 self.st_size = size
    ...         return Stat(self._size)
    ...     def read_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
    ...         return self._content
    >>> original_Path_stat = Path.stat
    >>> original_Path_read_text = Path.read_text
    >>> Path.stat = lambda self: MockFile("print('hello')\\n", 15).stat()
    >>> Path.read_text = lambda self, encoding, errors: MockFile("print('hello')\\n", 15).read_text()
    >>> code_file = read_code_file(Path("my_code.py"))
    >>> if code_file:
    ...     print(code_file.content)
    ...     print(code_file.lines)
    ...     print(code_file.size)
    print('hello')
    >>> Path.stat = lambda self: MockFile("a"*1000000, 1000000).stat()
    >>> Path.read_text = lambda self, encoding, errors: MockFile("a"*1000000, 1000000).read_text()
    >>> code_file_too_large = read_code_file(Path("large_code.py"), max_size=500000)
    >>> print(code_file_too_large is None)
    True
    >>> # Restore original methods
    >>> Path.stat = original_Path_stat
    >>> Path.read_text = original_Path_read_text
    """
    try:
        stat = file.stat()
    except FileNotFoundError:
        console.print(f"[yellow]Skipping {file}:[/yellow] File not found.")
        return None
    except PermissionError:
        console.print(f"[red]Error:[/red] Permission denied reading {file}.")
        return None
    except OSError as exc:
        console.print(f"[red]Error:[/red] Could not stat {file}: {exc}")
        return None

    if stat.st_size > max_size:
        console.print(
            f"[yellow]Skipping {file}:[/yellow] {stat.st_size:,} bytes exceeds limit of {max_size:,} bytes."
        )
        return None

    try:
        content = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        console.print(
            f"[yellow]Warning:[/yellow] {file} is not valid UTF-8. Replacing invalid characters."
        )
        content = file.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        console.print(f"[yellow]Skipping {file}:[/yellow] File disappeared before it could be read.")
        return None
    except PermissionError:
        console.print(f"[red]Error:[/red] Permission denied reading {file}.")
        return None
    except OSError as exc:
        console.print(f"[red]Error:[/red] Could not read {file}: {exc}")
        return None

    return CodeFile(path=file, content=content)

def get_project_context(root: Path) -> str:
    """Get project context from common files.

    This function inspects a given root directory for common project
    configuration files such as `pyproject.toml` and README files. It extracts
    relevant information like the project name, dependencies (from pyproject.toml),
    and an excerpt from the README to provide a concise context.

    Parameters
    ----------
    root : Path
        The root directory of the project to inspect.

    Returns
    -------
    str
        A string containing the project context, formatted with project name,
        dependencies, and a README excerpt, or an empty string if no context
        could be derived.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming a dummy project structure
    >>> dummy_root = Path("./dummy_project")
    >>> dummy_root.mkdir(exist_ok=True)
    >>> (dummy_root / "pyproject.toml").write_text(
    ...     '[project]\\n'
    ...     'name = "my_awesome_project"\\n'
    ...     'dependencies = ["requests", "numpy"]\\n'
    ... )
    >>> (dummy_root / "README.md").write_text("# My Awesome Project\\n\\nThis is a great project that does amazing things.")
    >>> print(get_project_context(dummy_root))
    Project: my_awesome_project
    Dependencies: requests, numpy...
    README excerpt: # My Awesome Project

    >>> # Clean up dummy project
    >>> (dummy_root / "pyproject.toml").unlink()
    >>> (dummy_root / "README.md").unlink()
    >>> dummy_root.rmdir()

    >>> # Example with no context files
    >>> empty_dir = Path("./empty_dir")
    >>> empty_dir.mkdir(exist_ok=True)
    >>> print(get_project_context(empty_dir))
    <BLANKLINE>
    >>> empty_dir.rmdir()
    """
    context_parts: list[str] = []

    # Check for pyproject.toml
    pyproject: Path = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # Use tomllib for Python 3.11+
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                project: dict = data.get("project", {})
                if project:
                    context_parts.append(f"Project: {project.get('name', 'unknown')}")
                    if deps := project.get("dependencies"):
                        # Limit displayed dependencies for brevity
                        context_parts.append(f"Dependencies: {', '.join(deps[:5])}{'...' if len(deps) > 5 else ''}")
        except Exception:
            # Ignore errors during TOML parsing or file reading
            pass

    # Check for README
    readme_filenames: list[str] = ["README.md", "README.rst", "README.txt"]
    for readme_filename in readme_filenames:
        readme_file: Path = root / readme_filename
        if readme_file.exists():
            try:
                content: str = readme_file.read_text(encoding="utf-8", errors="ignore")
                # Get first paragraph, limit to 200 characters
                first_para: str = content.split("\n\n")[0][:200]
                context_parts.append(f"README excerpt: {first_para}")
                break  # Stop after finding the first README
            except Exception:
                # Ignore errors during file reading
                pass

    return "\n".join(context_parts) if context_parts else ""


def render_review_summary(
    files: list[CodeFile],
    focus: ReviewFocus,
    git_status: str | None,
    project_context: str,
    display_root: Path | None = None,
) -> Group:
    """Build a Rich renderable summarizing files and metadata.

    Parameters
    ----------
    files : list[CodeFile]
        A list of code files to be included in the summary.
    focus : ReviewFocus
        The focus of the code review.
    git_status : str | None
        The current git status of the repository.
    project_context : str
        The project context, such as project name and dependencies.
    display_root : Path | None, optional
        Base folder used to show relative paths for file statistics.

    Returns
    -------
    Group
        A Rich Group containing the summary panel and file statistics.
    """

    summary_table = Table(box=None, show_header=False, padding=(0, 1))
    summary_table.add_column("Field", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    total_lines = sum(cf.lines for cf in files)
    summary_table.add_row("Focus", focus.value.capitalize())
    summary_table.add_row("Files", str(len(files)))
    summary_table.add_row("Lines", str(total_lines))
    if git_status:
        summary_table.add_row("Git", git_status)
    if project_context:
        summary_table.add_row("Context", project_context.splitlines()[0][:80])

    return Group(
        Panel(summary_table, title="Review Summary", border_style="blue"),
        format_file_stats(files, base_path=display_root),
    )



def render_quick_summary(code_file: CodeFile) -> Panel:
    """Render a concise summary panel for quick reviews.

    Parameters
    ----------
    code_file : CodeFile
        The code file to be summarized.

    Returns
    -------
    Panel
        A Rich Panel containing the quick review summary.
    """

    table = Table(box=None, show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("File", code_file.path.name)
    table.add_row("Lines", str(code_file.lines))
    table.add_row("Size", f"{code_file.size:,} bytes")

    return Panel(table, title="Quick Review Summary", border_style="yellow")

def build_review_prompt(
    files: list[CodeFile],
    focus: ReviewFocus,
    context: str,
    metadata_text: str,
    sanitized_contents: dict[Path, str] | None = None,
) -> str:
    """
    Build a review prompt for the AI.

    Constructs a detailed prompt for an AI code reviewer,
    incorporating specific instructions based on the review focus
    and providing context about the files being reviewed.

    Parameters
    ----------
    files : list[CodeFile]
        A list of CodeFile objects representing the files to be reviewed.
        Only the first 10 files will be listed in the summary section.
    focus : ReviewFocus
        An enum value specifying the primary focus of the review
        (e.g., GENERAL, SECURITY, PERFORMANCE).
    context : str
        Additional context or background information about the code or project
        to guide the review.
    metadata_text : str
        Pre-rendered metadata bullet list that must be included in the output.
    sanitized_contents : dict[Path, str] | None, optional
        Optional mapping of file paths to sanitized content that has been
        scrubbed of secrets. When provided, these contents are embedded
        instead of the raw file text.

    Returns
    -------
    str
        The generated review prompt string, ready to be sent to the AI.

    Examples
    --------
    >>> from pathlib import Path
    >>> files_to_review = [CodeFile(Path("src/main.py"), 150), CodeFile(Path("src/utils.py"), 80)]
    >>> review_focus = ReviewFocus.GENERAL
    >>> project_context = "This is a new feature for user authentication."
    >>> metadata = "- Timestamp (UTC): 2024-01-01 00:00:00 UTC"
    >>> prompt = build_review_prompt(files_to_review, review_focus, project_context, metadata)
    >>> print(prompt[:50] + "...") # Print a snippet of the generated prompt
    Review Request: ...
    """

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
        for f in files[:PROMPT_SUMMARY_LIMIT]
    ])

    # Build code content block
    sanitized_map = sanitized_contents or {}
    code_content = "\n".join([
        f"--- FILE: {f.path.name} ---\n```python\n{sanitized_map.get(f.path, f.content)}\n```"
        for f in files
    ])

    output_guidance = build_output_guidance(metadata_text)

    prompt = f"""You are an expert code reviewer. Review the following code files with a focus on '{focus.value}'.

{focus_instructions[focus]}

## Project Context
{context or "No additional context provided."}

## Files for Review
{file_list}

## Code to Review
{code_content}

{output_guidance}
"""

    return prompt


def format_file_stats(files: list[CodeFile], base_path: Path | None = None) -> Table:
    """Create a table of files to review.

    Parameters
    ----------
    files : list[CodeFile]
        A list of CodeFile objects to be displayed in the table.
    base_path : Path | None, optional
        Directory used to render relative file paths for clarity.

    Returns
    -------
    Table
        A rich.table.Table object representing the file statistics.

    Examples
    --------
    >>> from pathlib import Path
    >>> files_data = [
    ...     CodeFile(Path("main.py"), 100, 2048),
    ...     CodeFile(Path("utils.py"), 50, 1024),
    ... ]
    >>> table = format_file_stats(files_data)
    >>> # table.title would be "Files to Review"
    >>> # table.columns would contain "File", "Lines", "Size"
    >>> # table.rows would contain the formatted data and total if applicable
    """
    table = Table(title="Files to Review")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right", style="yellow")
    table.add_column("Size", justify="right", style="dim")

    total_lines = 0
    total_size = 0

    for file in files:
        display_path = path_for_display(file.path, base_path)
        table.add_row(
            display_path,
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


def check_git_status(path: Path) -> str | None:
    """Check if a path is within a Git repository and retrieve its status.

    This function determines if the provided `path` is part of a Git
    repository. If it is, it then fetches the short status of the
    repository (e.g., number of uncommitted changes or if it's clean).

    Parameters
    ----------
    path : Path
        The file system path to check. This can be a directory or a file.

    Returns
    -------
    str | None
        A string describing the Git status if the path is in a Git
        repository, formatted as "Git repository with X uncommitted changes"
        or "Git repository (clean)". Returns None if the path is not in
        a Git repository or if an error occurs.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming '/path/to/your/repo' is a Git repository
    >>> repo_path = Path('/path/to/your/repo')
    >>> status = check_git_status(repo_path)
    >>> print(status) # Example output: "Git repository (clean)" or "Git repository with 5 uncommitted changes"

    >>> # Assuming '/path/to/non/repo' is not a Git repository
    >>> non_repo_path = Path('/path/to/non/repo')
    >>> status = check_git_status(non_repo_path)
    >>> print(status) # Output: None
    """
    git_root = get_repo_root(path)
    if not git_root:
        return None

    try:
        result = sp.run(
            ["git", "status", "--short"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, sp.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    if result.stdout:
        lines = result.stdout.strip().split("\n")
        return f"Git repository with {len(lines)} uncommitted changes"

    return "Git repository (clean)"


def get_git_branch_and_commit(path: Path) -> tuple[str | None, str | None]:
    """Return the current branch name and short commit hash for the repo containing path."""

    git_root = get_repo_root(path)
    if not git_root:
        return None, None

    def _run_git(args: list[str]) -> str | None:
        try:
            result = sp.run(
                ["git", *args],
                cwd=git_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, OSError, sp.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git(["rev-parse", "--short", "HEAD"])
    return branch, commit


@app.command()
def review(
    path: Path = typer.Argument(..., help="File or directory to review"),
    focus: ReviewFocus = typer.Option(ReviewFocus.GENERAL, "--focus", "-f", help="Review focus area"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    max_files: int = typer.Option(10, "--max-files", help="Maximum files to review"),
    show_code: bool = typer.Option(False, "--show-code", "-s", help="Show code snippets in terminal"),
    include: list[str] = typer.Option(
        [],
        "--include",
        "-i",
        help="Glob patterns to include (repeatable). Paths evaluated relative to the target path.",
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        "-e",
        help="Glob patterns to exclude (repeatable). Paths evaluated relative to the target path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview file selection and context without sending code to the AI model.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the review markdown output.",
    ),
) -> None:
    """
    Review Python code with AI assistance.

    This command analyzes Python files using an AI model to provide comprehensive
    code reviews. It supports different focus areas and can review single files
    or entire directories. The review includes project context, file statistics,
    and formatted AI feedback.

    Parameters
    ----------
    path : Path
        File or directory path to review. Must exist.
    focus : ReviewFocus, optional
        Review focus area (general, security, performance, testing, style).
        Default is ReviewFocus.GENERAL.
    model : str, optional
        LLM model identifier to use for the review. Default is DEFAULT_MODEL.
    max_files : int, optional
        Maximum number of files to review. Default is 10.
    show_code : bool, optional
        Whether to display code snippets in the terminal. Default is False.
    include : list[str], optional
        One or more glob patterns to include. When provided, only matching
        paths (relative to ``path``) are reviewed.
    exclude : list[str], optional
        One or more glob patterns to exclude from the review.
    dry_run : bool, optional
        When True, displays the planned review summary without invoking the AI.

    Returns
    -------
    None
        This function does not return a value. It prints the review results
        to the console and exits.

    Raises
    ------
    typer.Exit
        If the path does not exist, no Python files are found, or an error
        occurs during the review process.

    Examples
    --------
    Review a single file with default settings:

    >>> review(Path("main.py"))

    Review a directory with security focus:

    >>> review(Path("src"), focus=ReviewFocus.SECURITY)

    Review with custom model and show code:

    >>> review(Path("."), model="gpt-4", show_code=True)
    """


    if not path.exists():
        console.print(f"[red]Error:[/red] Path '{path}' does not exist")
        raise typer.Exit(1)

    if max_files < 1:
        console.print("[red]Error:[/red] --max-files must be at least 1")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding Python files...", total=None)
        scope_root = path if path.is_dir() else path.parent
        repo_root = get_repo_root(scope_root)
        context_root = repo_root or scope_root
        gitignore_spec = get_gitignore_spec(context_root)
        files, truncated = find_python_files(
            path,
            gitignore_spec=gitignore_spec,
            gitignore_base=context_root,
            limit=max_files,
        )

        if not files:
            console.print("[yellow]No Python files found to review.[/yellow]")
            raise typer.Exit(0)

        if truncated:
            console.print(
                f"[yellow]Found more than {max_files} Python files, reviewing first {max_files}[/yellow]"
            )

        files, include_skipped, exclude_skipped = apply_path_filters(
            files,
            include_patterns=include,
            exclude_patterns=exclude,
            base=scope_root,
        )

        def _print_suppressed(message: str, entries: list[str]) -> None:
            console.print(message)
            for rel in entries[:5]:
                console.print(f"  - {rel}")
            if len(entries) > 5:
                console.print("  - ...")

        if include and include_skipped:
            _print_suppressed(
                f"[yellow]{len(include_skipped)} file(s) skipped because they did not match --include[/yellow]",
                include_skipped,
            )

        if exclude and exclude_skipped:
            _print_suppressed(
                f"[yellow]{len(exclude_skipped)} file(s) removed via --exclude[/yellow]",
                exclude_skipped,
            )

        if not files:
            console.print("[yellow]No Python files matched the provided include/exclude filters.[/yellow]")
            raise typer.Exit(0)

        progress.update(task, description="Reading files...")
        code_files = [cf for f in files if (cf := read_code_file(f))]
        if not code_files:
            console.print("[red]Error:[/red] No files could be read (are they too large?).")
            raise typer.Exit(1)

        if len(code_files) > MAX_PROMPT_FILES:
            console.print(
                f"[red]Too many files selected for a single review.[/red] "
                f"Limit is {MAX_PROMPT_FILES}, but {len(code_files)} files were collected. "
                "Reduce --max-files or set CODE_REVIEW_MAX_PROMPT_FILES to override."
            )
            raise typer.Exit(1)

        total_code_bytes = sum(cf.size for cf in code_files)
        if total_code_bytes > MAX_PROMPT_BYTES:
            console.print(
                f"[red]Combined code size {total_code_bytes:,} bytes exceeds the safety limit of "
                f"{MAX_PROMPT_BYTES:,} bytes for a single AI prompt.[/red] "
                "Review fewer files or smaller files, or raise CODE_REVIEW_MAX_PROMPT_BYTES if necessary."
            )
            raise typer.Exit(1)

        sanitized_contents, secret_warnings = sanitize_prompt_contents(code_files)
        if secret_warnings:
            console.print(
                "[yellow]Warning:[/yellow] Potential secrets detected and redacted before sending to the model:"
            )
            for warning in secret_warnings:
                console.print(f"  - {warning}")

        git_status = check_git_status(context_root)
        git_branch, git_commit = get_git_branch_and_commit(context_root)
        project_context = get_project_context(context_root)
        timeline = render_review_summary(
            code_files,
            focus,
            git_status,
            project_context,
            display_root=context_root,
        )
        console.print(timeline)

        metadata_entries = generate_metadata_entries(
            focus_label=focus.value,
            model_name=model,
            scope=context_root,
            git_status=git_status,
            git_branch=git_branch,
            git_commit=git_commit,
        )

        if show_code:
            for code_file in code_files:
                console.print(Panel(
                    highlight(code_file.content, PythonLexer(), TerminalFormatter()),
                    title=str(code_file.path.name),
                    border_style="green"
                ))

        if dry_run:
            console.print("[green]Dry run complete. Re-run without --dry-run to generate an AI review.[/green]")
            return

        progress.update(task, description="Generating AI review...")
        prompt = build_review_prompt(
            code_files,
            focus,
            project_context,
            metadata_entries,
            sanitized_contents,
        )
        ai_model = resolve_llm_model(model)

        try:
            response = ai_model.prompt(prompt)
            review_text = response.text()
        except Exception as e:
            console.print(f"[red]Error during AI review:[/red] {e}")
            raise typer.Exit(1)

    console.print(Panel(
        Markdown(review_text),
        title=f"AI Code Review ({focus.value.capitalize()})",
        border_style="blue"
    ))

    if output:
        try:
            output.write_text(review_text, encoding="utf-8")
            console.print(f"[green]Saved review to {output}[/green]")
        except Exception as exc:
            console.print(f"[red]Failed to save review to {output}:[/red] {exc}")

@app.command()
def quick(
    path: Path = typer.Argument(..., help="File to quickly review"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the quick review markdown output.",
    ),
) -> None:
    """
    Quick review of a single file with minimal output.

    This command performs a fast, focused review of a single Python file using
    an AI model. It prioritizes critical issues like bugs, security problems,
    and major performance issues, providing brief and actionable feedback.

    Parameters
    ----------
    path : Path
        Path to the Python file to review. Must be an existing ``.py`` file.
    model : str, optional
        LLM model identifier to use. Default is DEFAULT_MODEL.

    Returns
    -------
    None
        Prints the quick review results to the console.

    Raises
    ------
    typer.Exit
        If the path is invalid, not a file, or cannot be read.

    Examples
    --------
    Quick review of a file with default model:

    >>> quick(Path("utils.py"))

    Quick review with custom model:

    >>> quick(Path("main.py"), model="claude-3")
    """
    
    if not path.exists() or not path.is_file():
        console.print(f"[red]Error:[/red] '{path}' is not a valid file")
        raise typer.Exit(1)
    
    if path.suffix.lower() != ".py":
        console.print("[red]Error:[/red] Quick reviews only support Python files (.py)")
        raise typer.Exit(1)
    
    # Read file
    code_file = read_code_file(path)
    if not code_file:
        console.print("[red]Error:[/red] File is too large or cannot be read")
        raise typer.Exit(1)

    sanitized_content, secret_warnings = redact_hardcoded_secrets(code_file.content)
    if secret_warnings:
        console.print(
            "[yellow]Warning:[/yellow] Potential secrets detected and redacted before sending to the model:"
        )
        console.print("  - " + ", ".join(secret_warnings))

    git_status = check_git_status(path.parent)
    git_branch, git_commit = get_git_branch_and_commit(path.parent)
    metadata_entries = generate_metadata_entries(
        focus_label="quick",
        model_name=model,
        scope=path.parent,
        git_status=git_status,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    output_guidance = build_output_guidance(metadata_entries)

    # Simple prompt for quick review
    prompt = f"""Provide a rapid triage review of the following Python file. Prioritize:
1. Critical bugs or security issues
2. Major performance problems
3. Serious code quality risks that block merges

Keep the tone neutral and data-driven.

File: {path.name}
```python
{sanitized_content}
```

{output_guidance}
"""
    
    console.print(render_quick_summary(code_file))
    console.print(f"[cyan]Quick review of {path.name}...[/cyan]")
    
    ai_model = resolve_llm_model(model)
    try:
        response = ai_model.prompt(prompt)
        review_text = response.text()

        console.print(Panel(
            Markdown(review_text),
            title=f"Quick Review: {path.name}",
            border_style="yellow"
        ))

        if output:
            try:
                output.write_text(review_text, encoding="utf-8")
                console.print(f"[green]Saved quick review to {output}[/green]")
            except Exception as exc:
                console.print(f"[red]Failed to save quick review to {output}:[/red] {exc}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """List available AI models.

    Retrieves a list of available AI models from the LLM backend and displays them
    in a formatted table. If no models are found, it provides instructions on how
    to install LLM plugins.

    Returns
    -------
    None
        This function does not return any value. It prints information to the console.

    Examples
    --------
    >>> models()
    """
    models = llm.get_models()
    if not models:
        console.print("[yellow]No models found. Install LLM plugins first.[/yellow]")
        console.print("Example: uv tool install llm-gemini")
        return

    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="yellow")

    for model in models:
        provider: str = model.model_id.split("-")[0] if "-" in model.model_id else "unknown"
        table.add_row(model.model_id, provider)

    console.print(table)

if __name__ == "__main__":
    app()
