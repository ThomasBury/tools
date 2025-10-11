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
import subprocess as sp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import llm
import pathspec
import typer
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Defaults
DEFAULT_MODEL = os.environ.get("CODE_REVIEW_MODEL", "gemini-2.5-flash-lite")
MAX_FILE_SIZE = int(os.environ.get("CODE_REVIEW_MAX_SIZE", "100000"))  # 100KB

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
    lines: int
    size: int

    def __init__(self, path: Path, content: str) -> None:
        """Initialize a CodeFile instance.

        Parameters
        ----------
        path : Path
            The absolute path to the code file.
        content : str
            The entire content of the code file as a string.
        """
        self.path = path
        self.content = content
        self.size = len(content.encode('utf-8'))

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

    def __str__(self) -> str:
        """Return a user-friendly string representation of the CodeFile.

        Returns
        -------
        str
            A string representing the CodeFile, showing its path.
        """
        return str(self.path)


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


def find_python_files(path: Path, gitignore_spec: pathspec.PathSpec | None = None) -> list[Path]:
    """Find all Python files in path, respecting gitignore.

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

    Returns
    -------
    list[Path]
        A sorted list of `Path` objects representing the Python files found.

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
    >>> python_files_no_ignore = find_python_files(Path("my_project"))
    >>> print([str(p.relative_to("my_project")) for p in python_files_no_ignore])
    ['main.py', 'utils.py']
    >>>
    >>> # Example 2: Usage with gitignore
    >>> gitignore_content = Path("my_project/.gitignore").read_text()
    >>> spec = pathspec.PathSpec.from_lines(pathspec. patrones.GitPattern, gitignore_content.splitlines())
    >>> python_files_with_ignore = find_python_files(Path("my_project"), gitignore_spec=spec)
    >>> print([str(p.relative_to("my_project")) for p in python_files_with_ignore])
    ['main.py', 'utils.py'] # Note: data/raw.csv is not a python file and .venv is excluded by default
    >>>
    >>> # Example 3: Searching a single python file
    >>> single_file_result = find_python_files(Path("my_project/main.py"))
    >>> print([str(p.relative_to("my_project")) for p in single_file_result])
    ['main.py']
    >>>
    >>> # Example 4: Searching a non-python file
    >>> non_python_file_result = find_python_files(Path("my_project/data/raw.csv"))
    >>> print(non_python_file_result)
    []
    >>>
    >>> # Clean up dummy files and directories
    >>> import shutil
    >>> shutil.rmtree("my_project")
    """
    if path.is_file():
        return [path] if path.suffix == ".py" else []

    files: list[Path] = []
    # Recursively glob for all files ending with '.py'
    for file in path.rglob("*.py"):
        # Skip if gitignored
        # We need to compare the relative path to the parent of the path we started searching from
        # to correctly match gitignore rules which are usually relative to the git root.
        relative_file_path = file.relative_to(path.parent)
        if gitignore_spec and gitignore_spec.match_file(relative_file_path):
            continue

        # Skip common directories that are unlikely to contain relevant Python code
        # We check parts of the relative path to correctly identify ignored directories.
        parts = relative_file_path.parts
        if any(p in {".venv", "venv", ".uv", "__pycache__", ".git", "build", "dist"} for p in parts):
            continue

        files.append(file)

    # Sort the files for consistent output
    return sorted(files)

def read_code_file(file: Path, max_size: int = MAX_FILE_SIZE) -> CodeFile | None:
    """Read a code file with size limits.

    Reads the content of a given code file, checks if its size exceeds a
    specified maximum, and returns a CodeFile object if within limits.
    Handles potential exceptions during file reading by returning None.

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
        Returns None if the file size exceeds max_size or if any exception
        occurs during file reading.

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
    2
    15
    >>> Path.stat = lambda self: MockFile("a"*1000000, 1000000).stat()
    >>> Path.read_text = lambda self, encoding, errors: MockFile("a"*1000000, 1000000).read_text()
    >>> code_file_too_large = read_code_file(Path("large_code.py"), max_size=500000)
    >>> print(code_file_too_large is None)
    True
    >>> # Restore original methods
    >>> Path.stat = original_Path_stat
    >>> Path.read_text = original_Path_read_text
    """
    console.print(f"Debug: read_code_file called for {file}")
    try:
        stat = file.stat()
        console.print(f"Debug: {file} stat.st_size = {stat.st_size}, max_size = {max_size}")
        if stat.st_size > max_size:
            console.print(f"Debug: {file} is too large")
            return None

        content = file.read_text(encoding="utf-8", errors="replace")
        console.print(f"Debug: content read, len={len(content)}")

        console.print(f"Debug: creating CodeFile for {file}")

        return CodeFile(
            path=file,
            content=content
        )
    except Exception as e:
        console.print(f"Debug: exception in read_code_file for {file}: {e}")
        return None

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


def build_review_prompt(files: list[CodeFile], focus: ReviewFocus, context: str) -> str:
    """
    Build a review prompt for the AI.

    Constructs a detailed prompt for an AI code reviewer,
    incorporating specific instructions based on the review focus
    and providing context about the files being reviewed.

    Parameters
    ----------
    files : list[CodeFile]
        A list of CodeFile objects representing the files to be reviewed.
        Only the first 10 files will be listed in the prompt.
    focus : ReviewFocus
        An enum value specifying the primary focus of the review
        (e.g., GENERAL, SECURITY, PERFORMANCE).
    context : str
        Additional context or background information about the code or project
        to guide the review.

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
    >>> prompt = build_review_prompt(files_to_review, review_focus, project_context)
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
        for f in files[:10]  # List first 10 files
    ])

    # Build code content block
    code_content = "\n".join([
        f"--- FILE: {f.path.name} ---\n```python\n{f.content}\n```"
        for f in files
    ])

    prompt = f"""You are an expert code reviewer. Review the following code files with a focus on '{focus.value}'.

{focus_instructions[focus]}

## Project Context
{context or "No additional context provided."}

## Files for Review
{file_list}

## Code to Review
{code_content}

Please provide a structured review in Markdown format. Identify strengths, weaknesses, and specific, actionable suggestions for improvement.
"""

    return prompt


def format_file_stats(files: list[CodeFile]) -> Table:
    """Create a table of files to review.

    Parameters
    ----------
    files : list[CodeFile]
        A list of CodeFile objects to be displayed in the table.

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
        table.add_row(
            str(file.path.name),
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
    try:
        # Find the root of the Git repository
        result = sp.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path if path.is_dir() else path.parent,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            # Not a git repository
            return None

        git_root: Path = Path(result.stdout.strip())

        # Get the brief status of the repository
        result = sp.run(
            ["git", "status", "--short"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0 and result.stdout:
            # Repository has uncommitted changes
            lines = result.stdout.strip().split("\n")
            return f"Git repository with {len(lines)} uncommitted changes"
        elif result.returncode == 0:
            # Repository is clean
            return "Git repository (clean)"
    except Exception:  # Catch any potential exceptions during subprocess execution
        # Silently fail if any error occurs, returning None
        pass

    return None


@app.command()
def review(
    path: Path = typer.Argument(..., help="File or directory to review"),
    focus: ReviewFocus = typer.Option(ReviewFocus.GENERAL, "--focus", "-f", help="Review focus area"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    max_files: int = typer.Option(10, "--max-files", help="Maximum files to review"),
    show_code: bool = typer.Option(False, "--show-code", "-s", help="Show code snippets in terminal"),
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

    console.print(f"Debug: path={path}, exists={path.exists()}, cwd={Path.cwd()}")

    if not path.exists():
        console.print(f"[red]Error:[/red] Path '{path}' does not exist")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding Python files...", total=None)
        root = path if path.is_dir() else path.parent
        gitignore_spec = get_gitignore_spec(root)
        files = find_python_files(path, gitignore_spec)

        if not files:
            console.print("[yellow]No Python files found to review.[/yellow]")
            raise typer.Exit(0)

        if len(files) > max_files:
            console.print(f"[yellow]Found {len(files)} files, reviewing first {max_files}[/yellow]")
            files = files[:max_files]

        progress.update(task, description="Reading files...")
        code_files = [cf for f in files if (cf := read_code_file(f))]
        if not code_files:
            console.print("[red]Error:[/red] No files could be read (are they too large?).")
            raise typer.Exit(1)

        console.print(format_file_stats(code_files))
        if git_status := check_git_status(root):
            console.print(f"[dim]{git_status}[/dim]")

        if show_code:
            for code_file in code_files:
                console.print(Panel(
                    highlight(code_file.content, PythonLexer(), TerminalFormatter()),
                    title=str(code_file.path.name),
                    border_style="green"
                ))

        progress.update(task, description="Generating AI review...")
        project_context = get_project_context(root)
        prompt = build_review_prompt(code_files, focus, project_context)

        try:
            ai_model = llm.get_model(model)
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

@app.command()
def quick(
    path: Path = typer.Argument(..., help="File to quickly review"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model"),
) -> None:
    """
    Quick review of a single file with minimal output.

    This command performs a fast, focused review of a single Python file using
    an AI model. It prioritizes critical issues like bugs, security problems,
    and major performance issues, providing brief and actionable feedback.

    Parameters
    ----------
    path : Path
        Path to the Python file to review. Must be a valid file.
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
    
    if path.suffix != ".py":
        console.print("[yellow]Warning:[/yellow] File is not a Python file")
    
    # Read file
    code_file = read_code_file(path)
    if not code_file:
        console.print("[red]Error:[/red] File is too large or cannot be read")
        raise typer.Exit(1)
    
    # Simple prompt for quick review
    prompt = f"""Quickly review this Python code. Focus on:
1. Critical bugs or security issues
2. Major performance problems
3. Serious code quality issues

Be brief and actionable. List only important issues.

File: {path.name}
```python
{code_file.content}
```
"""
    
    console.print(f"[cyan]Quick review of {path.name}...[/cyan]")
    
    try:
        model = llm.get_model(model)
        response = model.prompt(prompt)
        
        console.print(Panel(
            Markdown(response.text()),
            title=f"Quick Review: {path.name}",
            border_style="yellow"
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

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
