# Python Micro-Agents Collection

A collection of lightweight, self-contained Python scripts that act as intelligent development assistants. Each micro-agent is a single executable file that combines modern Python tooling with AI capabilities to automate common development tasks.

## 🚀 Key Features

- **Self-contained**: Each script includes its dependencies via PEP 723 inline metadata
- **No installation required**: Just run with `uv` (modern Python package manager)
- **AI-powered**: Integrates with LLMs for intelligent assistance
- **Developer-focused**: Solves real-world development problems
- **Beautiful CLI**: Rich terminal output with colors and formatting

## 📦 What are Micro-Agents?

Micro-agents are standalone Python scripts that:

1. Use shebang with `uv` for zero-setup execution
2. Declare dependencies inline using PEP 723 metadata
3. Focus on a single, well-defined task
4. Provide intelligent assistance through AI integration
5. Offer beautiful, intuitive CLI interfaces

### The Pattern

```python
#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "llm>=0.26",
# ]
# ///
"""
Docstring explaining the micro-agent's purpose and usage.
"""

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()

@app.command()
def main():
    """Your command implementation."""
    console.print("[green]Hello from micro-agent![/green]")

if __name__ == "__main__":
    app()
```

## 🛠️ Available Micro-Agents

### 1. `pr_review.py` - GitHub PR Review Agent

AI-powered pull request reviewer that integrates with GitHub CLI.

**Features:**

- Fetches PR diffs using `gh` CLI
- Provides focused reviews (security, performance, testing, etc.)
- Posts reviews back to GitHub
- Supports multiple AI models

**Usage:**

```bash
# Review a PR in the current repo
./pr_review.py review 123

# Review with security focus
./pr_review.py review 123 --focus security

# Post review to GitHub
./pr_review.py review 123 --post

# Check open PRs
./pr_review.py check
```

### 2. `dev_assist.py` - Development Assistant

Comprehensive development workflow automation tool.

**Features:**

- Project initialization with `uv`
- Dependency management
- Conventional commits helper
- Test execution with coverage
- Code quality fixes

**Usage:**

```bash
# Create new project
./dev_assist.py init my-project --python 3.11

# Add dependencies
./dev_assist.py deps add httpx pytest --dev

# Run tests with coverage
./dev_assist.py test --coverage

# Interactive commit
./dev_assist.py commit feat --scope api

# Fix code issues
./dev_assist.py fix all
```

### 3. `code_review.py` - Lightweight Code Reviewer

Quick AI-powered code reviews without external dependencies.

**Features:**

- Reviews single files or entire directories
- Respects `.gitignore` patterns
- Multiple focus areas (security, performance, style)
- Syntax highlighting in terminal
- Project context awareness

**Usage:**

```bash
# Review a single file
./code_review.py review app.py

# Review entire directory
./code_review.py review src/

# Security-focused review
./code_review.py review . --focus security

# Quick single-file review
./code_review.py quick main.py
```

### 4. `lintfix.py` - Python Quality Gate

Fast, comprehensive code quality checker and fixer.

**Features:**

- Formats with Ruff
- Lints and auto-fixes issues
- Type-checks with mypy
- Runs tests with pytest
- Audits dependencies

**Usage:**

```bash
# Run all checks
./lintfix.py all

# Format only
./lintfix.py format

# Lint specific file
./lintfix.py lint --file app.py

# Run tests
./lintfix.py test
```

### 5. `mdtodo.py` - TODO Scanner

Scans repositories for TODO comments and task files.

**Usage:**

```bash
# Scan for TODO comments
./mdtodo.py todo

# Show tasks from YAML
./mdtodo.py tasks --file tasks.yaml
```

### 6. `daily_flow.py` - MCP-Powered Daily Workflow Agent 🌟

Comprehensive daily workflow automation using Model Context Protocol.

**Features:**

- **MCP Integration**: Connects to multiple data sources via MCP servers
- **Unified View**: Aggregates data from filesystem, git, GitHub, calendar
- **Intelligent Reports**: AI-powered insights and recommendations
- **Standup Automation**: Generate daily standup reports instantly
- **Blocker Analysis**: Identify and resolve workflow blockers
- **Multiple Formats**: Export as markdown, JSON, or terminal display

**MCP Servers Used:**

- Filesystem MCP - For notes and documents
- Git MCP - For repository activity
- GitHub MCP - For PRs and issues
- Extensible to Slack, Calendar, Obsidian, etc.

**Usage:**

```bash
# Generate daily standup
./daily_flow.py standup

# Weekly review
./daily_flow.py review --days 7

# Get today's focus items
./daily_flow.py focus

# Analyze blockers
./daily_flow.py blockers

# Export report as markdown
./daily_flow.py export --format markdown --output standup.md

# Check MCP server status
./daily_flow.py config
```

**Setup MCP Servers:**

```bash
# Install MCP servers (one-time)
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-git

# Optional: Set GitHub token for GitHub integration
export GITHUB_TOKEN="your-github-token"
```

### 7. `doctype.py` - Docstring and Type Hint Generator

Automatically adds comprehensive docstrings and type hints to Python code.

**Features:**

- Analyzes code to understand function signatures and logic
- Generates docstrings in multiple styles (NumPy, Google, Sphinx, PEP257)
- Adds mypy-compatible type hints
- Batch processes entire modules or directories
- Validates type hints with mypy

**Usage:**

```bash
# Add docstrings and types (NumPy style by default)
./doctype.py add file.py

# Use Google-style docstrings
./doctype.py add src/ --style google

# Check documentation coverage
./doctype.py check . --verbose

# Add only type hints
./doctype.py fix-types file.py

# Validate types with mypy
./doctype.py validate src/

# Show configuration and style examples
./doctype.py config
```

## 🔧 Prerequisites

### Install uv

The only requirement is `uv`, a fast Python package and project manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install GitHub CLI (for pr_review.py)

```bash
# macOS
brew install gh

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login
```

### Configure AI Models

Most agents use LLM libraries. Install and configure your preferred model:

```bash
# Install Gemini support (recommended, free tier available)
uv tool install llm-gemini

# Set API key
export GEMINI_API_KEY="your-api-key"

# Or use other models like GPT-4, Claude, etc.
uv tool install llm-openai
export OPENAI_API_KEY="your-api-key"
```

## 💡 Best Practices for Creating Micro-Agents

### 1. Keep It Focused

Each micro-agent should do one thing well. If it's getting complex, split it.

### 2. Use Modern Python Features

- Type hints for clarity
- Dataclasses for data structures
- Async/await where appropriate
- Pattern matching (Python 3.10+)

### 3. Provide Great UX

- Use `typer` for CLI with automatic help
- Use `rich` for beautiful output
- Show progress for long operations
- Provide clear error messages

### 4. Make It Executable

```bash
chmod +x my_agent.py
```

### 5. Document Inline

Use clear docstrings and inline comments. The script should be self-documenting.

### 6. Handle Errors Gracefully

```python
try:
    result = risky_operation()
except SpecificError as e:
    console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(1)
```

## 🎯 Creating Your Own Micro-Agent

Here's a template to get started:

```python
#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "httpx>=0.27",  # Add your deps here
# ]
# ///
"""
my-agent: Description of what your agent does.

Usage:
    ./my_agent.py command --option value
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

@app.command()
def hello(
    name: str = typer.Argument("World", help="Name to greet"),
    excited: bool = typer.Option(False, "--excited", "-e", help="Add excitement"),
) -> None:
    """Greet someone."""
    greeting = f"Hello, {name}{'!' if excited else '.'}"
    console.print(Panel(greeting, title="Greeting", border_style="green"))

@app.command()
def goodbye() -> None:
    """Say goodbye."""
    console.print("[yellow]Goodbye![/yellow]")

if __name__ == "__main__":
    app()
```

Make it executable and run:

```bash
chmod +x my_agent.py
./my_agent.py hello --excited Alice
```

## 🤝 Contributing

Feel free to contribute new micro-agents! Follow these guidelines:

1. One script = one micro-agent
2. Use PEP 723 inline dependencies
3. Include comprehensive docstrings
4. Add your agent to this README
5. Test on Python 3.11+

## 📚 Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 723 - Inline script metadata](https://peps.python.org/pep-0723/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [LLM Documentation](https://llm.datasette.io/)
- [GitHub CLI](https://cli.github.com/)

## 📄 License

These micro-agents are provided as examples and templates. Feel free to use, modify, and distribute them as needed for your projects.

## 🚦 Quick Start

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository:

   ```bash
   git clone <repository-url>
   cd tools
   ```

3. Make scripts executable:

   ```bash
   chmod +x *.py
   ```

4. Run any micro-agent:

   ```bash
   ./dev_assist.py init my-awesome-project
   ./code_review.py review .
   ./pr_review.py check
   ```

## 🎨 Why Micro-Agents?

Traditional development tools often require complex installation, configuration, and maintenance. Micro-agents flip this model:

- **Zero Configuration**: Just run the script
- **Self-Contained**: Dependencies are declared inline
- **Portable**: Copy a single file to any machine with uv
- **Transparent**: Read the source to understand exactly what it does
- **Composable**: Chain micro-agents together in scripts or workflows
- **AI-Enhanced**: Leverage LLMs for intelligent assistance

This approach is perfect for:

- Quick automation tasks
- CI/CD pipelines
- Development utilities
- Team productivity tools
- Learning and experimentation

Start with the existing micro-agents, then create your own to solve your specific challenges!
