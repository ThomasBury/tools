#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "rich>=13.7",
#   "mcp>=1.0",
#   "httpx>=0.27",
#   "pydantic>=2.0",
#   "llm>=0.26",
#   "llm-gemini>=0.24",
#   "python-dateutil>=2.8",
#   "aiofiles>=23.0",
# ]
# ///
"""
MCP-powered daily workflow and standup assistant.

This micro-agent integrates with multiple data sources via MCP to provide
comprehensive daily reviews, standup reports, and workflow insights.

Features
--------
- Aggregates data from filesystem, git, GitHub, calendar, and task systems
- Generates intelligent daily standup reports
- Tracks progress on goals and blockers
- Provides actionable insights and priority recommendations
- Exports reports in multiple formats

Examples
--------
Generate daily standup report:

>>> ./daily_flow.py standup

Weekly review:

>>> ./daily_flow.py review --days 7

Get focus recommendations:

>>> ./daily_flow.py focus

Identify blockers:

>>> ./daily_flow.py blockers

Export report in markdown format:

>>> ./daily_flow.py export --format markdown
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess as sp
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import httpx
import llm
import typer
from dateutil import parser as date_parser
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Configuration
DEFAULT_MODEL = os.environ.get("DAILY_FLOW_MODEL", "gemini-2.5-flash-lite")
DEFAULT_WORKSPACE = Path.home() / "workspace"
DEFAULT_NOTES_DIR = Path.home() / "notes"


class DataSource(str, Enum):
    """Enumeration of supported data sources for workflow collection.

    Attributes
    ----------
    FILESYSTEM : str
        Local filesystem and notes directory data source.
    GIT : str
        Git repository commit and status information.
    GITHUB : str
        GitHub pull requests and issues.
    CALENDAR : str
        Calendar events and scheduling data.
    OBSIDIAN : str
        Obsidian notes and knowledge base.
    SLACK : str
        Slack messages and team communication.
    """
    FILESYSTEM = "filesystem"
    GIT = "git"
    GITHUB = "github"
    CALENDAR = "calendar"
    OBSIDIAN = "obsidian"
    SLACK = "slack"


class ReportFormat(str, Enum):
    """Enumeration of supported report output formats.

    Attributes
    ----------
    TERMINAL : str
        Rich-formatted terminal display with colors and tables.
    MARKDOWN : str
        Markdown format suitable for documentation and sharing.
    JSON : str
        Structured JSON format for programmatic access.
    HTML : str
        HTML format for web display and reports.
    """
    TERMINAL = "terminal"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


@dataclass
class WorkItem:
    """Represents a unit of work from various data sources.

    Attributes
    ----------
    title : str
        Brief title or name of the work item.
    description : Optional[str]
        Detailed description of the work item.
    source : DataSource
        The data source this item originated from.
    timestamp : Optional[datetime]
        When this work item was created or last modified.
    status : str
        Current status (e.g., 'completed', 'in_progress', 'blocked').
    tags : List[str]
        List of tags for categorization and filtering.
    priority : Optional[int]
        Priority level (lower numbers indicate higher priority).
    url : Optional[str]
        URL link to the original item (e.g., GitHub PR/issue).
    metadata : Dict[str, Any]
        Additional source-specific metadata.
    """
    title: str
    description: Optional[str]
    source: DataSource
    timestamp: Optional[datetime]
    status: str
    tags: List[str] = field(default_factory=list)
    priority: Optional[int] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyReport:
    """Complete daily workflow report with categorized work items and insights.

    Attributes
    ----------
    date : datetime
        Date this report covers.
    completed_items : List[WorkItem]
        Work items that have been completed.
    in_progress_items : List[WorkItem]
        Work items currently being worked on.
    blocked_items : List[WorkItem]
        Work items that are blocked or waiting.
    upcoming_items : List[WorkItem]
        Work items planned for the future.
    insights : List[str]
        AI-generated insights about the workflow.
    recommendations : List[str]
        Actionable recommendations for improvement.
    metrics : Dict[str, Any]
        Productivity metrics and statistics.
    """
    date: datetime
    completed_items: List[WorkItem]
    in_progress_items: List[WorkItem]
    blocked_items: List[WorkItem]
    upcoming_items: List[WorkItem]
    insights: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]


class MCPServer:
    """Configuration for an MCP server.

    Parameters
    ----------
    name : str
        Name identifier for the MCP server.
    command : Optional[str], default=None
        Command to execute for the MCP server process.
    args : Optional[List[str]], default=None
        Arguments to pass to the MCP server command.

    Attributes
    ----------
    session : Optional[ClientSession]
        Active MCP client session when connected.
    """
    def __init__(self, name: str, command: Optional[str] = None, args: Optional[List[str]] = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._context = None
    
    async def connect(self) -> bool:
        """Connect to the MCP server.

        Establishes a connection to the configured MCP server using stdio
        communication. Initializes the client session and performs handshake.

        Returns
        -------
        bool
            True if connection was successful, False otherwise.

        Raises
        ------
        Exception
            Connection or initialization failures are caught and logged,
            returning False instead of raising.
        """
        try:
            if self.command:
                params = StdioServerParameters(
                    command=self.command,
                    args=self.args
                )
                self._context = stdio_client(params)
                self._read, self._write = await self._context.__aenter__()
            else:
                return False

            self.session = ClientSession(self._read, self._write)
            await self.session.__aenter__()
            await self.session.initialize()
            return True
        except Exception as e:
            console.print(f"[yellow]Warning: Could not connect to {self.name}: {e}[/yellow]")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        Properly closes the client session and stdio context.
        Silently handles any cleanup errors.
        """
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass
        if self._context:
            try:
                await self._context.__aexit__(None, None, None)
            except:
                pass
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Call a tool on the MCP server.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute on the server.
        arguments : Dict[str, Any]
            Arguments to pass to the tool.

        Returns
        -------
        Optional[Any]
            Tool execution result, or None if failed or no session.
            For list content, text fields are concatenated.

        Raises
        ------
        Exception
            Tool execution errors are caught and logged, returning None.
        """
        if not self.session:
            return None

        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            if result.content:
                # Handle different content types
                if isinstance(result.content, list):
                    # Combine text content
                    texts = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            texts.append(item.text)
                    return '\n'.join(texts) if texts else None
                return result.content
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Tool {tool_name} failed: {e}[/yellow]")
            return None


class DataCollector:
    """Collects data from various MCP sources.

    Attributes
    ----------
    servers : Dict[str, MCPServer]
        Dictionary of configured MCP servers by name.
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._setup_servers()
    
    def _setup_servers(self) -> None:
        """Setup MCP server configurations.

        Configures filesystem, git, and optionally GitHub MCP servers
        based on environment and default paths.
        """
        # Filesystem MCP for notes and documents
        self.servers['filesystem'] = MCPServer(
            'filesystem',
            'npx',
            ['-y', '@modelcontextprotocol/server-filesystem', str(DEFAULT_NOTES_DIR)]
        )

        # Git MCP for repository information
        self.servers['git'] = MCPServer(
            'git',
            'npx',
            ['-y', '@modelcontextprotocol/server-git', str(DEFAULT_WORKSPACE)]
        )

        # GitHub MCP (if configured)
        if github_token := os.environ.get('GITHUB_TOKEN'):
            self.servers['github'] = MCPServer(
                'github',
                'npx',
                ['-y', '@modelcontextprotocol/server-github']
            )
    
    async def connect_all(self) -> None:
        """Connect to all configured MCP servers.

        Attempts to establish connections to all servers in parallel.
        Reports the number of successful connections.
        """
        tasks = [server.connect() for server in self.servers.values()]
        results = await asyncio.gather(*tasks)

        connected = sum(1 for r in results if r)
        console.print(f"[green]Connected to {connected}/{len(self.servers)} MCP servers[/green]")
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers.

        Closes all active server connections in parallel.
        """
        tasks = [server.disconnect() for server in self.servers.values()]
        await asyncio.gather(*tasks)
    
    async def collect_filesystem_data(self, days_back: int = 1) -> List[WorkItem]:
        """Collect data from filesystem (notes, documents).

        Parameters
        ----------
        days_back : int, default=1
            Number of days to look back for recent file changes.

        Returns
        -------
        List[WorkItem]
            List of work items representing recent note updates.
        """
        items = []
        server = self.servers.get('filesystem')
        if not server or not server.session:
            return items

        # Get recent files
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        # List recent markdown files (notes)
        result = await server.call_tool('list_directory', {'path': '.'})
        if result:
            try:
                files = json.loads(result) if isinstance(result, str) else result
                for file_info in files:
                    if file_info.get('name', '').endswith('.md'):
                        # Read file metadata
                        file_path = file_info.get('path', file_info.get('name'))
                        stat_result = await server.call_tool('get_file_info', {'path': file_path})

                        if stat_result:
                            # Parse modification time
                            items.append(WorkItem(
                                title=f"Note: {Path(file_path).stem}",
                                description=f"Updated note in {file_path}",
                                source=DataSource.FILESYSTEM,
                                timestamp=datetime.now(timezone.utc),  # Would parse actual mtime
                                status='updated',
                                tags=['notes', 'documentation']
                            ))
            except Exception as e:
                console.print(f"[yellow]Warning: Error parsing filesystem data: {e}[/yellow]")

        return items
    
    async def collect_git_data(self, days_back: int = 1) -> List[WorkItem]:
        """Collect data from git repositories.

        Parameters
        ----------
        days_back : int, default=1
            Number of days to look back for commits (currently unused).

        Returns
        -------
        List[WorkItem]
            List of work items from recent commits and repository status.
        """
        items = []
        server = self.servers.get('git')
        if not server or not server.session:
            return items

        # Get recent commits
        result = await server.call_tool('git_log', {'max_count': 20})
        if result:
            try:
                # Parse git log output
                for line in result.split('\n'):
                    if line.strip():
                        # Simple parsing - in production would be more robust
                        items.append(WorkItem(
                            title=f"Commit: {line[:50]}",
                            description=line,
                            source=DataSource.GIT,
                            timestamp=datetime.now(timezone.utc),
                            status='completed',
                            tags=['git', 'code']
                        ))
            except Exception as e:
                console.print(f"[yellow]Warning: Error parsing git data: {e}[/yellow]")

        # Get current branch status
        branch_result = await server.call_tool('git_status', {})
        if branch_result:
            if 'modified' in branch_result.lower() or 'untracked' in branch_result.lower():
                items.append(WorkItem(
                    title="Uncommitted changes",
                    description="You have uncommitted changes in your repository",
                    source=DataSource.GIT,
                    timestamp=datetime.now(timezone.utc),
                    status='in_progress',
                    tags=['git', 'wip']
                ))

        return items
    
    async def collect_github_data(self, days_back: int = 1) -> List[WorkItem]:
        """Collect data from GitHub.

        Parameters
        ----------
        days_back : int, default=1
            Number of days to look back (currently unused, fetches recent items).

        Returns
        -------
        List[WorkItem]
            List of work items from pull requests and issues.
        """
        items = []

        # Use gh CLI as fallback if MCP server not available
        try:
            # Get user's recent PRs
            result = sp.run(
                ['gh', 'pr', 'list', '--author', '@me', '--json',
                 'number,title,state,createdAt,url', '--limit', '10'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                prs = json.loads(result.stdout)
                for pr in prs:
                    items.append(WorkItem(
                        title=f"PR #{pr['number']}: {pr['title']}",
                        description=pr['title'],
                        source=DataSource.GITHUB,
                        timestamp=date_parser.parse(pr['createdAt']),
                        status=pr['state'].lower(),
                        tags=['github', 'pr'],
                        url=pr['url']
                    ))

            # Get assigned issues
            result = sp.run(
                ['gh', 'issue', 'list', '--assignee', '@me', '--json',
                 'number,title,state,createdAt,url', '--limit', '10'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                issues = json.loads(result.stdout)
                for issue in issues:
                    items.append(WorkItem(
                        title=f"Issue #{issue['number']}: {issue['title']}",
                        description=issue['title'],
                        source=DataSource.GITHUB,
                        timestamp=date_parser.parse(issue['createdAt']),
                        status=issue['state'].lower(),
                        tags=['github', 'issue'],
                        url=issue['url']
                    ))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch GitHub data: {e}[/yellow]")

        return items
    
    async def collect_all_data(self, days_back: int = 1) -> List[WorkItem]:
        """Collect data from all available sources.

        Parameters
        ----------
        days_back : int, default=1
            Number of days to look back for data collection.

        Returns
        -------
        List[WorkItem]
            Combined list of work items from all sources.
        """
        tasks = [
            self.collect_filesystem_data(days_back),
            self.collect_git_data(days_back),
            self.collect_github_data(days_back),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                console.print(f"[yellow]Warning: Data collection error: {result}[/yellow]")

        return all_items


class ReportGenerator:
    """Generates intelligent reports from collected data.

    Parameters
    ----------
    model_name : str, default=DEFAULT_MODEL
        Name of the LLM model to use for AI-powered insights.

    Attributes
    ----------
    model_name : str
        The configured LLM model name.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
    
    def categorize_items(self, items: List[WorkItem]) -> Tuple[List[WorkItem], List[WorkItem], List[WorkItem], List[WorkItem]]:
        """Categorize work items by status.

        Parameters
        ----------
        items : List[WorkItem]
            List of work items to categorize.

        Returns
        -------
        Tuple[List[WorkItem], List[WorkItem], List[WorkItem], List[WorkItem]]
            Tuple of (completed, in_progress, blocked, upcoming) items.
        """
        completed = []
        in_progress = []
        blocked = []
        upcoming = []

        for item in items:
            if item.status in ['completed', 'closed', 'merged']:
                completed.append(item)
            elif item.status in ['in_progress', 'open', 'active']:
                in_progress.append(item)
            elif item.status in ['blocked', 'waiting']:
                blocked.append(item)
            else:
                upcoming.append(item)

        return completed, in_progress, blocked, upcoming
    
    def calculate_metrics(self, items: List[WorkItem]) -> Dict[str, Any]:
        """Calculate productivity metrics.

        Parameters
        ----------
        items : List[WorkItem]
            List of work items to analyze.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metrics like total_items, completed,
            completion_rate, by_source, and by_tag.
        """
        total = len(items)
        by_source = {}
        by_tag = {}

        for item in items:
            # Count by source
            source_key = item.source.value
            by_source[source_key] = by_source.get(source_key, 0) + 1

            # Count by tag
            for tag in item.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        completed_count = sum(1 for i in items if i.status in ['completed', 'closed', 'merged'])

        return {
            'total_items': total,
            'completed': completed_count,
            'completion_rate': (completed_count / total * 100) if total > 0 else 0,
            'by_source': by_source,
            'by_tag': by_tag
        }
    
    async def generate_insights(self, report: DailyReport) -> List[str]:
        """Generate AI-powered insights from the report.

        Parameters
        ----------
        report : DailyReport
            The daily report to analyze for insights.

        Returns
        -------
        List[str]
            List of insight strings, combining basic heuristics and AI analysis.
        """
        insights = []

        # Basic insights without AI
        metrics = report.metrics

        if metrics['completion_rate'] > 80:
            insights.append("🎯 Excellent completion rate! You're being highly productive.")
        elif metrics['completion_rate'] < 30:
            insights.append("⚠️ Low completion rate. Consider focusing on fewer tasks.")

        if len(report.blocked_items) > 0:
            insights.append(f"🚧 You have {len(report.blocked_items)} blocked items that need attention.")

        if len(report.in_progress_items) > 5:
            insights.append("📊 High WIP count. Consider completing current tasks before starting new ones.")

        # AI-powered insights if model available
        try:
            model = llm.get_model(self.model_name)

            # Build context for AI
            context = f"""
            Daily workflow analysis:
            - Completed: {len(report.completed_items)} items
            - In Progress: {len(report.in_progress_items)} items
            - Blocked: {len(report.blocked_items)} items
            - Sources: {', '.join(metrics['by_source'].keys())}

            Completed items:
            {chr(10).join([f"- {item.title}" for item in report.completed_items[:5]])}

            In progress:
            {chr(10).join([f"- {item.title}" for item in report.in_progress_items[:5]])}

            Provide 2-3 concise, actionable insights about this workflow.
            """

            response = model.prompt(context)
            ai_insights = response.text().split('\n')
            insights.extend([i.strip() for i in ai_insights if i.strip()][:3])
        except:
            pass  # Fall back to basic insights

        return insights
    
    async def generate_recommendations(self, report: DailyReport) -> List[str]:
        """Generate actionable recommendations.

        Parameters
        ----------
        report : DailyReport
            The daily report to analyze for recommendations.

        Returns
        -------
        List[str]
            List of recommendation strings for next actions.
        """
        recommendations = []

        # Priority recommendations
        if report.blocked_items:
            recommendations.append(f"Unblock: {report.blocked_items[0].title}")

        if report.in_progress_items:
            # Find oldest in-progress item
            oldest = min(report.in_progress_items,
                        key=lambda x: x.timestamp or datetime.now(timezone.utc))
            recommendations.append(f"Complete: {oldest.title}")

        # Balance recommendations
        by_source = report.metrics['by_source']
        if 'git' in by_source and by_source['git'] == 0:
            recommendations.append("Commit your code changes")

        if 'github' in by_source and 'pr' in report.metrics['by_tag']:
            recommendations.append("Review and merge pending PRs")

        return recommendations
    
    async def create_report(self, items: List[WorkItem], date: datetime) -> DailyReport:
        """Create a comprehensive daily report.

        Parameters
        ----------
        items : List[WorkItem]
            List of work items to include in the report.
        date : datetime
            Date for the report.

        Returns
        -------
        DailyReport
            Complete report with categorized items, metrics, insights, and recommendations.
        """
        completed, in_progress, blocked, upcoming = self.categorize_items(items)
        metrics = self.calculate_metrics(items)

        report = DailyReport(
            date=date,
            completed_items=completed,
            in_progress_items=in_progress,
            blocked_items=blocked,
            upcoming_items=upcoming,
            insights=[],
            recommendations=[],
            metrics=metrics
        )

        report.insights = await self.generate_insights(report)
        report.recommendations = await self.generate_recommendations(report)

        return report


class ReportFormatter:
    """Formats reports for different outputs."""
    
    @staticmethod
    def format_terminal(report: DailyReport) -> None:
        """Format report for terminal display.

        Parameters
        ----------
        report : DailyReport
            The report to format and display in the terminal.
        """
        # Header
        console.print(Panel.fit(
            f"[bold]Daily Workflow Report[/bold]\n"
            f"Date: {report.date.strftime('%Y-%m-%d')}",
            border_style="cyan"
        ))

        # Metrics summary
        metrics_table = Table(title="Metrics", show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")

        metrics_table.add_row("Total Items", str(report.metrics['total_items']))
        metrics_table.add_row("Completed", str(report.metrics['completed']))
        metrics_table.add_row("Completion Rate", f"{report.metrics['completion_rate']:.1f}%")

        console.print(metrics_table)
        console.print()

        # Work items by category
        if report.completed_items:
            console.print("[bold green]✅ Completed[/bold green]")
            for item in report.completed_items[:5]:
                console.print(f"  • {item.title}")
            if len(report.completed_items) > 5:
                console.print(f"  ... and {len(report.completed_items) - 5} more")
            console.print()

        if report.in_progress_items:
            console.print("[bold yellow]🚧 In Progress[/bold yellow]")
            for item in report.in_progress_items[:5]:
                console.print(f"  • {item.title}")
            if len(report.in_progress_items) > 5:
                console.print(f"  ... and {len(report.in_progress_items) - 5} more")
            console.print()

        if report.blocked_items:
            console.print("[bold red]⛔ Blocked[/bold red]")
            for item in report.blocked_items:
                console.print(f"  • {item.title}")
            console.print()

        # Insights
        if report.insights:
            console.print("[bold]💡 Insights[/bold]")
            for insight in report.insights:
                console.print(f"  {insight}")
            console.print()

        # Recommendations
        if report.recommendations:
            console.print("[bold]🎯 Recommendations[/bold]")
            for i, rec in enumerate(report.recommendations, 1):
                console.print(f"  {i}. {rec}")
    
    @staticmethod
    def format_markdown(report: DailyReport) -> str:
        """Format report as markdown.

        Parameters
        ----------
        report : DailyReport
            The report to format as markdown.

        Returns
        -------
        str
            Markdown formatted report string.
        """
        lines = [
            f"# Daily Workflow Report",
            f"**Date:** {report.date.strftime('%Y-%m-%d')}",
            "",
            "## Metrics",
            f"- Total Items: {report.metrics['total_items']}",
            f"- Completed: {report.metrics['completed']}",
            f"- Completion Rate: {report.metrics['completion_rate']:.1f}%",
            ""
        ]

        if report.completed_items:
            lines.append("## ✅ Completed")
            for item in report.completed_items:
                url_suffix = f" ([link]({item.url}))" if item.url else ""
                lines.append(f"- {item.title}{url_suffix}")
            lines.append("")

        if report.in_progress_items:
            lines.append("## 🚧 In Progress")
            for item in report.in_progress_items:
                url_suffix = f" ([link]({item.url}))" if item.url else ""
                lines.append(f"- {item.title}{url_suffix}")
            lines.append("")

        if report.blocked_items:
            lines.append("## ⛔ Blocked")
            for item in report.blocked_items:
                lines.append(f"- {item.title}")
            lines.append("")

        if report.insights:
            lines.append("## 💡 Insights")
            for insight in report.insights:
                lines.append(f"- {insight}")
            lines.append("")

        if report.recommendations:
            lines.append("## 🎯 Recommendations")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)
    
    @staticmethod
    def format_json(report: DailyReport) -> str:
        """Format report as JSON.

        Parameters
        ----------
        report : DailyReport
            The report to format as JSON.

        Returns
        -------
        str
            JSON formatted report string.
        """
        data = {
            'date': report.date.isoformat(),
            'metrics': report.metrics,
            'completed': [
                {
                    'title': item.title,
                    'description': item.description,
                    'source': item.source.value,
                    'status': item.status,
                    'tags': item.tags,
                    'url': item.url
                }
                for item in report.completed_items
            ],
            'in_progress': [
                {
                    'title': item.title,
                    'source': item.source.value,
                    'status': item.status
                }
                for item in report.in_progress_items
            ],
            'blocked': [
                {
                    'title': item.title,
                    'source': item.source.value
                }
                for item in report.blocked_items
            ],
            'insights': report.insights,
            'recommendations': report.recommendations
        }
        return json.dumps(data, indent=2, default=str)


@app.command()
def standup(
    days_back: int = typer.Option(1, "--days", "-d", help="Days to look back"),
    format: ReportFormat = typer.Option(ReportFormat.TERMINAL, "--format", "-f", help="Output format"),
    export_path: Optional[Path] = typer.Option(None, "--export", "-e", help="Export to file"),
) -> None:
    """Generate daily standup report.

    Parameters
    ----------
    days_back : int, default=1
        Number of days to look back for data collection.
    format : ReportFormat, default=ReportFormat.TERMINAL
        Output format for the report.
    export_path : Optional[Path], default=None
        Path to export the report to file, if specified.
    """
    
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize collector
            task = progress.add_task("Connecting to data sources...", total=None)
            collector = DataCollector()
            await collector.connect_all()
            
            # Collect data
            progress.update(task, description="Collecting workflow data...")
            items = await collector.collect_all_data(days_back)
            
            # Generate report
            progress.update(task, description="Generating report...")
            generator = ReportGenerator()
            report = await generator.create_report(
                items,
                datetime.now(timezone.utc)
            )
            
            # Cleanup
            await collector.disconnect_all()
        
        # Format and display
        formatter = ReportFormatter()
        
        if format == ReportFormat.TERMINAL:
            formatter.format_terminal(report)
        elif format == ReportFormat.MARKDOWN:
            output = formatter.format_markdown(report)
            if export_path:
                export_path.write_text(output)
                console.print(f"[green]Report exported to {export_path}[/green]")
            else:
                console.print(Markdown(output))
        elif format == ReportFormat.JSON:
            output = formatter.format_json(report)
            if export_path:
                export_path.write_text(output)
                console.print(f"[green]Report exported to {export_path}[/green]")
            else:
                console.print(output)
    
    asyncio.run(run())


@app.command()
def review(
    days: int = typer.Option(7, "--days", "-d", help="Days to review"),
    format: ReportFormat = typer.Option(ReportFormat.TERMINAL, "--format", "-f", help="Output format"),
) -> None:
    """Generate a comprehensive review for the specified period.

    Parameters
    ----------
    days : int, default=7
        Number of days to include in the review.
    format : ReportFormat, default=ReportFormat.TERMINAL
        Output format for the report.
    """
    
    # Reuse standup with more days
    standup(days_back=days, format=format)


@app.command()
def focus() -> None:
    """Get focus recommendations for today.

    Analyzes current work items and provides prioritized recommendations
    for what to focus on in the immediate term.
    """
    
    async def run():
        # Quick focused analysis
        collector = DataCollector()
        await collector.connect_all()
        
        items = await collector.collect_all_data(1)
        
        generator = ReportGenerator()
        report = await generator.create_report(
            items,
            datetime.now(timezone.utc)
        )
        
        await collector.disconnect_all()
        
        # Display focus items
        console.print(Panel.fit(
            "[bold]Today's Focus[/bold]",
            border_style="cyan"
        ))
        
        if report.blocked_items:
            console.print("\n[red]⛔ Unblock these first:[/red]")
            for item in report.blocked_items[:3]:
                console.print(f"  • {item.title}")
        
        if report.in_progress_items:
            console.print("\n[yellow]🎯 Complete these:[/yellow]")
            for item in report.in_progress_items[:3]:
                console.print(f"  • {item.title}")
        
        if report.recommendations:
            console.print("\n[green]💡 Recommended actions:[/green]")
            for rec in report.recommendations[:3]:
                console.print(f"  • {rec}")
    
    asyncio.run(run())


@app.command()
def blockers() -> None:
    """Identify and analyze blockers.

    Scans for blocked work items and stale in-progress tasks,
    providing analysis and suggested actions to unblock progress.
    """
    
    async def run():
        collector = DataCollector()
        await collector.connect_all()
        
        items = await collector.collect_all_data(7)  # Look back a week
        
        generator = ReportGenerator()
        _, in_progress, blocked, _ = generator.categorize_items(items)
        
        await collector.disconnect_all()
        
        console.print(Panel.fit(
            "[bold]Blocker Analysis[/bold]",
            border_style="red"
        ))
        
        if not blocked and not in_progress:
            console.print("[green]✅ No blockers identified![/green]")
            return
        
        if blocked:
            console.print("\n[red]Explicitly Blocked:[/red]")
            for item in blocked:
                console.print(f"  • {item.title}")
                if item.description:
                    console.print(f"    {item.description[:100]}")
        
        # Find stale in-progress items
        stale_items = []
        for item in in_progress:
            if item.timestamp:
                age = datetime.now(timezone.utc) - item.timestamp
                if age.days > 3:
                    stale_items.append(item)
        
        if stale_items:
            console.print("\n[yellow]Potentially Stale (>3 days):[/yellow]")
            for item in stale_items:
                age_days = (datetime.now(timezone.utc) - item.timestamp).days if item.timestamp else 0
                console.print(f"  • {item.title} ({age_days} days)")
        
        # AI-powered blocker analysis
        try:
            model = llm.get_model(DEFAULT_MODEL)
            
            context = f"""
            Analyze these blocked and stale work items:
            
            Blocked: {[item.title for item in blocked]}
            Stale: {[item.title for item in stale_items]}
            
            Provide 2-3 specific actions to unblock progress.
            """
            
            response = model.prompt(context)
            
            console.print("\n[cyan]Suggested Actions:[/cyan]")
            console.print(response.text())
        except:
            pass
    
    asyncio.run(run())


@app.command()
def export(
    format: ReportFormat = typer.Option(ReportFormat.MARKDOWN, "--format", "-f", help="Export format"),
    output: Path = typer.Option(Path("daily_report"), "--output", "-o", help="Output file path"),
    days: int = typer.Option(1, "--days", "-d", help="Days to include"),
) -> None:
    """Export workflow report to file.

    Parameters
    ----------
    format : ReportFormat, default=ReportFormat.MARKDOWN
        Format for the exported report.
    output : Path, default=Path("daily_report")
        Base path for the output file (extension added automatically).
    days : int, default=1
        Number of days of data to include in the report.
    """
    
    # Determine file extension
    extensions = {
        ReportFormat.MARKDOWN: ".md",
        ReportFormat.JSON: ".json",
        ReportFormat.HTML: ".html"
    }
    
    output_file = output.with_suffix(extensions.get(format, ".txt"))
    
    # Generate and export
    standup(days_back=days, format=format, export_path=output_file)


@app.command()
def config() -> None:
    """Show configuration and MCP server status.

    Displays current settings, default paths, and tests connectivity
    to configured MCP servers. Useful for troubleshooting setup issues.
    """
    
    async def run():
        console.print(Panel.fit(
            "[bold]Daily Flow Configuration[/bold]",
            border_style="cyan"
        ))
        
        # Show configuration
        console.print("\n[bold]Settings:[/bold]")
        console.print(f"  Default Model: {DEFAULT_MODEL}")
        console.print(f"  Workspace: {DEFAULT_WORKSPACE}")
        console.print(f"  Notes Directory: {DEFAULT_NOTES_DIR}")
        
        # Test MCP connections
        console.print("\n[bold]MCP Server Status:[/bold]")
        
        collector = DataCollector()
        
        for name, server in collector.servers.items():
            connected = await server.connect()
            status = "[green]✓ Connected[/green]" if connected else "[red]✗ Not available[/red]"
            console.print(f"  {name}: {status}")
            if connected:
                await server.disconnect()
        
        # Show available tools
        console.print("\n[bold]Available Commands:[/bold]")
        console.print("  standup    - Generate daily standup report")
        console.print("  review     - Weekly/monthly review")
        console.print("  focus      - Today's focus items")
        console.print("  blockers   - Identify blockers")
        console.print("  export     - Export reports")
        
        console.print("\n[dim]Tip: Set GITHUB_TOKEN environment variable for GitHub integration[/dim]")
    
    asyncio.run(run())


if __name__ == "__main__":
    app()