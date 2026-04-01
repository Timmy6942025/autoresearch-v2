"""Writer agent — report synthesis and markdown generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import Agent, Message, Tool, ToolResult

logger = logging.getLogger("autoresearch.agents.writer")


class ReportGeneratorTool(Tool):
    """Generate structured research reports."""

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return "Generate a structured research report from findings."

    async def execute(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]] = None,
        format: str = "markdown",
    ) -> ToolResult:
        try:
            report = self._build_report(query, findings, analysis, format)
            return ToolResult(tool_name=self.name, result=report)
        except Exception as e:
            return ToolResult(tool_name=self.name, result=None, error=str(e))

    @staticmethod
    def _build_report(
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        format: str,
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if format == "markdown":
            report = f"# Research Report: {query}\n\n"
            report += f"**Generated:** {timestamp}\n\n"
            report += "## Executive Summary\n\n"
            report += f"This report presents findings on: {query}\n\n"

            if findings:
                report += "## Key Findings\n\n"
                for i, finding in enumerate(findings, 1):
                    title = finding.get("title", f"Finding {i}")
                    content = finding.get(
                        "content", finding.get("snippet", "No content")
                    )
                    url = finding.get("url", "")
                    report += f"### {i}. {title}\n\n"
                    report += f"{content}\n\n"
                    if url:
                        report += f"**Source:** {url}\n\n"

            if analysis:
                report += "## Analysis\n\n"
                report += json.dumps(analysis, indent=2) + "\n\n"

            report += "## Sources\n\n"
            for i, finding in enumerate(findings, 1):
                url = finding.get("url", "")
                title = finding.get("title", f"Source {i}")
                if url:
                    report += f"{i}. [{title}]({url})\n"
                else:
                    report += f"{i}. {title}\n"

            return report

        elif format == "json":
            return json.dumps(
                {
                    "query": query,
                    "timestamp": timestamp,
                    "findings": findings,
                    "analysis": analysis,
                },
                indent=2,
            )

        else:
            return f"Unsupported format: {format}"


class WriterAgent(Agent):
    """Synthesizes research findings into comprehensive reports."""

    def __init__(self):
        tools = [ReportGeneratorTool()]
        system_prompt = """You are a report synthesis agent. Your job is to create comprehensive research reports.

When given research findings:
1. Organize findings logically
2. Write clear executive summary
3. Present key findings with sources
4. Include analysis if available
5. Format as markdown or JSON

Output format: Structured report with sections for summary, findings, analysis, and sources
"""
        super().__init__(name="writer", system_prompt=system_prompt, tools=tools)

    async def process(self, message: Message) -> Message:
        self.add_message(message)

        try:
            data = json.loads(message.content)
        except json.JSONDecodeError:
            data = {"query": message.content, "findings": [], "format": "markdown"}

        query = data.get("query", "Research Topic")
        findings = data.get("findings", [])
        analysis = data.get("analysis")
        format = data.get("format", "markdown")

        result = await self.use_tool(
            "generate_report",
            query=query,
            findings=findings,
            analysis=analysis,
            format=format,
        )

        if result.error:
            return Message(
                role="assistant", content=f"Report generation error: {result.error}"
            )

        return Message(role="assistant", content=result.result)
