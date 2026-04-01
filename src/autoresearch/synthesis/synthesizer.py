"""Multi-source synthesis engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.synthesis.synthesizer")


class Synthesizer:
    """Synthesizes findings from multiple sources into coherent reports."""

    def synthesize(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]] = None,
        format: str = "markdown",
    ) -> str:
        """Generate a synthesized report from research findings."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if format == "markdown":
            return self._markdown_report(query, findings, analysis, timestamp)
        elif format == "json":
            return self._json_report(query, findings, analysis, timestamp)
        elif format == "html":
            return self._html_report(query, findings, analysis, timestamp)
        else:
            return self._markdown_report(query, findings, analysis, timestamp)

    @staticmethod
    def _markdown_report(
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        timestamp: str,
    ) -> str:
        report = f"# Research Report: {query}\n\n"
        report += f"**Generated:** {timestamp}\n\n"
        report += "## Executive Summary\n\n"
        report += f"This report presents findings on: {query}\n\n"

        if findings:
            report += "## Key Findings\n\n"
            for i, finding in enumerate(findings, 1):
                title = finding.get("title", f"Finding {i}")
                content = finding.get("content", finding.get("snippet", "No content"))
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

    @staticmethod
    def _json_report(
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        timestamp: str,
    ) -> str:
        return json.dumps(
            {
                "query": query,
                "timestamp": timestamp,
                "findings": findings,
                "analysis": analysis,
            },
            indent=2,
        )

    @staticmethod
    def _html_report(
        query: str,
        findings: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        timestamp: str,
    ) -> str:
        html = f"""<!DOCTYPE html>
<html><head><title>Research: {query}</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px}}
h1{{border-bottom:2px solid #333}}.source{{color:#666;font-size:0.9em}}</style></head>
<body>
<h1>Research Report: {query}</h1>
<p><strong>Generated:</strong> {timestamp}</p>
<h2>Executive Summary</h2>
<p>This report presents findings on: {query}</p>
<h2>Key Findings</h2>
"""
        for i, finding in enumerate(findings, 1):
            title = finding.get("title", f"Finding {i}")
            content = finding.get("content", finding.get("snippet", ""))
            url = finding.get("url", "")
            html += f"<h3>{i}. {title}</h3><p>{content}</p>"
            if url:
                html += f'<p class="source">Source: <a href="{url}">{url}</a></p>'

        html += "</body></html>"
        return html
