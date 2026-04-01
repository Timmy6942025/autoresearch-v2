"""Output formatting for research reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ReportFormatter:
    """Formats research reports for different output types."""

    @staticmethod
    def save_markdown(content: str, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    @staticmethod
    def save_html(content: str, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    @staticmethod
    def save_json(data: str, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        return p

    @staticmethod
    def markdown_to_html(markdown: str) -> str:
        """Convert markdown to basic HTML."""
        html = "<!DOCTYPE html>\n<html><head><meta charset='utf-8'></head><body>\n"
        for line in markdown.split("\n"):
            if line.startswith("# "):
                html += f"<h1>{line[2:]}</h1>\n"
            elif line.startswith("## "):
                html += f"<h2>{line[3:]}</h2>\n"
            elif line.startswith("### "):
                html += f"<h3>{line[4:]}</h3>\n"
            elif line.startswith("- "):
                html += f"<li>{line[2:]}</li>\n"
            elif line.startswith("**") and line.endswith("**"):
                html += f"<p><strong>{line[2:-2]}</strong></p>\n"
            elif line.strip():
                html += f"<p>{line}</p>\n"
            else:
                html += "\n"
        html += "</body></html>"
        return html
