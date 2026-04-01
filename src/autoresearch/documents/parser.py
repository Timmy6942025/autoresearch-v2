"""Document processing — multi-format parser (PDF, HTML, TXT)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.documents.parser")


class DocumentParser:
    """Parse documents in multiple formats (PDF, HTML, TXT)."""

    @staticmethod
    def parse(path: str | Path) -> Dict[str, Any]:
        """Parse a document and return structured content."""
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}

        ext = path.suffix.lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(path)
        elif ext in [".html", ".htm"]:
            return DocumentParser._parse_html(path)
        elif ext == ".txt":
            return DocumentParser._parse_txt(path)
        elif ext == ".md":
            return DocumentParser._parse_txt(path)
        elif ext == ".json":
            return DocumentParser._parse_json(path)
        else:
            return DocumentParser._parse_txt(path)

    @staticmethod
    def _parse_pdf(path: Path) -> Dict[str, Any]:
        """Parse PDF document."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            return {
                "path": str(path),
                "format": "pdf",
                "title": path.stem,
                "content": text,
                "page_count": len(reader.pages),
                "success": True,
            }
        except ImportError:
            return {
                "path": str(path),
                "format": "pdf",
                "content": "",
                "error": "pypdf not installed. Run: pip install pypdf",
                "success": False,
            }
        except Exception as e:
            return {
                "path": str(path),
                "format": "pdf",
                "content": "",
                "error": str(e),
                "success": False,
            }

    @staticmethod
    def _parse_html(path: Path) -> Dict[str, Any]:
        """Parse HTML document."""
        try:
            from bs4 import BeautifulSoup

            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")

            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else path.stem

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            return {
                "path": str(path),
                "format": "html",
                "title": title_text,
                "content": text,
                "success": True,
            }
        except ImportError:
            return {
                "path": str(path),
                "format": "html",
                "content": "",
                "error": "beautifulsoup4 not installed",
                "success": False,
            }
        except Exception as e:
            return {
                "path": str(path),
                "format": "html",
                "content": "",
                "error": str(e),
                "success": False,
            }

    @staticmethod
    def _parse_txt(path: Path) -> Dict[str, Any]:
        """Parse plain text document."""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {
                "path": str(path),
                "format": "text",
                "title": path.stem,
                "content": text,
                "success": True,
            }
        except Exception as e:
            return {
                "path": str(path),
                "format": "text",
                "content": "",
                "error": str(e),
                "success": False,
            }

    @staticmethod
    def _parse_json(path: Path) -> Dict[str, Any]:
        """Parse JSON document."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "path": str(path),
                "format": "json",
                "title": path.stem,
                "content": json.dumps(data, indent=2),
                "data": data,
                "success": True,
            }
        except Exception as e:
            return {
                "path": str(path),
                "format": "json",
                "content": "",
                "error": str(e),
                "success": False,
            }


class DocumentChunker:
    """Split documents into semantic chunks for processing."""

    @staticmethod
    def chunk(
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk_words = words[i : i + max_chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            i += max_chunk_size - overlap

        return chunks
