"""DuckDuckGo search adapter."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.search.duckduckgo")


class DuckDuckGoSearch:
    """Search engine adapter for DuckDuckGo."""

    def __init__(self, max_results: int = 20, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("url", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]
        except ImportError:
            logger.error("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []
