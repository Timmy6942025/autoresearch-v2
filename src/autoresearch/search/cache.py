"""Search result caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SearchCache:
    """Cache for search results and fetched content."""

    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".autoresearch" / "search_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        path = self.cache_dir / self._key(query)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("expires", 0) < time.time():
                path.unlink()
                return None
            return data.get("results")
        except Exception:
            return None

    def set(self, query: str, results: Any) -> None:
        path = self.cache_dir / self._key(query)
        data = {"results": results, "expires": time.time() + self.ttl, "query": query}
        path.write_text(json.dumps(data, default=str))

    def clear(self) -> None:
        for path in self.cache_dir.glob("*"):
            path.unlink()
