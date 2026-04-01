"""Research state persistence and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.core.state")


@dataclass
class ResearchState:
    """Persistent state for a research session."""

    query: str = ""
    status: str = "idle"
    started_at: float = 0.0
    completed_at: float = 0.0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    report: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "findings": self.findings,
            "analysis": self.analysis,
            "report": self.report,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StateManager:
    """Manages research state persistence."""

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = (
            Path(state_dir) if state_dir else Path.home() / ".autoresearch" / "state"
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: ResearchState, session_id: str = "current") -> Path:
        """Save research state to disk."""
        path = self.state_dir / f"{session_id}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2, default=str))
        logger.debug(f"State saved: {path}")
        return path

    def load(self, session_id: str = "current") -> Optional[ResearchState]:
        """Load research state from disk."""
        path = self.state_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ResearchState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        return [p.stem for p in self.state_dir.glob("*.json")]

    def delete(self, session_id: str) -> bool:
        """Delete a saved session."""
        path = self.state_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


class CacheManager:
    """Simple file-based cache for search results and responses."""

    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 2048):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".autoresearch" / "cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb

    def _key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        path = self.cache_dir / self._key(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("expires", 0) < time.time():
                path.unlink()
                return None
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Cache a value with TTL in seconds."""
        path = self.cache_dir / self._key(key)
        data = {
            "value": value,
            "expires": time.time() + ttl,
            "created": time.time(),
        }
        path.write_text(json.dumps(data, default=str))
        self._enforce_size_limit()

    def clear(self) -> None:
        """Clear all cached data."""
        for path in self.cache_dir.glob("*"):
            path.unlink()

    def _enforce_size_limit(self) -> None:
        """Remove oldest files if cache exceeds size limit."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*"))
        max_bytes = self.max_size_mb * 1024 * 1024

        if total_size > max_bytes:
            files = sorted(self.cache_dir.glob("*"), key=lambda f: f.stat().st_mtime)
            while total_size > max_bytes * 0.8 and files:
                oldest = files.pop(0)
                total_size -= oldest.stat().st_size
                oldest.unlink()
