"""KV cache with transparent TurboQuant compression."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .compressor import KVCacheCompressor

logger = logging.getLogger("autoresearch.turboquant.cache")


class CompressedKVCache:
    """KV cache that transparently compresses entries using TurboQuant."""

    def __init__(
        self,
        method: str = "turboquant",
        target_bits: float = 4.5,
        block_size: int = 128,
        fidelity_target: float = 0.99,
    ):
        self.compressor = KVCacheCompressor(
            method=method,
            target_bits=target_bits,
            block_size=block_size,
            fidelity_target=fidelity_target,
        )
        self._entries: Dict[str, Any] = {}
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0, "compressions": 0}

    def put(self, key: str, kv_tensor: Any) -> None:
        compressed = self.compressor.compress(kv_tensor)
        self._entries[key] = compressed
        self._stats["compressions"] += 1

    def get(self, key: str) -> Optional[Any]:
        if key in self._entries:
            self._stats["hits"] += 1
            return self.compressor.decompress(self._entries[key])
        self._stats["misses"] += 1
        return None

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(total, 1)
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 3),
            "compression": self.compressor.get_stats(),
        }
