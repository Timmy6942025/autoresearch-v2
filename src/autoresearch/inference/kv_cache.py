"""KV cache management for MLX inference."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("autoresearch.inference.kv_cache")


class KVCacheManager:
    """Manages KV cache for inference with optional compression."""

    def __init__(
        self,
        max_tokens: int = 32768,
        turboquant: bool = False,
        block_size: int = 128,
        target_bits: float = 4.5,
    ):
        self.max_tokens = max_tokens
        self.turboquant = turboquant
        self.block_size = block_size
        self.target_bits = target_bits
        self._cache: Dict[str, Any] = {}
        self._compressed: Dict[str, Any] = {}

    def store(self, key: str, kv_tensor: Any) -> None:
        if self.turboquant:
            try:
                from ..turboquant.compressor import KVCacheCompressor

                compressor = KVCacheCompressor(
                    method="turboquant",
                    target_bits=self.target_bits,
                    block_size=self.block_size,
                )
                self._compressed[key] = compressor.compress(kv_tensor)
            except ImportError:
                logger.warning("turboquant not available, storing uncompressed")
                self._cache[key] = kv_tensor
        else:
            self._cache[key] = kv_tensor

    def retrieve(self, key: str) -> Optional[Any]:
        if key in self._compressed:
            try:
                from ..turboquant.compressor import KVCacheCompressor

                compressor = KVCacheCompressor(
                    method="turboquant",
                    target_bits=self.target_bits,
                    block_size=self.block_size,
                )
                return compressor.decompress(self._compressed[key])
            except ImportError:
                return None
        return self._cache.get(key)

    def clear(self) -> None:
        self._cache.clear()
        self._compressed.clear()

    @property
    def size(self) -> int:
        return len(self._cache) + len(self._compressed)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "uncompressed_entries": len(self._cache),
            "compressed_entries": len(self._compressed),
            "turboquant_enabled": self.turboquant,
            "block_size": self.block_size,
            "target_bits": self.target_bits,
        }
