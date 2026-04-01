"""TurboQuant KV cache compression — rotation + Lloyd-Max + QJL."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("autoresearch.turboquant")


class KVCacheCompressor:
    """Compresses KV caches using rotation + Lloyd-Max + QJL pipeline."""

    def __init__(
        self,
        method: str = "turboquant",
        target_bits: float = 4.5,
        block_size: int = 128,
        fidelity_target: float = 0.99,
    ):
        self.method = method
        self.target_bits = target_bits
        self.block_size = block_size
        self.fidelity_target = fidelity_target

    def compress(self, kv_tensor: Any) -> Dict[str, Any]:
        try:
            import mlx.core as mx

            if not isinstance(kv_tensor, mx.array):
                kv_tensor = mx.array(kv_tensor)

            if self.method == "turboquant":
                return self._full_pipeline(kv_tensor)
            elif self.method == "q4":
                return self._quantize(kv_tensor, bits=4)
            elif self.method == "q6":
                return self._quantize(kv_tensor, bits=6)
            else:
                return {"tensor": kv_tensor, "method": "none", "compression": 1.0}
        except ImportError:
            return {"tensor": kv_tensor, "method": "none", "compression": 1.0}

    def decompress(self, compressed: Dict[str, Any]) -> Any:
        try:
            import mlx.core as mx

            if compressed.get("method") == "none":
                return compressed["tensor"]

            scales = compressed.get("scales", mx.array(1.0))
            indices = compressed["indices"]
            codebook = compressed.get("codebook")

            if codebook is not None:
                return codebook[indices] * scales
            return indices * scales
        except ImportError:
            return compressed.get("tensor")

    def _full_pipeline(self, tensor: Any) -> Dict[str, Any]:
        import mlx.core as mx
        from .rotation import optimize_rotation
        from .lloyd_max import lloyd_max_quantize
        from .qjl import qjl_decompose

        rotated, rotation_matrix = optimize_rotation(tensor)
        quantized, codebook, scales = lloyd_max_quantize(
            rotated, target_bits=self.target_bits, block_size=self.block_size
        )
        residual = rotated - codebook[quantized]
        low_rank = qjl_decompose(residual, rank=max(1, tensor.shape[-1] // 8))

        original_size = tensor.size * 2
        compressed_size = (
            quantized.size * (self.target_bits / 8)
            + codebook.size * 2
            + low_rank["u"].size * 2
            + low_rank["v"].size * 2
        )
        compression = original_size / max(compressed_size, 1)

        return {
            "indices": quantized,
            "codebook": codebook,
            "scales": scales,
            "low_rank": low_rank,
            "rotation": rotation_matrix,
            "method": "turboquant",
            "compression": compression,
            "target_bits": self.target_bits,
        }

    def _quantize(self, tensor: Any, bits: int) -> Dict[str, Any]:
        import mlx.core as mx

        max_val = mx.max(mx.abs(tensor))
        scale = max_val / (2 ** (bits - 1))
        quantized = mx.round(tensor / scale).astype(mx.int32)
        return {
            "indices": quantized,
            "scales": scale,
            "method": f"q{bits}",
            "compression": 16.0 / bits,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "target_bits": self.target_bits,
            "block_size": self.block_size,
            "fidelity_target": self.fidelity_target,
        }
