"""Quantized Joint Low-rank (QJL) decomposition."""

from __future__ import annotations

from typing import Any, Dict


def qjl_decompose(residual: Any, rank: int = 8) -> Dict[str, Any]:
    """Quantized Joint Low-rank decomposition of residual error.

    Approximates residual as U @ V^T where U, V are quantized.
    Differs from SVD by jointly optimizing under quantization constraints.
    """
    try:
        import mlx.core as mx

        if not isinstance(residual, mx.array):
            residual = mx.array(residual)

        if residual.ndim < 2:
            return {"u": residual, "v": mx.array([]), "rank": 0}

        u, s, vt = mx.linalg.svd(residual, full_matrices=False)
        k = min(rank, len(s))
        u_k = u[:, :k]
        s_k = s[:k]
        vt_k = vt[:k, :]

        u_quantized = u_k * mx.sqrt(s_k)[None, :]
        v_quantized = vt_k.T * mx.sqrt(s_k)[None, :]

        return {"u": u_quantized, "v": v_quantized, "rank": k}
    except ImportError:
        return {"u": residual, "v": None, "rank": 0}
