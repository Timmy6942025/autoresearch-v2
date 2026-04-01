"""Rotation optimization for KV cache compression."""

from __future__ import annotations

from typing import Any, Tuple


def optimize_rotation(tensor: Any) -> Tuple[Any, Any]:
    """Apply rotation optimization to concentrate energy along principal axes.

    Finds optimal orthogonal transformation that minimizes quantization error.
    """
    try:
        import mlx.core as mx

        if not isinstance(tensor, mx.array):
            tensor = mx.array(tensor)

        if tensor.ndim < 2:
            return tensor, mx.eye(tensor.shape[0]) if tensor.shape else mx.array(1.0)

        cov = tensor.T @ tensor / tensor.shape[0]
        eigenvalues, eigenvectors = mx.linalg.eigh(cov)
        rotation = eigenvectors[:, ::-1]
        rotated = tensor @ rotation
        return rotated, rotation
    except ImportError:
        return tensor, None
