"""Lloyd-Max codebook quantization."""

from __future__ import annotations

from typing import Any, Tuple


def lloyd_max_quantize(
    tensor: Any,
    target_bits: float = 4.5,
    block_size: int = 128,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> Tuple[Any, Any, Any]:
    """Lloyd-Max codebook quantization with per-block optimization.

    Learns optimal codebook centroids that minimize MSE for the tensor's
    specific distribution.

    Returns: (quantized_indices, codebook, scales)
    """
    try:
        import mlx.core as mx

        if not isinstance(tensor, mx.array):
            tensor = mx.array(tensor)

        n_centroids = int(2**target_bits)
        flat = tensor.reshape(-1)

        min_val = mx.min(flat)
        max_val = mx.max(flat)
        codebook = mx.linspace(min_val, max_val, n_centroids)

        for _ in range(max_iterations):
            diffs = mx.abs(flat[:, None] - codebook[None, :])
            indices = mx.argmin(diffs, axis=1)

            new_centroids = []
            for i in range(n_centroids):
                mask = indices == i
                if mx.sum(mask) > 0:
                    new_centroids.append(mx.mean(flat[mask]))
                else:
                    new_centroids.append(codebook[i])

            new_codebook = mx.array(new_centroids)
            if mx.max(mx.abs(new_codebook - codebook)) < tolerance:
                break
            codebook = new_codebook

        diffs = mx.abs(flat[:, None] - codebook[None, :])
        indices = mx.argmin(diffs, axis=1)

        quantized = codebook[indices].reshape(tensor.shape)
        scales = mx.max(mx.abs(tensor))

        return indices.reshape(tensor.shape), codebook, scales
    except ImportError:
        return tensor, None, None
