"""Memory usage tracking."""

from __future__ import annotations

import resource
from typing import Dict


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


def get_memory_stats() -> Dict[str, float]:
    """Get detailed memory statistics."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "max_rss_mb": usage.ru_maxrss / 1024,
        "shared_mem_mb": usage.ru_ixrss / 1024,
        "unshared_data_mb": usage.ru_idrss / 1024,
        "unshared_stack_mb": usage.ru_isrss / 1024,
    }
