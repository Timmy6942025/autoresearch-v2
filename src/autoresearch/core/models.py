"""Model loading and MLX integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.core.models")


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    path: str
    params_m: int = 0
    max_context: int = 32768
    supports_turboquant: bool = False
    quant_level: str = "4bit"


class ModelRegistry:
    """Registry for MLX models."""

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(self, info: ModelInfo) -> None:
        self._models[info.name] = info

    def get(self, name: str) -> Optional[ModelInfo]:
        return self._models.get(name)

    def list_all(self) -> List[ModelInfo]:
        return list(self._models.values())

    def remove(self, name: str) -> bool:
        if name in self._models:
            del self._models[name]
            return True
        return False


DEFAULT_MODELS = [
    ModelInfo(
        "llama-3.2-3b", "mlx-community/Llama-3.2-3B-Instruct-4bit", 3000, 32768, True
    ),
    ModelInfo(
        "llama-3.1-70b", "mlx-community/Llama-3.1-70B-Instruct-4bit", 70000, 8192, True
    ),
    ModelInfo(
        "qwen2.5-7b", "mlx-community/Qwen2.5-7B-Instruct-4bit", 7000, 32768, True
    ),
    ModelInfo(
        "qwen2.5-3b", "mlx-community/Qwen2.5-3B-Instruct-4bit", 3000, 32768, True
    ),
    ModelInfo(
        "mistral-7b", "mlx-community/Mistral-7B-Instruct-v0.3-4bit", 7000, 32768, True
    ),
    ModelInfo(
        "phi-3-mini", "mlx-community/Phi-3-mini-4k-instruct-4bit", 3800, 4096, False
    ),
]

registry = ModelRegistry()
for model in DEFAULT_MODELS:
    registry.register(model)
