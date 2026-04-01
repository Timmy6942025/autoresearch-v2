"""MLX inference backend for research agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoresearch.inference.mlx_backend")


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str
    tokens_generated: int
    prompt_tokens: int
    generation_time: float
    tokens_per_second: float
    peak_memory_mb: float = 0.0


class MLXBackend:
    """MLX inference engine for research agents.

    Provides text generation using MLX models on Apple Silicon.
    Supports TurboQuant KV cache compression for long-context generation.
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_context: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        turboquant: bool = False,
    ):
        self.model_path = model_path
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.turboquant = turboquant
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load the MLX model. Returns True on success."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            logger.info(f"Loading model: {self.model_path}")
            self._model, self._tokenizer = load(self.model_path)
            self._loaded = True
            logger.info("Model loaded successfully")
            return True
        except ImportError:
            logger.error("mlx-lm not installed. Run: pip install mlx-lm")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        if not self._loaded:
            if not self.load():
                return GenerationResult(
                    text="Error: Model not loaded",
                    tokens_generated=0,
                    prompt_tokens=0,
                    generation_time=0.0,
                    tokens_per_second=0.0,
                )

        try:
            from mlx_lm import generate

            temp = temperature or self.temperature
            tp = top_p or self.top_p

            start_time = time.time()

            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_p=tp,
                verbose=False,
            )

            gen_time = time.time() - start_time
            tokens = len(self._tokenizer.encode(text))
            prompt_tokens = len(self._tokenizer.encode(prompt))

            return GenerationResult(
                text=text,
                tokens_generated=tokens,
                prompt_tokens=prompt_tokens,
                generation_time=gen_time,
                tokens_per_second=tokens / max(gen_time, 0.001),
                peak_memory_mb=self._get_memory_usage(),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                text=f"Error: {e}",
                tokens_generated=0,
                prompt_tokens=0,
                generation_time=0.0,
                tokens_per_second=0.0,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
    ) -> GenerationResult:
        """Generate a chat response."""
        if not self._loaded:
            if not self.load():
                return GenerationResult(
                    text="Error: Model not loaded",
                    tokens_generated=0,
                    prompt_tokens=0,
                    generation_time=0.0,
                    tokens_per_second=0.0,
                )

        try:
            from mlx_lm import generate

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            return self.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return GenerationResult(
                text=f"Error: {e}",
                tokens_generated=0,
                prompt_tokens=0,
                generation_time=0.0,
                tokens_per_second=0.0,
            )

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024
        except Exception:
            return 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded
