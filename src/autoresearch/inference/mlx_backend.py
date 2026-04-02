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
        target_bits: float = 4.5,
        block_size: int = 128,
    ):
        self.model_path = model_path
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.turboquant = turboquant
        self.target_bits = target_bits
        self.block_size = block_size
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._kv_cache = None

    def load(self) -> bool:
        """Load the MLX model. Returns True on success."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            logger.info(f"Loading model: {self.model_path}")
            self._model, self._tokenizer = load(self.model_path)
            self._loaded = True

            if self.turboquant:
                from ..turboquant.cache import CompressedKVCache

                self._kv_cache = CompressedKVCache(
                    method="turboquant",
                    target_bits=self.target_bits,
                    block_size=self.block_size,
                )
                logger.info("TurboQuant KV cache compression enabled")

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
            import mlx.core as mx
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import generate_step

            temp = temperature or self.temperature
            tp = top_p or self.top_p

            start_time = time.time()

            prompt_tokens = self._tokenizer.encode(prompt)
            prompt_arr = mx.array(prompt_tokens)

            sampler = make_sampler(temp=temp, top_p=tp)
            tokens = []
            cache = None

            if self.turboquant and self._kv_cache:
                from mlx_lm.models.base import make_prompt_cache

                cache = make_prompt_cache(self._model)
                compressed_layers = set()

                for token in generate_step(
                    prompt_arr, self._model, cache=cache, sampler=sampler
                ):
                    token_val = token[0] if isinstance(token, tuple) else token
                    tokens.append(token_val.item())
                    if len(tokens) >= max_tokens:
                        break

                    if len(tokens) % 16 == 0:
                        for i, c in enumerate(cache):
                            if i in compressed_layers:
                                continue
                            if hasattr(c, "keys") and hasattr(c, "values"):
                                k_compressed = self._kv_cache.compressor.compress(
                                    c.keys
                                )
                                v_compressed = self._kv_cache.compressor.compress(
                                    c.values
                                )
                                self._kv_cache.put(
                                    f"layer_{i}_k_{len(tokens)}", k_compressed
                                )
                                self._kv_cache.put(
                                    f"layer_{i}_v_{len(tokens)}", v_compressed
                                )
                                c.keys = k_compressed
                                c.values = v_compressed
                                compressed_layers.add(i)
                                v_compressed = self._kv_cache.compressor.compress(
                                    c.values
                                )
                                self._kv_cache.put(
                                    f"layer_{i}_k_{len(tokens)}", k_compressed
                                )
                                self._kv_cache.put(
                                    f"layer_{i}_v_{len(tokens)}", v_compressed
                                )
                                c.keys = k_compressed
                                c.values = v_compressed
                                compressed_layers.add(i)
            else:
                for token in generate_step(prompt_arr, self._model, sampler=sampler):
                    token_val = token[0] if isinstance(token, tuple) else token
                    tokens.append(token_val.item())
                    if len(tokens) >= max_tokens:
                        break

            text = self._tokenizer.decode(tokens)
            gen_time = time.time() - start_time

            return GenerationResult(
                text=text,
                tokens_generated=len(tokens),
                prompt_tokens=len(prompt_tokens),
                generation_time=gen_time,
                tokens_per_second=len(tokens) / max(gen_time, 0.001),
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

    def get_turboquant_stats(self) -> Dict[str, Any]:
        """Get TurboQuant compression statistics."""
        if not self._kv_cache:
            return {"enabled": False}
        return {
            "enabled": True,
            "cache_size": self._kv_cache.size,
            **self._kv_cache.get_stats(),
        }
