"""Configuration management for research system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_DIR = Path.home() / ".autoresearch"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"


@dataclass
class ModelConfig:
    """Model configuration."""

    path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    max_context: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repeat_penalty: float = 1.1


@dataclass
class SearchConfig:
    """Search configuration."""

    engine: str = "duckduckgo"
    max_results: int = 20
    timeout: int = 10


@dataclass
class CrawlerConfig:
    """Crawler configuration."""

    concurrent: int = 8
    timeout: int = 15
    user_agent: str = "AutoResearchBot/2.0"


@dataclass
class TurboQuantConfig:
    """TurboQuant configuration."""

    enabled: bool = False
    method: str = "turboquant"
    block_size: int = 128
    target_bits: float = 4.5
    fidelity_target: float = 0.99


@dataclass
class AgentConfig:
    """Agent configuration."""

    max_iterations: int = 10
    parallel_searches: int = 4
    synthesis_model: Optional[str] = None


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    directory: str = str(Path.home() / ".autoresearch" / "cache")
    max_size_mb: int = 2048


@dataclass
class ResearchConfig:
    """Main research configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    turboquant: TurboQuantConfig = field(default_factory=TurboQuantConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchConfig":
        """Create config from dictionary."""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "search" in data:
            config.search = SearchConfig(**data["search"])
        if "crawler" in data:
            config.crawler = CrawlerConfig(**data["crawler"])
        if "turboquant" in data:
            config.turboquant = TurboQuantConfig(**data["turboquant"])
        if "agent" in data:
            config.agent = AgentConfig(**data["agent"])
        if "cache" in data:
            config.cache = CacheConfig(**data["cache"])

        return config

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "ResearchConfig":
        """Load config from YAML or JSON file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_FILE

        if not config_path.exists():
            return cls()

        try:
            if config_path.suffix in [".yaml", ".yml"]:
                import yaml

                with open(config_path) as f:
                    data = yaml.safe_load(f)
            else:
                with open(config_path) as f:
                    data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "search": self.search.__dict__,
            "crawler": self.crawler.__dict__,
            "turboquant": self.turboquant.__dict__,
            "agent": self.agent.__dict__,
            "cache": self.cache.__dict__,
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save config to file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.suffix in [".yaml", ".yml"]:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            with open(config_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
