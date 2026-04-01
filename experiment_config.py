"""Config-driven experiment system replacing regex-based source mutation in launch.py."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class HyperparamConfig:
    """All hyperparameters used by train.py and train_mlx.py.

    Fields are stored as JSON-serialisable primitives (e.g. ADAM_BETAS as
    a list instead of a tuple) so the config round-trips through JSON cleanly.
    """

    depth: int = 8
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"
    dim: Optional[int] = None
    n_head: Optional[int] = None
    n_kv_head: Optional[int] = None
    mlp_type: str = "relu2"
    block_type: str = "parallel"
    dropout: float = 0.0
    stochastic_depth: float = 0.0
    seq_len: Optional[int] = None

    total_batch_size: int = 524288
    device_batch_size: int = 256
    batch_size: Optional[int] = None
    time_budget: Optional[int] = None

    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.05
    scalar_lr: float = 0.5
    learning_rate: Optional[float] = None
    lr: Optional[float] = None

    adam_betas: List[float] = field(default_factory=lambda: [0.8, 0.95])
    weight_decay: float = 0.15

    warmup_ratio: float = 0.02
    warmup_pct: Optional[float] = None
    warmup_steps: Optional[int] = None
    warmdown_ratio: float = 0.1
    final_lr_frac: float = 0.1

    momentum: Optional[float] = None

    max_grad_norm: float = 1.0
    grad_clip: Optional[float] = None
    clip_grad_norm: Optional[float] = None

    label_smoothing: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperparamConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


def default_config(engine: str = "pytorch") -> HyperparamConfig:
    return HyperparamConfig()


def write_config(config: HyperparamConfig, engine: str = "pytorch",
                 experiment_id: Optional[str] = None) -> Path:
    """Write config to PROJECT_ROOT / 'config' / 'exp_{experiment_id}.json'.

    Creates the config directory if it doesn't exist.
    Returns the path to the written file.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if experiment_id is None:
        from datetime import datetime, timezone
        experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    config_path = CONFIG_DIR / f"exp_{experiment_id}.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return config_path


def load_config(path: Union[str, Path]) -> HyperparamConfig:
    return HyperparamConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def apply_config_to_script(config: HyperparamConfig,
                           engine: str = "pytorch") -> str:
    """Write config to a temp JSON file and return its path.

    The path can be passed to train scripts via EXPERIMENT_CONFIG env var
    or a command-line argument.  Caller is responsible for cleanup.
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"exp_{engine}_")
    with os.fdopen(fd, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    return path


_PARAM_MAP = {
    "DEPTH": "depth",
    "ASPECT_RATIO": "aspect_ratio",
    "HEAD_DIM": "head_dim",
    "WINDOW_PATTERN": "window_pattern",
    "DIM": "dim",
    "N_HEAD": "n_head",
    "N_KV_HEAD": "n_kv_head",
    "MLP_TYPE": "mlp_type",
    "BLOCK_TYPE": "block_type",
    "DROPOUT": "dropout",
    "STOCHASTIC_DEPTH": "stochastic_depth",
    "SEQ_LEN": "seq_len",
    "TOTAL_BATCH_SIZE": "total_batch_size",
    "DEVICE_BATCH_SIZE": "device_batch_size",
    "BATCH_SIZE": "batch_size",
    "TIME_BUDGET": "time_budget",
    "EMBEDDING_LR": "embedding_lr",
    "UNEMBEDDING_LR": "unembedding_lr",
    "MATRIX_LR": "matrix_lr",
    "SCALAR_LR": "scalar_lr",
    "LEARNING_RATE": "learning_rate",
    "LR": "lr",
    "ADAM_BETAS": "adam_betas",
    "WEIGHT_DECAY": "weight_decay",
    "WARMUP_RATIO": "warmup_ratio",
    "WARMUP_PCT": "warmup_pct",
    "WARMUP_STEPS": "warmup_steps",
    "WARMDOWN_RATIO": "warmdown_ratio",
    "FINAL_LR_FRAC": "final_lr_frac",
    "MOMENTUM": "momentum",
    "MAX_GRAD_NORM": "max_grad_norm",
    "GRAD_CLIP": "grad_clip",
    "CLIP_GRAD_NORM": "clip_grad_norm",
    "LABEL_SMOOTHING": "label_smoothing",
}


def resolve_alias(param: str) -> Optional[str]:
    return _PARAM_MAP.get(param)
