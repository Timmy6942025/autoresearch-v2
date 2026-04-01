"""Tests for experiment_config module."""

import json
import os
from pathlib import Path

from experiment_config import (
    HyperparamConfig,
    write_config,
    load_config,
    default_config,
    apply_config_to_script,
    resolve_alias,
)


class TestHyperparamConfig:
    def test_defaults(self):
        cfg = HyperparamConfig()
        assert cfg.depth == 8
        assert cfg.device_batch_size == 256
        assert cfg.embedding_lr == 0.6
        assert cfg.window_pattern == "SSSL"
        assert cfg.max_grad_norm == 1.0

    def test_to_dict_roundtrip(self):
        cfg = HyperparamConfig(depth=12, device_batch_size=128)
        d = cfg.to_dict()
        restored = HyperparamConfig.from_dict(d)
        assert restored.depth == 12
        assert restored.device_batch_size == 128

    def test_from_dict_filters_unknown(self):
        d = {"depth": 10, "nonexistent_field": 42}
        cfg = HyperparamConfig.from_dict(d)
        assert cfg.depth == 10


class TestConfigIO:
    def test_write_and_load(self, tmp_path):
        cfg = HyperparamConfig(depth=10, window_pattern="LLLS")
        path = write_config(cfg, experiment_id="test_exp")
        path.rename(tmp_path / path.name)
        loaded = load_config(tmp_path / path.name)
        assert loaded.depth == 10
        assert loaded.window_pattern == "LLLS"

    def test_default_config(self):
        cfg = default_config("pytorch")
        assert cfg.depth == 8

    def test_apply_config_to_script(self):
        cfg = HyperparamConfig(depth=4)
        path = apply_config_to_script(cfg, engine="mlx")
        data = json.loads(Path(path).read_text())
        assert data["depth"] == 4
        try:
            os.remove(path)
        except OSError:
            pass


class TestResolveAlias:
    def test_direct_mapping(self):
        assert resolve_alias("DEPTH") == "depth"
        assert resolve_alias("MATRIX_LR") == "matrix_lr"
        assert resolve_alias("DEVICE_BATCH_SIZE") == "device_batch_size"

    def test_alias(self):
        assert resolve_alias("LEARNING_RATE") == "learning_rate"
        assert resolve_alias("BATCH_SIZE") == "batch_size"
        assert resolve_alias("GRAD_CLIP") == "grad_clip"

    def test_unknown(self):
        assert resolve_alias("TOTALLY_UNKNOWN") is None
