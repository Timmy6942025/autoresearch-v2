"""Tests for log parsing and TSV handling in launch.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launch import parse_pytorch_output, parse_mlx_output


class TestParsePytorchOutput:
    def test_full_output(self):
        log = (
            "step 00100 (20.0%) | loss: 1.234567\n"
            "---\n"
            "val_bpb:          1.432100\n"
            "training_seconds: 295.3\n"
            "total_seconds:    310.5\n"
            "peak_vram_mb:     44500.0\n"
            "mfu_percent:      45.20\n"
            "total_tokens_M:   152.0\n"
            "num_steps:        290\n"
            "num_params_M:     12.5\n"
            "depth:            8\n"
        )
        result = parse_pytorch_output(log)
        assert result["val_bpb"] == 1.4321
        assert result["metrics"]["training_seconds"] == 295.3
        assert result["metrics"]["mfu_percent"] == 45.2
        assert result["metrics"]["total_tokens_M"] == 152.0
        assert result["metrics"]["num_steps"] == 290
        assert result["metrics"]["depth"] == 8
        assert result["metrics"]["peak_vram_mb"] == 44500.0

    def test_empty_output(self):
        result = parse_pytorch_output("")
        assert result["val_bpb"] is None
        assert result["metrics"] == {}


class TestParseMlxOutput:
    def test_full_output(self):
        log = (
            "step 00100 (20.0%) | loss: 1.234567\n"
            "---\n"
            "val_bpb:          1.432100\n"
            "training_seconds: 295.3\n"
            "total_seconds:    310.5\n"
            "num_steps:        290\n"
            "depth:            8\n"
        )
        result = parse_mlx_output(log)
        assert result["val_bpb"] == 1.4321
        assert result["metrics"]["training_seconds"] == 295.3
        assert result["metrics"]["total_seconds"] == 310.5
        assert result["metrics"]["num_steps"] == 290
        assert result["metrics"]["depth"] == 8

    def test_val_bpb_space_format(self):
        log = "val_bpb  1.234567\n"
        result = parse_mlx_output(log)
        assert result["val_bpb"] == 1.234567

    def test_empty_output(self):
        result = parse_mlx_output("")
        assert result["val_bpb"] is None
        assert result["metrics"] == {}
