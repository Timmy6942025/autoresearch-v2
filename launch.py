#!/usr/bin/env python3
"""
Autoresearch v2: Recursive Self-Improving ML Research System.

Master entry point that orchestrates the full research loop:
  1. Initialize / load knowledge base
  2. Get next experiment from orchestrator / experiment_designer
  3. Apply changes to train.py (read / modify / write)
  4. Run `uv run train.py` or `uv run train_mlx.py` with timeout
  5. Parse val_bpb and metrics from run output
  6. Record result in knowledge base + results.tsv
  7. Every N experiments: run meta_analyzer + dashboard
  8. Every M experiments: run self_improve
  9. Checkpoint state for crash recovery

Modes:
  baseline  - Run a single baseline experiment
  single    - Run N experiments (sequential)
  night     - ~100 experiments (overnight run)
  deep      - With meta-analysis cycles
  recursive - Full self-improving loop

Engines:
  pytorch   - CPU/CUDA via PyTorch (train.py)
  mlx       - Apple Silicon GPU via MLX (train_mlx.py)

Usage:
  python launch.py --mode baseline
  python launch.py --mode single -n 10
  python launch.py --mode night --engine mlx
  python launch.py --mode deep --engine pytorch
  python launch.py --mode recursive --engine mlx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = PROJECT_ROOT / "state.json"
TSV_FILE = RESULTS_DIR / "results.tsv"

DEFAULT_TRAIN_TIMEOUT = 600  # 10 minutes per experiment
META_ANALYZER_INTERVAL = 20  # run meta analysis every N experiments
SELF_IMPROVE_INTERVAL = 50   # run self-improve every N experiments
DEFAULT_NIGHT_COUNT = 100

# Engine-specific paths and settings
ENGINE_CONFIG = {
    "pytorch": {
        "train_script": "train.py",
        "target_file": "train.py",
        "reset_target": "train.py",
        "output_prefix": "results/",
    },
    "mlx": {
        "train_script": "train_mlx.py",
        "target_file": "train.py",  # MLX still reads hyperparams from shared config
        "reset_target": "train.py",
        "output_prefix": "results/",
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "launch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("launch")

# ---------------------------------------------------------------------------
# Global engine context (set by main(), read by helpers)
# ---------------------------------------------------------------------------

_current_engine: str = "pytorch"


def get_engine() -> str:
    """Return the current engine being used."""
    return _current_engine


def get_engine_config(engine: str) -> Dict[str, str]:
    """Return configuration dict for the given engine."""
    return ENGINE_CONFIG.get(engine, ENGINE_CONFIG["pytorch"])


# ---------------------------------------------------------------------------
# State checkpoint
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Serializable checkpoint for crash recovery."""
    mode: str = ""
    experiment_count: int = 0
    experiments_completed: int = 0
    experiments_failed: int = 0
    start_time: str = ""
    last_experiment_id: str = ""
    last_val_bpb: Optional[float] = None
    best_val_bpb: Optional[float] = None
    best_experiment_id: str = ""
    interrupted: bool = False
    pending_experiments: List[Dict[str, Any]] = field(default_factory=list)
    completed_ids: set = field(default_factory=set)
    engine: str = "pytorch"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["completed_ids"] = list(self.completed_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        d = dict(d)
        d["completed_ids"] = set(d.get("completed_ids", []))
        return cls(**d)


def save_state(state: RunState) -> None:
    """Atomically save state to disk."""
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, default=str))
    tmp.rename(STATE_FILE)


def load_state() -> Optional[RunState]:
    """Load saved state if it exists."""
    if STATE_FILE.exists():
        try:
            return RunState.from_dict(json.loads(STATE_FILE.read_text()))
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
    return None


# ---------------------------------------------------------------------------
# TSV results writer
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    "experiment_id\texperiment_name\tval_bpb\tdelta\tphase\tstatus\t"
    "timestamp\tduration\tchanges\tnotes\tengine"
)


def _ensure_tsv_header() -> bool:
    """Ensure the TSV file has a header. Returns True if file was created."""
    if not TSV_FILE.exists() or TSV_FILE.stat().st_size == 0:
        TSV_FILE.parent.mkdir(parents=True, exist_ok=True)
        TSV_FILE.write_text(_TSV_HEADER + "\n", encoding="utf-8")
        return True
    # If old header without engine column, check
    first_line = TSV_FILE.open(encoding="utf-8").readline().strip()
    if "engine" not in first_line:
        return True  # will be handled by append
    return False


def append_tsv(
    experiment_id: str,
    experiment_name: str,
    val_bpb: Optional[float],
    delta: Optional[float],
    phase: int,
    status: str,
    duration: float,
    changes: str,
    notes: str = "",
    engine: str = "pytorch",
) -> None:
    """Append a row to results.tsv."""
    _ensure_tsv_header()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "N/A"
    delta_str = f"{delta:+.6f}" if delta is not None else "N/A"
    with open(TSV_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{experiment_id}\t{experiment_name}\t{bpb_str}\t{delta_str}\t"
            f"{phase}\t{status}\t{ts}\t{duration:.1f}\t{changes}\t{notes}\t{engine}\n"
        )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message: str) -> Optional[str]:
    """Stage changes and commit. Returns commit hash or None on failure."""
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return hash_result.stdout.strip()
    except Exception as e:
        logger.debug(f"Git commit failed: {e}")
    return None


def git_current_branch() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def git_reset_train(engine: str = "pytorch") -> bool:
    """Reset the target training script to the last committed version."""
    config = get_engine_config(engine)
    target = config.get("reset_target", "train.py")
    try:
        subprocess.run(
            ["git", "checkout", "--", target],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# train.py / train_mlx.py mutation
# ---------------------------------------------------------------------------

def apply_changes_to_train(changes: Dict[str, Any], engine: str = "pytorch") -> Optional[str]:
    """
    Apply parameter changes to the training script by modifying constants.
    For MLX, edits train.py which shares config with train_mlx.py.
    Returns a description of changes made, or None on failure.
    """
    config = get_engine_config(engine)
    train_path = PROJECT_ROOT / config.get("target_file", "train.py")
    if not train_path.exists():
        logger.error(f"{train_path.name} not found!")
        return None

    content = train_path.read_text()
    original = content
    applied = []

    # Mapping from experiment param names to variable names
    PARAM_MAP = {
        "DEPTH": ("n_layer", int),
        "DIM": ("n_embd", int),
        "N_HEAD": ("n_head", int),
        "N_KV_HEAD": ("n_kv_head", int),
        "HEAD_DIM": ("head_dim", int),
        "LEARNING_RATE": ("learning_rate", float),
        "LR": ("learning_rate", float),
        "WARMUP_PCT": ("warmup_pct", float),
        "WARMUP_STEPS": ("warmup_steps", int),
        "WEIGHT_DECAY": ("weight_decay", float),
        "BATCH_SIZE": ("batch_size", int),
        "SEQ_LEN": ("sequence_len", int),
        "GRAD_CLIP": ("clip_grad_norm", float),
        "CLIP_GRAD_NORM": ("clip_grad_norm", float),
        "DROPOUT": ("dropout", float),
        "LABEL_SMOOTHING": ("label_smoothing", float),
        "MOMENTUM": ("muon_momentum", float),
        "STOCHASTIC_DEPTH": ("stochastic_depth", float),
        "WINDOW_PATTERN": ("window_pattern", str),
        "MLP_TYPE": ("mlp_type", str),
        "BLOCK_TYPE": ("block_type", str),
        # MLX-specific aliases
        "TOTAL_BATCH_SIZE": ("TOTAL_BATCH_SIZE", int),
        "DEVICE_BATCH_SIZE": ("DEVICE_BATCH_SIZE", int),
        "TIME_BUDGET": ("TIME_BUDGET", int),
        "EMBEDDING_LR": ("EMBEDDING_LR", float),
        "MATRIX_LR": ("MATRIX_LR", float),
        "SCALAR_LR": ("SCALAR_LR", float),
        "ADAM_BETAS": ("ADAM_BETAS", str),
        "MAX_GRAD_NORM": ("MAX_GRAD_NORM", float),
        "WARMUP_RATIO": ("WARMUP_RATIO", float),
    }

    for param, value in changes.items():
        key, dtype = PARAM_MAP.get(param, (None, None))
        if key is None:
            logger.warning(f"Unknown param '{param}' - skipping")
            continue

        if dtype == str:
            patterns = [
                rf'^(\s*{re.escape(key)}\s*=\s*)(["\']).*?\2',
                rf"^(\s*{re.escape(key)}\s*=\s*)\S+",
            ]
        else:
            patterns = [
                rf"^(\s*{re.escape(key)}\s*=\s*)\S+",
            ]

        replaced = False
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                if dtype == str:
                    new_line = f'{match.group(1)}"{value}"'
                elif dtype == int:
                    new_line = f"{match.group(1)}{int(value)}"
                else:
                    new_line = f"{match.group(1)}{float(value)}"
                content = content[:match.start()] + new_line + content[match.end():]
                replaced = True
                break

        if replaced:
            applied.append(f"{param}={value}")
        else:
            logger.warning(f"Could not find '{key}' in {train_path.name} for param '{param}'")

    if applied:
        train_path.write_text(content)
        return ", ".join(applied)
    return None


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def parse_pytorch_output(log: str) -> Dict[str, Any]:
    """Parse PyTorch training output (train.py)."""
    result = {
        "val_bpb": None,
        "metrics": {},
    }
    for line in log.splitlines():
        stripped = line.strip()
        m_bpb = re.match(r"^val_bpb:\s*([\d.]+)", stripped)
        if m_bpb:
            result["val_bpb"] = float(m_bpb.group(1))

        m_sec = re.match(r"^training_seconds:\s*([\d.]+)", stripped)
        if m_sec:
            result["metrics"]["training_seconds"] = float(m_sec.group(1))

        m_mfu = re.match(r"^mfu_percent:\s*([\d.]+)", stripped)
        if m_mfu:
            result["metrics"]["mfu_percent"] = float(m_mfu.group(1))

        m_tok = re.match(r"^total_tokens_M:\s*([\d.]+)", stripped)
        if m_tok:
            result["metrics"]["total_tokens_M"] = float(m_tok.group(1))

        m_steps = re.match(r"^num_steps:\s*(\d+)", stripped)
        if m_steps:
            result["metrics"]["num_steps"] = int(m_steps.group(1))

        m_depth = re.match(r"^depth:\s*(\d+)", stripped)
        if m_depth:
            result["metrics"]["depth"] = int(m_depth.group(1))

        m_vram = re.match(r"^peak_vram_mb:\s*([\d.]+)", stripped)
        if m_vram:
            result["metrics"]["peak_vram_mb"] = float(m_vram.group(1))

    return result


def parse_mlx_output(log: str) -> Dict[str, Any]:
    """Parse MLX training output (train_mlx.py)."""
    result = {
        "val_bpb": None,
        "metrics": {},
    }
    for line in log.splitlines():
        stripped = line.strip()
        # MLX uses "val_bpb: 1.234567" format (same pattern, but also try colon-space)
        m_bpb = re.match(r"^val_bpb:\s*([\d.]+)", stripped)
        if m_bpb:
            result["val_bpb"] = float(m_bpb.group(1))
            continue

        m_bpb2 = re.match(r"^val_bpb\s+([\d.]+)", stripped)
        if m_bpb2:
            result["val_bpb"] = float(m_bpb2.group(1))
            continue

        # training_seconds
        m_sec = re.match(r"^training_seconds:\s*([\d.]+)", stripped)
        if m_sec:
            result["metrics"]["training_seconds"] = float(m_sec.group(1))
            continue

        # total_seconds (MLX-specific)
        m_total = re.match(r"^total_seconds:\s*([\d.]+)", stripped)
        if m_total:
            result["metrics"]["total_seconds"] = float(m_total.group(1))
            continue

        # num_steps
        m_steps = re.match(r"^num_steps:\s*(\d+)", stripped)
        if m_steps:
            result["metrics"]["num_steps"] = int(m_steps.group(1))
            continue

        # mfu_percent (if MLX reports it)
        m_mfu = re.match(r"^mfu_percent:\s*([\d.]+)", stripped)
        if m_mfu:
            result["metrics"]["mfu_percent"] = float(m_mfu.group(1))
            continue

        # peak_vram_mb (MLX unified memory)
        m_vram = re.match(r"^peak_vram_mb:\s*([\d.]+)", stripped)
        if m_vram:
            result["metrics"]["peak_vram_mb"] = float(m_vram.group(1))
            continue

        # depth (if reported)
        m_depth = re.match(r"^depth:\s*(\d+)", stripped)
        if m_depth:
            result["metrics"]["depth"] = int(m_depth.group(1))
            continue

    return result


def run_training(timeout: int = DEFAULT_TRAIN_TIMEOUT, engine: str = "pytorch") -> Dict[str, Any]:
    """
    Run training via the appropriate engine's script.
    - pytorch: uv run train.py
    - mlx: uv run train_mlx.py
    Returns dict with val_bpb, metrics, status, log output.
    """
    config = get_engine_config(engine)
    train_script = config.get("train_script", "train.py")

    result = {
        "val_bpb": None,
        "status": "success",
        "duration": 0.0,
        "log": "",
        "metrics": {},
        "error": "",
        "engine": engine,
    }

    t0 = time.time()
    try:
        logger.info(f"Starting {engine} training: uv run {train_script} (timeout={timeout}s)")
        proc = subprocess.run(
            ["uv", "run", train_script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        result["duration"] = time.time() - t0
        result["log"] = proc.stdout + proc.stderr

        # Parse output using engine-specific parser
        if engine == "mlx":
            parsed = parse_mlx_output(result["log"])
        else:
            parsed = parse_pytorch_output(result["log"])

        result["val_bpb"] = parsed["val_bpb"]
        result["metrics"] = parsed["metrics"]

        # Check for failure signals
        if proc.returncode != 0:
            if "FAIL" in result["log"] or proc.returncode == 1:
                result["status"] = "failed"
                result["error"] = f"Training exited with code {proc.returncode}"
            elif "nan" in result["log"].lower() or "isnan" in result["log"].lower():
                result["status"] = "failed"
                result["error"] = "Training diverged (NaN detected)"
            else:
                result["status"] = "crashed"
                result["error"] = f"Training crashed (exit code {proc.returncode})"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration"] = timeout
        result["error"] = f"Training timed out after {timeout}s"
    except Exception as e:
        result["status"] = "crashed"
        result["duration"] = time.time() - t0
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Meta-analysis wrapper
# ---------------------------------------------------------------------------

def run_meta_analyzer(
    results_tsv: str,
    analysis_name: str = "latest",
) -> Dict[str, Any]:
    """Run the meta_analyzer script and return its output summary."""
    analyzer = SCRIPT_DIR / "scripts" / "meta_analyzer.py"
    if not analyzer.exists():
        logger.warning("meta_analyzer.py not found - skipping")
        return {}

    analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_md = analysis_dir / f"{analysis_name}_analysis.md"

    try:
        proc = subprocess.run(
            [
                sys.executable, str(analyzer),
                "--results", results_tsv,
                "--output", str(output_md),
                "--next-experiments", "10",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            logger.info(f"Meta-analysis written to {output_md}")
            return {"output": str(output_md), "status": "success"}
        else:
            logger.warning(f"Meta-analyzer exited with code {proc.returncode}")
            return {"status": "error", "error": proc.stderr[:500]}
    except Exception as e:
        logger.warning(f"Meta-analyzer failed: {e}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Dashboard wrapper
# ---------------------------------------------------------------------------

def run_dashboard(
    results_tsv: str,
    plots_dir: str,
) -> Dict[str, Any]:
    """Run the dashboard script to generate plots and reports."""
    dashboard = SCRIPT_DIR / "scripts" / "dashboard.py"
    if not dashboard.exists():
        logger.warning("dashboard.py not found - skipping")
        return {}

    report_md = PROJECT_ROOT / "analysis" / "latest_report.md"

    try:
        proc = subprocess.run(
            [
                sys.executable, str(dashboard),
                "--results", results_tsv,
                "--plots-dir", plots_dir,
                "--output-md", str(report_md),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            logger.info(f"Dashboard generated plots in {plots_dir}")
            return {"plots_dir": plots_dir, "report": str(report_md), "status": "success"}
        else:
            logger.warning(f"Dashboard exited with code {proc.returncode}")
            return {"status": "error", "error": proc.stderr[:500]}
    except Exception as e:
        logger.warning(f"Dashboard failed: {e}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Self-improve wrapper
# ---------------------------------------------------------------------------

def run_self_improve(
    results_tsv: str,
    knowledge_path: str,
) -> Dict[str, Any]:
    """Run the self_improve script to update train.py based on learnings."""
    self_improve = SCRIPT_DIR / "scripts" / "self_improve.py"
    if not self_improve.exists():
        logger.warning("self_improve.py not found - skipping")
        return {}

    try:
        proc = subprocess.run(
            [
                sys.executable, str(self_improve),
                "--results", results_tsv,
                "--knowledge", knowledge_path,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0:
            logger.info("Self-improve applied successfully")
            return {"status": "success"}
        else:
            logger.warning(f"Self-improve exited with code {proc.returncode}")
            return {"status": "error", "error": proc.stderr[:500]}
    except Exception as e:
        logger.warning(f"Self-improve failed: {e}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Experiment queue generation
# ---------------------------------------------------------------------------

def get_experiment_queue(
    mode: str,
    count: int,
    knowledge_path: str,
    state: RunState,
) -> List[Dict[str, Any]]:
    """
    Generate or resume an experiment queue using the orchestrator /
    experiment_designer.
    """
    designer = SCRIPT_DIR / "scripts" / "experiment_designer.py"
    orchestrator = SCRIPT_DIR / "scripts" / "research_orchestrator.py"

    # Resume: use pending experiments from state
    if state.pending_experiments:
        logger.info(f"Resuming with {len(state.pending_experiments)} pending experiments")
        return state.pending_experiments

    # Try experiment_designer first
    if designer.exists():
        try:
            proc = subprocess.run(
                [
                    sys.executable, str(designer),
                    "--mode", mode,
                    "--count", str(count),
                    "--knowledge", knowledge_path,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                queue = json.loads(proc.stdout)
                if isinstance(queue, list):
                    logger.info(f"Loaded {len(queue)} experiments from experiment_designer")
                    return queue
        except Exception as e:
            logger.warning(f"experiment_designer failed: {e}")

    # Fallback: use research_orchestrator
    if orchestrator.exists():
        try:
            proc = subprocess.run(
                [
                    sys.executable, str(orchestrator),
                    "--generate",
                    "--count", str(count),
                    "--mode", mode,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                queue = json.loads(proc.stdout)
                if isinstance(queue, list):
                    logger.info(f"Loaded {len(queue)} experiments from orchestrator")
                    return queue
        except Exception as e:
            logger.warning(f"orchestrator generate failed: {e}")

    # Ultimate fallback: generate a simple inline queue
    logger.info("Using inline experiment generation fallback")
    return generate_simple_queue(mode, count)


def generate_simple_queue(mode: str, count: int) -> List[Dict[str, Any]]:
    """Generate a basic experiment queue as fallback."""
    experiments = []

    # Learning rate sweep
    for lr in [0.001, 0.003, 0.005, 0.01, 0.03]:
        experiments.append({
            "experiment_id": f"exp_{len(experiments)+1:03d}",
            "changes": {"LEARNING_RATE": lr},
            "category": "optimization",
            "hypothesis": f"LR {lr} for convergence tuning",
            "priority": 8,
        })

    # Architecture depth
    for depth in [4, 8, 12, 16]:
        experiments.append({
            "experiment_id": f"exp_{len(experiments)+1:03d}",
            "changes": {"DEPTH": depth},
            "category": "architecture",
            "hypothesis": f"Depth {depth} capacity test",
            "priority": 7,
        })

    # Batch size
    for bs in [32, 64, 128, 256]:
        experiments.append({
            "experiment_id": f"exp_{len(experiments)+1:03d}",
            "changes": {"BATCH_SIZE": bs},
            "category": "training",
            "hypothesis": f"Batch size {bs}",
            "priority": 6,
        })

    # Dropout
    for do in [0.0, 0.05, 0.1, 0.2]:
        experiments.append({
            "experiment_id": f"exp_{len(experiments)+1:03d}",
            "changes": {"DROPOUT": do},
            "category": "regularization",
            "hypothesis": f"Dropout {do}",
            "priority": 5,
        })

    # Sequence length
    for sl in [512, 1024, 4096]:
        experiments.append({
            "experiment_id": f"exp_{len(experiments)+1:03d}",
            "changes": {"SEQ_LEN": sl},
            "category": "training",
            "hypothesis": f"Seq len {sl}",
            "priority": 6,
        })

    # Filter and limit
    experiments = experiments[:count]

    if mode == "deep" or mode == "recursive":
        for i, e1 in enumerate(experiments[:5]):
            for e2 in experiments[i+1:8]:
                if len(experiments) >= count:
                    break
                combined = {
                    "experiment_id": f"combo_{len(experiments)+1:03d}",
                    "changes": {**e1["changes"], **e2["changes"]},
                    "category": "combo",
                    "hypothesis": f"Combo: {e1['hypothesis']} + {e2['hypothesis']}",
                    "priority": 9,
                }
                experiments.append(combined)

    return experiments[:count]


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def print_banner() -> None:
    width = 70
    engine_label = get_engine().upper()
    print()
    print("=" * width)
    print(f"  AUTORESEARCH v2: Recursive Self-Improving ML Research System  [{engine_label}]")
    print("=" * width)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Branch       : {git_current_branch()}")
    print(f"  Engine       : {_current_engine}")
    print(f"  Results dir  : {RESULTS_DIR}")
    print(f"  State file   : {STATE_FILE}")
    print(f"  TSV file     : {TSV_FILE}")
    print(f"  Python       : {sys.version.split()[0]}")
    print("=" * width)
    print()


def print_progress(
    current: int,
    total: int,
    experiment_id: str,
    experiment_name: str,
    val_bpb: Optional[float],
    delta: Optional[float],
    best_bpb: Optional[float],
    duration: float,
    status: str,
    start_time: float,
) -> None:
    """Print a single-line progress update."""
    elapsed = time.time() - start_time
    eta = "?"
    if current > 0:
        avg_time = elapsed / current
        remaining = (total - current) * avg_time
        eta = f"{remaining/60:.0f}m"

    bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "N/A"
    delta_str = f"{delta:+.6f}" if delta is not None else "N/A"
    best_str = f"{best_bpb:.6f}" if best_bpb is not None else "N/A"

    status_icon = {
        "success": "[OK]",
        "failed": "[FAIL]",
        "crashed": "[CRASH]",
        "timeout": "[TO]",
    }.get(status, "[??]")

    print(
        f"  {current}/{total} {status_icon} | {experiment_id:12s} | "
        f"bpb={bpb_str} delta={delta_str} | "
        f"best={best_str} | {duration:6.1f}s | "
        f"{experiment_name} | ETA:{eta}"
    )


def print_summary(state: RunState, baseline_bpb: Optional[float]) -> None:
    """Print final run summary."""
    width = 70
    print()
    print("=" * width)
    print(f"  RUN SUMMARY [{get_engine().upper()}]")
    print("=" * width)
    print(f"  Mode                 : {state.mode}")
    print(f"  Engine               : {state.engine}")
    print(f"  Total experiments    : {state.experiments_completed}")
    print(f"  Failed               : {state.experiments_failed}")
    total = state.experiments_completed + state.experiments_failed
    success_rate = (state.experiments_completed / max(total, 1)) * 100
    print(f"  Success rate         : {success_rate:.1f}%")
    print(f"  Best val_bpb         : {state.best_val_bpb:.6f}" if state.best_val_bpb else "  Best val_bpb         : N/A")
    print(f"  Best experiment      : {state.best_experiment_id}")
    if baseline_bpb and state.best_val_bpb:
        total_improvement = baseline_bpb - state.best_val_bpb
        print(f"  Total improvement    : {total_improvement:+.6f}")
    print(f"  Results TSV          : {TSV_FILE}")
    print(f"  Knowledge base       : {PROJECT_ROOT / 'knowledge.json'}")
    print(f"  Plots                : {PLOTS_DIR}")
    print(f"  State (resumable)    : {STATE_FILE}")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

_interrupted = False
_current_engine = "pytorch"


def handle_interrupt(signum, frame):
    global _interrupted
    _interrupted = True
    logger.info("\nInterrupt received! Saving state and shutting down gracefully...")


def main():
    global _interrupted, _current_engine
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    parser = argparse.ArgumentParser(
        description="Autoresearch v2: Recursive Self-Improving ML Research System",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "single", "night", "deep", "recursive"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument(
        "--engine",
        choices=["pytorch", "mlx"],
        default="pytorch",
        help="ML engine to use: pytorch (train.py) or mlx (train_mlx.py). Default: pytorch",
    )
    parser.add_argument(
        "-n", "--num-experiments",
        type=int,
        default=None,
        help="Number of experiments to run (default: mode-dependent)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TRAIN_TIMEOUT,
        help=f"Training timeout in seconds (default: {DEFAULT_TRAIN_TIMEOUT})",
    )
    parser.add_argument(
        "--meta-interval",
        type=int,
        default=META_ANALYZER_INTERVAL,
        help=f"Run meta-analysis every N experiments (default: {META_ANALYZER_INTERVAL})",
    )
    parser.add_argument(
        "--self-improve-interval",
        type=int,
        default=SELF_IMPROVE_INTERVAL,
        help=f"Run self-improve every N experiments (default: {SELF_IMPROVE_INTERVAL})",
    )
    parser.add_argument(
        "--knowledge",
        type=str,
        default=str(PROJECT_ROOT / "knowledge.json"),
        help="Path to knowledge base file",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Experiment phase number",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing state and start fresh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment queue without executing",
    )
    args = parser.parse_args()

    # Set the global engine context
    _current_engine = args.engine

    print_banner()

    # Determine experiment count
    mode_counts = {
        "baseline": 1,
        "single": args.num_experiments or 10,
        "night": args.num_experiments or DEFAULT_NIGHT_COUNT,
        "deep": args.num_experiments or 50,
        "recursive": args.num_experiments or 200,
    }
    count = mode_counts[args.mode]

    # Load or initialize state
    state = RunState()
    if args.resume and not args.no_resume:
        saved_state = load_state()
        if saved_state:
            state = saved_state
            if state.interrupted:
                logger.info("Resuming from previous interrupted run")
            else:
                logger.info(f"Loaded state: {state.experiments_completed} completed, {len(state.pending_experiments)} pending")
            # Ensure engine matches resumed state, or warn
            if hasattr(state, 'engine') and state.engine and state.engine != _current_engine:
                logger.warning(f"Resumed state used engine '{state.engine}', but current engine is '{_current_engine}'")

    state.mode = args.mode
    state.engine = _current_engine

    # Initialize knowledge base
    logger.info(f"Engine: {_current_engine} | Knowledge base: {args.knowledge}")

    # Get experiment queue
    logger.info(f"Mode: {args.mode} | Experiments: {count} | Phase: {args.phase}")
    logger.info("Generating experiment queue...")
    queue = get_experiment_queue(args.mode, count, args.knowledge, state)

    if args.dry_run:
        logger.info("DRY RUN - Experiment queue:")
        for i, exp in enumerate(queue):
            changes_str = ", ".join(f"{k}={v}" for k, v in exp.get("changes", {}).items())
            print(f"  {i+1}. [{exp.get('experiment_id', '?')}] {changes_str} | {exp.get('hypothesis', '')}")
        print(f"\nTotal: {len(queue)} experiments")
        return

    # Filter out already-completed experiments
    completed_ids = state.completed_ids
    queue = [e for e in queue if e.get("experiment_id") not in completed_ids]
    logger.info(f"Experiment queue: {len(queue)} experiments remaining")

    if not queue:
        logger.info("No experiments remaining. Done!")
        state.start_time = state.start_time or datetime.now(timezone.utc).isoformat()
        print_summary(state, None)
        return

    # Run experiments
    baseline_bpb = None
    train_file = PROJECT_ROOT / "train.py"

    # MLX: for baseline mode, just run train_mlx.py directly (no config changes)
    if args.mode == "baseline":
        logger.info(f"Running baseline experiment (engine={_current_engine})...")
        # Use the engine's default script without config mutations
        baseline_result = run_training(timeout=args.timeout, engine=_current_engine)
        logger.info(f"Baseline result: {baseline_result['status']} val_bpb={baseline_result['val_bpb']}")
        if baseline_result["val_bpb"] is not None:
            baseline_bpb = baseline_result["val_bpb"]
            append_tsv(
                experiment_id="baseline_001",
                experiment_name="Baseline",
                val_bpb=baseline_bpb,
                delta=None,
                phase=args.phase,
                status=baseline_result["status"],
                duration=baseline_result["duration"],
                changes="none",
                notes="Initial baseline measurement",
                engine=_current_engine,
            )
        state.experiments_completed = 1
        state.best_val_bpb = baseline_bpb
        state.best_experiment_id = "baseline_001"
        save_state(state)
        print_summary(state, baseline_bpb)
        return

    # For non-baseline modes, we mutate the config and run experiments
    start_time = time.time()

    for idx, experiment in enumerate(queue):
        if _interrupted:
            logger.info("Interrupt detected - saving state before exit")
            state.pending_experiments = queue[idx:]
            state.interrupted = True
            save_state(state)
            break

        exp_id = experiment.get("experiment_id", f"exp_{idx:03d}")
        changes = experiment.get("changes", {})
        hypothesis = experiment.get("hypothesis", "")
        category = experiment.get("category", "unknown")

        changes_str = ", ".join(f"{k}={v}" for k, v in changes.items())
        logger.info(f"\n--- Experiment {idx+1}/{len(queue)}: {exp_id} [{category}] ({_current_engine}) ---")
        logger.info(f"  Changes: {changes_str}")
        logger.info(f"  Hypothesis: {hypothesis}")

        # Step 1: Apply changes (only for pytorch, or if target file exists)
        applied = None
        if changes:
            applied = apply_changes_to_train(changes, engine=_current_engine)
            if applied is None:
                logger.error("Failed to apply changes, skipping experiment")
                state.experiments_failed += 1
                state.completed_ids.add(exp_id)
                append_tsv(exp_id, category, None, None, args.phase, "failed", 0, changes_str,
                           "Failed to apply changes", engine=_current_engine)
                save_state(state)
                continue

        # Step 2: Run training
        logger.info(f"  Running {_current_engine} training (timeout: {args.timeout}s)...")
        result = run_training(timeout=args.timeout, engine=_current_engine)

        # Step 3: Record result
        val_bpb = result["val_bpb"]
        delta = None
        if baseline_bpb is not None and val_bpb is not None:
            delta = baseline_bpb - val_bpb

        # Update best
        if val_bpb is not None:
            if state.best_val_bpb is None or val_bpb < state.best_val_bpb:
                state.best_val_bpb = val_bpb
                state.best_experiment_id = exp_id

            # Record in knowledge base
            kb_record_result = record_in_knowledge(
                args.knowledge,
                exp_id,
                changes,
                val_bpb,
                result["status"],
                applied or changes_str,
            )

        status = result["status"]
        is_success = status == "success" and val_bpb is not None

        if is_success:
            state.experiments_completed += 1
        else:
            state.experiments_failed += 1
            logger.warning(f"  Experiment failed: {result.get('error', status)}")

        state.experiment_count = idx + 1
        state.last_experiment_id = exp_id
        state.last_val_bpb = val_bpb
        state.completed_ids.add(exp_id)

        # Append to TSV
        append_tsv(
            experiment_id=exp_id,
            experiment_name=f"{category}:{changes_str}",
            val_bpb=val_bpb,
            delta=delta,
            phase=args.phase,
            status=status,
            duration=result["duration"],
            changes=changes_str,
            notes=hypothesis[:200] if hypothesis else "",
            engine=_current_engine,
        )

        # Print progress
        print_progress(
            current=state.experiments_completed + state.experiments_failed,
            total=len(queue),
            experiment_id=exp_id,
            experiment_name=category,
            val_bpb=val_bpb,
            delta=delta,
            best_bpb=state.best_val_bpb,
            duration=result["duration"],
            status=status,
            start_time=start_time,
        )

        # Step 4: Meta-analysis at intervals
        total_done = state.experiments_completed + state.experiments_failed
        if total_done > 0 and total_done % args.meta_interval == 0:
            logger.info(f"--- Meta-analysis at experiment {total_done} ---")
            run_meta_analyzer(str(TSV_FILE), f"phase{args.phase}_exp{total_done}")
            run_dashboard(str(TSV_FILE), str(PLOTS_DIR))

        # Step 5: Self-improve at intervals
        if args.mode == "recursive" and total_done > 0 and total_done % args.self_improve_interval == 0:
            logger.info(f"--- Self-improve cycle at experiment {total_done} ---")
            self_improve_result = run_self_improve(str(TSV_FILE), args.knowledge)
            if self_improve_result.get("status") == "success":
                logger.info("Self-improve applied - train script updated")
                if state.best_val_bpb is not None:
                    baseline_bpb = state.best_val_bpb
                    logger.info(f"New baseline set to {baseline_bpb:.6f}")
            else:
                logger.warning(f"Self-improve failed: {self_improve_result.get('error', 'unknown')}")

        # Deep mode: run meta-analysis more frequently
        if args.mode == "deep" and total_done > 0 and total_done % 10 == 0:
            logger.info(f"--- Deep analysis at experiment {total_done} ---")
            run_meta_analyzer(str(TSV_FILE), f"deep_{total_done}")

        # Save checkpoint
        save_state(state)

        # Reset training script for next experiment
        git_reset_train(_current_engine)

    # Final summary
    if _interrupted:
        logger.info("Run was interrupted; state saved for resume")

    # Final meta-analysis and dashboard
    if state.experiments_completed > 0:
        run_meta_analyzer(str(TSV_FILE), "final")
        run_dashboard(str(TSV_FILE), str(PLOTS_DIR))

    print_summary(state, baseline_bpb)


def record_in_knowledge(
    kb_path: str,
    exp_id: str,
    changes: Dict[str, Any],
    val_bpb: float,
    status: str,
    notes: str,
) -> Optional[str]:
    """Record experiment result in the knowledge base."""
    kb_script = SCRIPT_DIR / "scripts" / "knowledge_base.py"
    if not kb_script.exists():
        return None

    status_map = {
        "success": "confirmed",
        "failed": "failed",
        "crashed": "failed",
        "timeout": "failed",
    }
    kb_status = status_map.get(status, "tentative")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("knowledge_base", kb_script)
        kb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kb_module)

        KB = kb_module.KnowledgeBase
        kb = KB(path=kb_path)
        record = kb.record_result(exp_id, changes, val_bpb, kb_status, notes)
        return record.get("id", exp_id)
    except Exception as e:
        logger.warning(f"Failed to record in knowledge base: {e}")
        return None


if __name__ == "__main__":
    main()
