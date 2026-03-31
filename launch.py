#!/usr/bin/env python3
"""
Autoresearch v2 — Recursive Self-Improving ML Research System

Master entry point that orchestrates all research components:
  - knowledge_base.py     (persistent memory across sessions)
  - research_orchestrator.py  (strategic experiment management)
  - experiment_designer.py    (systematic plan generation)
  - meta_analyzer.py          (pattern discovery from results)
  - dashboard.py              (visual analytics)
  - self_improve.py           (recursive process improvement)

Usage:
    # Quick modes
    python launch.py --mode baseline           # Single baseline run
    python launch.py --mode single -n 10       # Run N experiments
    python launch.py --mode night              # ~100 experiments (overnight)
    python launch.py --mode deep               # With meta-analysis every 20
    python launch.py --mode recursive          # Full self-improving loop

    # Advanced
    python launch.py --mode single -n 50 --branch autoresearch/v2-night3
    python launch.py --mode recursive --max-experiments 200
    python launch.py --resume                  # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
STATE_FILE = ROOT / "state.json"

# Ensure directories exist
for d in [RESULTS_DIR, PLOTS_DIR, SCRIPTS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("autoresearch-v2")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file = ROOT / "logs" / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(str(log_file)))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    LOGGER.addHandler(logging.getLogger())


# ---------------------------------------------------------------------------
# State management (checkpoint / resume)
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return _default_state()


def save_state(state: dict[str, Any]) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, str(STATE_FILE))


def _default_state() -> dict[str, Any]:
    return {
        "mode": None,
        "branch": None,
        "experiment_count": 0,
        "best_val_bpb": None,
        "best_config": None,
        "phase": 1,
        "last_meta_analysis_at": 0,
        "last_self_improve_at": 0,
        "status": "idle",
        "started_at": None,
    }


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True,
        cwd=ROOT, check=check,
    )


def ensure_branch(branch: str) -> None:
    """Create or check out the research branch."""
    try:
        run_git("checkout", branch)
        LOGGER.info("Already on branch %s", branch)
    except subprocess.CalledProcessError:
        run_git("checkout", "-b", branch)
        LOGGER.info("Created new branch %s", branch)


def commit_changes(message: str) -> str | None:
    """Stage train.py changes and commit. Returns short hash or None."""
    run_git("add", "train.py")
    result = run_git("diff", "--staged", "--quiet", check=False)
    if result.returncode == 0:
        LOGGER.warning("No changes to commit (diff is empty)")
        return None
    run_git("commit", "-m", message)
    short = run_git("rev-parse", "--short=7", "HEAD").stdout.strip()
    LOGGER.info("Committed: [%s] %s", short, message)
    return short


def reset_to_commit(h: str) -> None:
    """Hard reset to a known-good commit."""
    run_git("reset", "--hard", h)
    LOGGER.info("Reset to %s", h)


def get_current_commit() -> str:
    r = run_git("rev-parse", "--short=7", "HEAD")
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(timeout: int = 620) -> dict[str, Any] | None:
    """
    Run uv run train.py with timeout.
    Returns parsed metrics dict on success, None on failure/timeout.
    """
    logf = ROOT / "run.log"
    env = os.environ.copy()
    start = time.monotonic()

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "uv", "run", "train.py"],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=ROOT,
            env=env,
        )
        logf.write_text(proc.stdout + proc.stderr)
        LOGGER.debug("Training exit code: %s (%.1fs)", proc.returncode, time.monotonic() - start)

        if proc.returncode != 0:
            LOGGER.warning("Training failed (rc=%d). Last 30 lines:", proc.returncode)
            for line in (proc.stdout + proc.stderr).splitlines()[-30:]:
                LOGGER.warning("  %s", line)
            return None

    except subprocess.TimeoutExpired:
        LOGGER.warning("Training timed out after %ds", timeout)
        return None
    except Exception as e:
        LOGGER.error("Training crash: %s", e)
        return None

    # Parse output
    text = logf.read_text()
    metrics: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("val_bpb:"):
            try:
                metrics["val_bpb"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("peak_vram_mb:"):
            try:
                metrics["peak_vram_mb"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("training_seconds:"):
            try:
                metrics["training_seconds"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("num_steps:"):
            try:
                metrics["num_steps"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("depth:"):
            try:
                metrics["depth"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if "val_bpb" in metrics:
        return metrics
    LOGGER.warning("Could not parse val_bpb from training output")
    return None


# ---------------------------------------------------------------------------
# Component runners (shell out to scripts)
# ---------------------------------------------------------------------------

def run_script(name: str, args: list[str], capture: bool = False) -> str | None:
    """Run a script from the scripts/ directory."""
    script = SCRIPTS / name
    if not script.exists():
        LOGGER.error("Script not found: %s", script)
        return None
    cmd = [sys.executable, str(script), *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=120)
        out = result.stdout
        if result.returncode != 0:
            LOGGER.warning("Script %s failed (rc=%d): %s", name, result.returncode, result.stderr[:300])
        if capture:
            return out
        if out:
            LOGGER.debug("  %s output: %s", name, out[:200])
        return out
    except subprocess.TimeoutExpired:
        LOGGER.warning("Script %s timed out", name)
        return None
    except Exception as e:
        LOGGER.error("Script %s crashed: %s", name, e)
        return None


def run_meta_analysis() -> None:
    LOGGER.info("Running meta-analysis...")
    run_script("meta_analyzer.py", [
        "--results", str(RESULTS_DIR / "results.tsv"),
        "--output", str(RESULTS_DIR / "analysis.md"),
    ])
    LOGGER.info("Meta-analysis complete.")


def run_dashboard() -> None:
    LOGGER.info("Generating dashboard...")
    run_script("dashboard.py", [
        "--results", str(RESULTS_DIR / "results.tsv"),
        "--output-md", str(ROOT / "report.md"),
        "--plots-dir", str(PLOTS_DIR),
    ])
    LOGGER.info("Dashboard updated.")


def run_self_improvement(experiment_count: int) -> None:
    LOGGER.info("Running self-improvement cycle...")
    run_script("self_improve.py", [
        "--experiments-done", str(experiment_count),
        "--results-dir", str(RESULTS_DIR),
        "--knowledge", str(ROOT / "knowledge.json"),
    ])
    LOGGER.info("Self-improvement complete.")


def run_baseline_check(best_before: float | None) -> bool:
    """Re-run baseline to check for metric drift. Returns True if stable."""
    LOGGER.info("Running baseline check...")
    # Reset train.py to baseline (would need to track this; simplify:
    # just run current train.py and compare)
    metrics = run_training(timeout=620)
    if metrics is None:
        LOGGER.warning("Baseline check failed")
        return True  # don't block on this
    after = metrics.get("val_bpb")
    if best_before is not None and after is not None:
        drift = abs(after - best_before)
        LOGGER.info("Baseline drift: %.4f (before=%.4f, after=%.4f)", drift, best_before, after)
        if drift > 0.01:
            LOGGER.warning("BASELINE DRIFT EXCEEDS THRESHOLD! Investigate.")
            return False
    return True


# ---------------------------------------------------------------------------
# Results recording
# ---------------------------------------------------------------------------

def append_tsv(commit_hash: str, val_bpb: float, memory_gb: float,
               status: str, description: str) -> None:
    tsv = RESULTS_DIR / "results.tsv"
    write_header = not tsv.exists()
    with open(tsv, "a") as f:
        if write_header:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        f.write(f"{commit_hash}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def update_knowledge(exp_id: str, config: dict, val_bpb: float,
                     status: str, description: str) -> None:
    """Record result in the knowledge base."""
    kb_module_path = str(SCRIPTS)
    if kb_module_path not in sys.path:
        sys.path.insert(0, kb_module_path)

    try:
        from knowledge_base import KnowledgeBase
        kb_path = ROOT / "knowledge.json"
        kb = KnowledgeBase(path=str(kb_path))
        kb.record_result(exp_id, config, val_bpb, status, description)
        kb.save()
    except Exception as e:
        LOGGER.error("Failed to update knowledge base: %s", e)
        # Fallback: just append to TSV
        LOGGER.info("Using TSV fallback instead")


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def get_next_experiment(state: dict, design: dict | None) -> dict | None:
    """
    Ask the experiment designer / knowledge base for the next experiment.
    Returns a dict with keys: {id, config, description, hypothesis, category}
    """
    kb_path = ROOT / "knowledge.json"
    exp_plan = ROOT / "scripts" / "experiments.json"

    # Prefer knowledge base suggestions
    if kb_path.exists():
        try:
            with open(kb_path) as f:
                kb_data = json.load(f)
            suggestions = kb_data.get("experiments", [])
            # Find first untried
            tried = {e["id"] for e in kb_data.get("experiments", [])
                     if "id" in e}
            # Pull from experiment plan
            if exp_plan.exists():
                with open(exp_plan) as f:
                    plans = json.load(f)
                if isinstance(plans, dict):
                    plans = plans.get("experiments", [])
                for p in plans:
                    pid = p.get("id", "")
                    if pid not in tried:
                        return {
                            "id": pid,
                            "config": p.get("changes", {}),
                            "description": p.get("hypothesis", p.get("name", "")),
                            "hypothesis": p.get("hypothesis", ""),
                            "category": p.get("category", "unknown"),
                        }
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # Fallback: generate simple next experiment
    n = state["experiment_count"] + 1
    phase = state["phase"]

    if phase == 1:
        configs = [
            {"name": "depth_12",   "changes": {"DEPTH": 12}, "desc": "Increase depth 8→12"},
            {"name": "depth_10",   "changes": {"DEPTH": 10}, "desc": "Increase depth 8→10"},
            {"name": "depth_6",    "changes": {"DEPTH": 6},  "desc": "Decrease depth 8→6"},
            {"name": "lr_0.03",    "changes": {"MATRIX_LR": 0.03}, "desc": "Matrix LR 0.04→0.03"},
            {"name": "lr_0.05",    "changes": {"MATRIX_LR": 0.05}, "desc": "Matrix LR 0.04→0.05"},
            {"name": "batch_256",  "changes": {"DEVICE_BATCH_SIZE": 256}, "desc": "Batch size 128→256"},
            {"name": "batch_64",   "changes": {"DEVICE_BATCH_SIZE": 64},  "desc": "Batch size 128→64"},
            {"name": "warmup_005", "changes": {"WARMUP_RATIO": 0.05},     "desc": "Add 5% warmup"},
            {"name": "wd_0.1",     "changes": {"WEIGHT_DECAY": 0.1},      "desc": "Weight decay 0.2→0.1"},
            {"name": "wd_0.3",     "changes": {"WEIGHT_DECAY": 0.3},      "desc": "Weight decay 0.2→0.3"},
        ]
        idx = (n - 1) % len(configs)
        c = configs[idx]
        return {"id": f"exp_{n:03d}", "config": c["changes"],
                "description": c["desc"], "hypothesis": c["desc"],
                "category": _category_for(c["changes"])}
    elif phase == 2:
        return {"id": f"exp_{n:03d}", "config": {},
                "description": f"Combination experiment #{n}",
                "hypothesis": "Combine top improvements",
                "category": "combination"}
    else:
        return {"id": f"exp_{n:03d}", "config": {},
                "description": f"Radical experiment #{n}",
                "hypothesis": "Try architectural change",
                "category": "radical"}


def _category_for(changes: dict) -> str:
    arch_keys = {"DEPTH", "HEAD_DIM", "ASPECT_RATIO", "BLOCK_TYPE", "MLP_TYPE",
                 "N_KV_HEAD", "WINDOW_PATTERN"}
    opt_keys = {"MATRIX_LR", "EMBEDDING_LR", "WEIGHT_DECAY", "WARMUP_RATIO",
                "WARMDOWN_RATIO", "ADAM_BETAS"}
    train_keys = {"DEVICE_BATCH_SIZE", "TOTAL_BATCH_SIZE", "GRAD_CLIP"}
    reg_keys = {"DROPOUT", "LABEL_SMOOTHING"}

    keys = set(changes.keys())
    if keys & arch_keys:
        return "architecture"
    if keys & opt_keys:
        return "optimization"
    if keys & train_keys:
        return "training"
    if keys & reg_keys:
        return "regularization"
    return "novel"


def apply_experiment_config(experiment: dict) -> bool:
    """
    Apply the experiment's config changes to train.py.
    Uses patch-based replacement for each parameter.
    """
    changes = experiment.get("config", {})
    if not changes:
        LOGGER.warning("No config changes for experiment %s", experiment.get("id"))
        return False

    train_py = ROOT / "train.py"
    content = train_py.read_text()
    original = content

    # Parameter replacement map: key -> (old_pattern, new_value)
    PARAM_PATTERNS = {
        "DEPTH": (
            r"^DEPTH\s*=\s*\d+",
            lambda v: f"DEPTH = {v}"
        ),
        "ASPECT_RATIO": (
            r"^ASPECT_RATIO\s*=\s*\d+",
            lambda v: f"ASPECT_RATIO = {v}"
        ),
        "HEAD_DIM": (
            r"^HEAD_DIM\s*=\s*\d+",
            lambda v: f"HEAD_DIM = {v}"
        ),
        "WINDOW_PATTERN": (
            r'^WINDOW_PATTERN\s*=\s*"[^"]*"',
            lambda v: f'WINDOW_PATTERN = "{v}"'
        ),
        "MATRIX_LR": (
            r"^MATRIX_LR\s*=\s*[\d.]+",
            lambda v: f"MATRIX_LR = {v}"
        ),
        "EMBEDDING_LR": (
            r"^EMBEDDING_LR\s*=\s*[\d.]+",
            lambda v: f"EMBEDDING_LR = {v}"
        ),
        "WEIGHT_DECAY": (
            r"^WEIGHT_DECAY\s*=\s*[\d.]+",
            lambda v: f"WEIGHT_DECAY = {v}"
        ),
        "WARMUP_RATIO": (
            r"^WARMUP_RATIO\s*=\s*[\d.]+",
            lambda v: f"WARMUP_RATIO = {v}"
        ),
        "WARMDOWN_RATIO": (
            r"^WARMDOWN_RATIO\s*=\s*[\d.]+",
            lambda v: f"WARMDOWN_RATIO = {v}"
        ),
        "DEVICE_BATCH_SIZE": (
            r"^DEVICE_BATCH_SIZE\s*=\s*\d+",
            lambda v: f"DEVICE_BATCH_SIZE = {v}"
        ),
        "TOTAL_BATCH_SIZE": (
            r"^TOTAL_BATCH_SIZE\s*=\s*[\d*]+",
            lambda v: f"TOTAL_BATCH_SIZE = {v}"
        ),
        "ADAM_BETAS": (
            r"^ADAM_BETAS\s*=\s*\([^)]*\)",
            lambda v: f"ADAM_BETAS = {v}"
        ),
    }

    import re

    for key, value in changes.items():
        if key in PARAM_PATTERNS:
            pattern, formatter = PARAM_PATTERNS[key]
            new_line = formatter(value)
            # Try matching the line
            flags = re.MULTILINE
            if re.search(pattern, content, flags):
                content = re.sub(pattern, new_line, content, count=1, flags=flags)
                LOGGER.info("  Applied %s = %s", key, value)
            else:
                LOGGER.warning("  Could not find pattern for %s in train.py", key)
        else:
            LOGGER.warning("  Unknown parameter %s (skipping)", key)

    if content == original:
        LOGGER.warning("  No changes applied to train.py!")
        return False

    train_py.write_text(content)
    return True


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def run_baseline() -> dict | None:
    """Run a single baseline experiment."""
    LOGGER.info("=" * 60)
    LOGGER.info("BASELINE RUN")
    LOGGER.info("=" * 60)
    return run_training(timeout=620)


def run_loop(state: dict, max_experiments: int, phase: int = 1) -> None:
    """
    Main experiment loop.
    """
    state["phase"] = phase
    baseline_val_bpb = None

    # Read last known best from TSV
    tsv = RESULTS_DIR / "results.tsv"
    if tsv.exists():
        for line in reversed(tsv.read_text().strip().splitlines()):
            if line.startswith("commit"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4 and parts[3] == "keep":
                try:
                    baseline_val_bpb = float(parts[1])
                except ValueError:
                    pass
                break

    LOGGER.info("=" * 60)
    LOGGER.info("RESEARCH LOOP STARTING")
    LOGGER.info("  Phase: %d", phase)
    LOGGER.info("  Max experiments: %d", max_experiments)
    LOGGER.info("  Starting from experiment: %d", state["experiment_count"] + 1)
    if baseline_val_bpb is not None:
        LOGGER.info("  Current best val_bpb: %.6f", baseline_val_bpb)
    LOGGER.info("=" * 60)

    best_val = baseline_val_bpb
    start_time = time.monotonic()

    try:
        for _ in range(max_experiments):
            n = state["experiment_count"] + 1

            # Periodic tasks
            if n > 1 and (n - 1) % 20 == 0:
                LOGGER.info("\n--- Periodic analysis at experiment %d ---", n)
                run_meta_analysis()
                run_dashboard()

            if n > 1 and (n - 1) % 50 == 0:
                LOGGER.info("\n=== SELF-IMPROVEMENT CYCLE at experiment %d ===", n)
                run_self_improvement(n)

            # Get next experiment plan
            experiment = get_next_experiment(state, None)
            if experiment is None:
                LOGGER.warning("No experiment plan available, stopping.")
                break

            LOGGER.info("\n%s Experiment %d: %s [%s]",
                        "=" * 40, n, experiment.get("description", ""),
                        experiment.get("category", ""))
            if experiment.get("hypothesis"):
                LOGGER.info("  Hypothesis: %s", experiment["hypothesis"])
            LOGGER.info("  Config: %s", experiment.get("config", {}))

            # Save pre-experiment train.py for recovery
            train_py = ROOT / "train.py"
            pre_content = train_py.read_text()

            # Apply changes
            if experiment.get("config"):
                ok = apply_experiment_config(experiment)
                if not ok:
                    LOGGER.warning("Failed to apply changes, skipping.")
                    append_tsv("skip", 0.0, 0.0, "skip", experiment.get("description", ""))
                    state["experiment_count"] += 1
                    save_state(state)
                    continue

            # Commit
            commit_msg = f"exp_{n:03d}: [{experiment.get('category', '?')}] {experiment.get('description', '')}"
            commit_hash = commit_changes(commit_msg)
            if commit_hash is None:
                commit_hash = f"skip_{n}"

            # Run training
            LOGGER.info("  Starting training (timeout: 10 min)...")
            metrics = run_training(timeout=620)

            if metrics is None:
                LOGGER.warning("  CRASH / TIMEOUT — reverting")
                status = "crash"
                val_bpb = 0.0
                memory_gb = 0.0
            else:
                val_bpb = metrics.get("val_bpb", 0.0)
                memory_gb = metrics.get("peak_vram_mb", 0.0) / 1024
                elapsed = time.monotonic() - start_time
                eta = "?"
                if n > 0:
                    avg_time = elapsed / n
                    remaining = (max_experiments - n) * avg_time
                    eta = str(timedelta(seconds=int(remaining)))

                LOGGER.info("  RESULT: val_bpb=%.6f  VRAM=%.1fGB  "
                            "steps=%s  (%d/%d done, ETA: %s)",
                            val_bpb, memory_gb,
                            metrics.get("num_steps", "?"),
                            n, max_experiments, eta)

                if best_val is not None:
                    delta = -(val_bpb - best_val)
                    if delta > 0.001:
                        status = "keep"
                        best_val = val_bpb
                        state["best_val_bpb"] = best_val
                        LOGGER.info("  ** IMPROVEMENT! Delta: +%.6f", delta)
                    else:
                        status = "discard"
                        LOGGER.info("  No improvement (delta: %.6f)", delta)
                else:
                    status = "keep"
                    best_val = val_bpb
                    state["best_val_bpb"] = best_val

            # Record results
            append_tsv(commit_hash, val_bpb, memory_gb, status,
                      experiment.get("description", ""))
            update_knowledge(
                experiment.get("id", f"exp_{n:03d}"),
                experiment.get("config", {}),
                val_bpb,
                status,
                experiment.get("description", ""),
            )

            # Revert on discard/crash
            if status in ("discard", "crash"):
                commit_hash_for_reset = get_current_commit()
                # Find previous good commit
                try:
                    parent = run_git("rev-parse", "HEAD~1").stdout.strip()
                    reset_to_commit(parent)
                except subprocess.CalledProcessError:
                    # If reset fails, restore file content
                    train_py.write_text(pre_content)
                    run_git("checkout", "train.py", check=False)

            # Update state
            state["experiment_count"] = n
            state["status"] = "running"
            save_state(state)

    except KeyboardInterrupt:
        LOGGER.info("\nInterrupted! Saving state for resume...")
        state["status"] = "interrupted"
        save_state(state)
        LOGGER.info("State saved. Resume with: python launch.py --resume")

    # Final summary
    end_time = time.monotonic()
    total_time = end_time - start_time
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("RESEARCH SESSION COMPLETE")
    LOGGER.info("  Experiments: %d", state["experiment_count"])
    LOGGER.info("  Duration: %s", str(timedelta(seconds=int(total_time))))
    LOGGER.info("  Best val_bpb: %s", state.get("best_val_bpb", "N/A"))
    LOGGER.info("  Avg time/experiment: %.1fs",
                total_time / max(state["experiment_count"], 1))
    LOGGER.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch v2 — Recursive Self-Improving ML Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", choices=["baseline", "single", "night", "deep", "recursive"],
                        default="single", help="Research mode")
    parser.add_argument("-n", "--num-experiments", type=int, default=10,
                        help="Number of experiments (single/deep mode)")
    parser.add_argument("--branch", type=str, default=None,
                        help="Git branch name")
    parser.add_argument("--phase", type=int, default=1,
                        help="Research phase (1-4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--timeout", type=int, default=620,
                        help="Training timeout in seconds")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Override max experiments for any mode")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Determine branch
    branch = args.branch
    if branch is None:
        try:
            r = run_git("rev-parse", "--abbrev-ref", "HEAD", check=False)
            branch = r.stdout.strip()
        except Exception:
            branch = "autoresearch/v2"
    if args.mode in ("night", "deep", "recursive"):
        branch = f"autoresearch/v2-{datetime.now(timezone.utc):%b%d}"

    state = load_state() if args.resume else _default_state()
    state["mode"] = args.mode
    state["branch"] = branch
    state["started_at"] = datetime.now(timezone.utc).isoformat()

    max_exp = args.max_experiments or {
        "baseline": 0,
        "single": args.num_experiments,
        "night": 100,
        "deep": 200,
        "recursive": 500,
    }.get(args.mode, 10)

    LOGGER.info("Starting Autoresearch v2")
    LOGGER.info("  Mode: %s | Branch: %s | Phase: %d | Max experiments: %d",
                args.mode, branch, args.phase, max_exp)

    # Ensure git branch
    try:
        ensure_branch(branch)
    except Exception as e:
        LOGGER.error("Git setup failed: %s", e)
        sys.exit(1)

    if args.mode == "baseline":
        metrics = run_baseline()
        if metrics:
            LOGGER.info("Baseline val_bpb: %.6f", metrics["val_bpb"])
            LOGGER.info("Baseline VRAM: %.1f MB", metrics.get("peak_vram_mb", 0))
            append_tsv(get_current_commit(), metrics["val_bpb"],
                      metrics.get("peak_vram_mb", 0) / 1024,
                      "keep", "baseline")
        sys.exit(0 if metrics else 1)

    # Signal handler for graceful shutdown
    def sigint_handler(sig, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, sigint_handler)

    # Run the loop
    run_loop(state, max_exp, phase=args.phase)


if __name__ == "__main__":
    main()
