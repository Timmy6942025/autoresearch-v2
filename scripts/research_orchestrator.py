#!/usr/bin/env python3
"""
SYSTEMATIC RESEARCH ORCHESTRATOR v2
Replaces simple hill-climbing with sophisticated multi-strategy research system.

Usage:
    python research_orchestrator.py --phase 1 --branch autoresearch/v2 --max-experiments 100
    python research_orchestrator.py --phase 2 --from-results results.json
    python research_orchestrator.py --analyze --results-dir results/
"""

import argparse
import copy
import datetime
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

DEFAULT_BASELINE_BPB = 1.45
IMPROVEMENT_THRESHOLD = 0.001
CRASH_THRESHOLD = 3
BASELINE_RECHECK_INTERVAL = 10
DEFAULT_TRAIN_TIMEOUT = 300  # 5 minutes default

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "orchestrator.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("research_orchestrator")


# ─── Data Structures ─────────────────────────────────────────────────────────


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CRASHED = "crashed"
    SKIPPED = "skipped"
    BASELINE = "baseline"


class ChangeType(Enum):
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURAL = "architectural"
    TRAINING_LOOP = "training_loop"
    DATA_PIPELINE = "data_pipeline"
    REGULARIZATION = "regularization"


@dataclass
class ExperimentConfig:
    """Individual hyperparameter change or combination."""
    name: str
    change_type: ChangeType
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    template_file: Optional[str] = None
    patch_content: Optional[str] = None
    expected_improvement: float = 0.0
    risk_level: str = "medium"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    experiment_name: str
    configuration: Dict[str, Any]
    val_bpb: Optional[float]
    status: str
    description: str
    phase: int
    git_commit: Optional[str] = None
    git_diff_hash: Optional[str] = None
    baseline_bpb: Optional[float] = None
    delta: Optional[float] = None
    timestamp: str = ""
    duration_seconds: float = 0.0
    crash_count: int = 0
    changes_applied: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()
        if self.baseline_bpb and self.val_bpb is not None:
            self.delta = round(self.baseline_bpb - self.val_bpb, 6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── Results Database ─────────────────────────────────────────────────────────


class ResultsDatabase:
    """Persistent JSON-based results tracking with analysis capabilities."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or RESULTS_DIR / "research_results.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.baseline_history: List[float] = []
        self.crash_log: Dict[str, int] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.results = [
                    ExperimentResult.from_dict(r) for r in data.get("results", [])
                ]
                self.baseline_history = data.get("baseline_history", [])
                self.crash_log = {k: int(v) for k, v in data.get("crash_log", {}).items()}
                logger.info(f"Loaded {len(self.results)} results from {self.db_path}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load results database: {e}")
                self.results = []

    def _save(self):
        data = {
            "results": [r.to_dict() for r in self.results],
            "baseline_history": self.baseline_history,
            "crash_log": self.crash_log,
            "metadata": {
                "total_experiments": len(
                    [r for r in self.results if r.status != ExperimentStatus.BASELINE.value]
                ),
                "successful_experiments": len(
                    [r for r in self.results if r.status == ExperimentStatus.SUCCESS.value]
                ),
                "failed_experiments": len(
                    [r for r in self.results
                     if r.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value)]
                ),
                "last_updated": datetime.datetime.now().isoformat(),
                "best_val_bpb": self.get_best_bpb(),
                "best_experiment": self.get_best_experiment_id(),
            },
        }
        tmp_path = self.db_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.rename(self.db_path)

    def add_result(self, result: ExperimentResult):
        self.results.append(result)
        if result.status == ExperimentStatus.BASELINE.value and result.val_bpb:
            self.baseline_history.append(result.val_bpb)
        self._save()

    def update_crash_count(self, change_name: str, increment: int = 1):
        self.crash_log[change_name] = self.crash_log.get(change_name, 0) + increment
        self._save()

    def get_crash_count(self, change_name: str) -> int:
        return self.crash_log.get(change_name, 0)

    def is_permanently_broken(self, change_name: str) -> bool:
        return self.get_crash_count(change_name) >= CRASH_THRESHOLD

    def get_by_phase(self, phase: int) -> List[ExperimentResult]:
        return [r for r in self.results if r.phase == phase]

    def get_by_status(self, status: str) -> List[ExperimentResult]:
        return [r for r in self.results if r.status == status]

    def get_successful(self) -> List[ExperimentResult]:
        return [
            r for r in self.results
            if r.status == ExperimentStatus.SUCCESS.value
            and r.delta is not None
            and r.delta > 0
        ]

    def get_best_bpb(self) -> Optional[float]:
        successful = self.get_successful()
        if not successful:
            return None
        return min(r.val_bpb for r in successful if r.val_bpb is not None)

    def get_best_experiment_id(self) -> Optional[str]:
        successful = self.get_successful()
        if not successful:
            return None
        best = min(successful, key=lambda r: r.val_bpb if r.val_bpb is not None else float("inf"))
        return best.experiment_id

    def get_current_baseline(self) -> Optional[float]:
        if self.baseline_history:
            return self.baseline_history[-1]
        baseline_results = [
            r for r in self.results
            if r.status == ExperimentStatus.BASELINE.value and r.val_bpb
        ]
        if baseline_results:
            return baseline_results[-1].val_bpb
        return DEFAULT_BASELINE_BPB

    def get_experiments_with_change(self, change_name: str) -> List[ExperimentResult]:
        return [r for r in self.results if change_name in r.changes_applied]

    def get_all_changes(self) -> Set[str]:
        changes = set()
        for r in self.results:
            changes.update(r.changes_applied)
        return changes

    def export_tsv(self, path: Optional[Path] = None):
        path = path or RESULTS_DIR / "results.tsv"
        with open(path, "w") as f:
            f.write(
                "experiment_id\texperiment_name\tval_bpb\tdelta\tphase\tstatus\t"
                "timestamp\tduration\tchanges\n"
            )
            for r in sorted(self.results, key=lambda x: x.timestamp):
                changes = "|".join(r.changes_applied)
                f.write(
                    f"{r.experiment_id}\t{r.experiment_name}\t{r.val_bpb or 'N/A'}\t"
                    f"{r.delta or 'N/A'}\t{r.phase}\t{r.status}\t{r.timestamp}\t"
                    f"{r.duration_seconds:.1f}\t{changes}\n"
                )
        logger.info(f"Exported TSV to {path}")

    def analyze(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results to analyze"}

        successful = self.get_successful()
        failed = self.get_by_status(ExperimentStatus.FAILED.value)
        crashed = self.get_by_status(ExperimentStatus.CRASHED.value)

        phase_results = defaultdict(list)
        for r in self.results:
            phase_results[r.phase].append(r)

        change_stats = defaultdict(list)
        for r in successful:
            for change in r.changes_applied:
                if r.delta is not None:
                    change_stats[change].append(r.delta)

        change_effectiveness = {}
        for change, deltas in change_stats.items():
            change_effectiveness[change] = {
                "avg_improvement": round(sum(deltas) / len(deltas), 6),
                "max_improvement": round(max(deltas), 6),
                "min_improvement": round(min(deltas), 6),
                "num_experiments": len(deltas),
                "success_rate": round(
                    len([d for d in deltas if d > 0]) / max(len(deltas), 1) * 100, 1
                ),
            }

        baseline_stability = {}
        if len(self.baseline_history) >= 2:
            mean_bl = sum(self.baseline_history) / len(self.baseline_history)
            var_bl = sum((x - mean_bl) ** 2 for x in self.baseline_history) / len(self.baseline_history)
            baseline_stability = {
                "mean": round(mean_bl, 6),
                "std": round(math.sqrt(var_bl), 6),
                "min": round(min(self.baseline_history), 6),
                "max": round(max(self.baseline_history), 6),
                "measurements": len(self.baseline_history),
            }

        combo_scores = defaultdict(list)
        for r in successful:
            if len(r.changes_applied) > 1:
                key = tuple(sorted(r.changes_applied))
                if r.delta is not None:
                    combo_scores[key].append(r.delta)

        top_combos = sorted(
            [{"combination": list(k), "avg_delta": round(sum(v) / len(v), 6)}
             for k, v in combo_scores.items()],
            key=lambda x: x["avg_delta"],
            reverse=True,
        )[:10]

        return {
            "summary": {
                "total_experiments": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "crashed": len(crashed),
                "skipped": len(self.get_by_status(ExperimentStatus.SKIPPED.value)),
                "best_val_bpb": self.get_best_bpb(),
                "best_experiment": self.get_best_experiment_id(),
                "current_baseline": self.get_current_baseline(),
            },
            "by_phase": {
                phase: {
                    "total": len(results),
                    "successful": len([r for r in results if r.status == ExperimentStatus.SUCCESS.value]),
                    "failed": len([r for r in results if r.status == ExperimentStatus.FAILED.value]),
                    "avg_delta": round(
                        sum(r.delta or 0 for r in results if r.status == ExperimentStatus.SUCCESS.value)
                        / max(1, len([r for r in results if r.status == ExperimentStatus.SUCCESS.value])),
                        6,
                    ),
                }
                for phase, results in phase_results.items()
            },
            "change_effectiveness": dict(
                sorted(change_effectiveness.items(), key=lambda x: x[1]["avg_improvement"], reverse=True)
            ),
            "top_combinations": top_combos,
            "baseline_stability": baseline_stability,
            "crash_prone_changes": {
                k: v for k, v in self.crash_log.items() if v >= CRASH_THRESHOLD
            },
        }

    def generate_report(self) -> str:
        analysis = self.analyze()
        if "error" in analysis:
            return analysis["error"]

        lines = []
        lines.append("=" * 72)
        lines.append("RESEARCH ORCHESTRATOR - ANALYSIS REPORT")
        lines.append("=" * 72)
        lines.append("")

        summary = analysis["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Experiments:  {summary['total_experiments']}")
        lines.append(f"  Successful:         {summary['successful']}")
        lines.append(f"  Failed:             {summary['failed']}")
        lines.append(f"  Crashed:            {summary['crashed']}")
        lines.append(f"  Best Val BPB:       {summary['best_val_bpb']}")
        lines.append(f"  Best Experiment:    {summary['best_experiment']}")
        lines.append(f"  Current Baseline:   {summary['current_baseline']}")
        lines.append("")

        lines.append("PHASE RESULTS")
        lines.append("-" * 40)
        for phase, stats in sorted(analysis["by_phase"].items()):
            lines.append(
                f"  Phase {phase}: {stats['successful']}/{stats['total']} successful "
                f"(avg delta: {stats['avg_delta']:+.6f})"
            )
        lines.append("")

        if analysis["change_effectiveness"]:
            lines.append("CHANGE EFFECTIVENESS (sorted by avg improvement)")
            lines.append("-" * 40)
            for change, stats in analysis["change_effectiveness"].items():
                lines.append(
                    f"  {change:30s} avg:{stats['avg_improvement']:+.4f} "
                    f"max:{stats['max_improvement']:+.4f} "
                    f"n={stats['num_experiments']} "
                    f"success={stats['success_rate']}%"
                )
            lines.append("")

        if analysis["top_combinations"]:
            lines.append("TOP COMBINATIONS")
            lines.append("-" * 40)
            for combo in analysis["top_combinations"][:5]:
                lines.append(f"  {combo['combination']}  delta:{combo['avg_delta']:+.6f}")
            lines.append("")

        if analysis["baseline_stability"]:
            lines.append("BASELINE STABILITY")
            lines.append("-" * 40)
            bl = analysis["baseline_stability"]
            lines.append(
                f"  Mean:{bl['mean']:.4f}  Std:{bl['std']:.4f}  "
                f"[{bl['min']:.4f}, {bl['max']:.4f}]  N={bl['measurements']}"
            )
            lines.append("")

        if analysis["crash_prone_changes"]:
            lines.append("PERMANENTLY SKIPPED CHANGES")
            lines.append("-" * 40)
            for change, count in analysis["crash_prone_changes"].items():
                lines.append(f"  {change}: {count} crashes")
            lines.append("")

        return "\n".join(lines)


# ─── Prioritization Engine ────────────────────────────────────────────────────


class PrioritizationEngine:
    """
    Bayesian-inspired + Multi-Armed Bandit + Diversity ranking system.
    Tracks change effectiveness and allocates experiments intelligently.
    """

    def __init__(self, db: ResultsDatabase):
        self.db = db
        # Thompson Sampling: Beta distribution parameters per change
        self.successes: Dict[str, float] = defaultdict(lambda: 1.0)
        self.failures: Dict[str, float] = defaultdict(lambda: 1.0)

        # Track all changes ever tried and their average improvements
        self.change_improvements: Dict[str, List[float]] = defaultdict(list)
        # Track which changes have been combined together
        self.co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track phase-level priors
        self.phase_counts: Dict[int, int] = defaultdict(int)

        self._initialize_from_db()

    def _initialize_from_db(self):
        """Bootstrap from existing results."""
        for r in self.db.results:
            if r.status == ExperimentStatus.SUCCESS.value and r.delta:
                for change in r.changes_applied:
                    self.successes[change] += 1.0
                    if r.delta is not None:
                        self.change_improvements[change].append(r.delta)
            elif r.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value):
                for change in r.changes_applied:
                    self.failures[change] += 1.0

            # Track co-occurrence for combination experiments
            if len(r.changes_applied) > 1:
                for i, c1 in enumerate(r.changes_applied):
                    for c2 in r.changes_applied[i + 1:]:
                        pair = tuple(sorted([c1, c2]))
                        self.co_occurrence[pair] += 1

            self.phase_counts[r.phase] += 1

    def thompson_sample(self, change_name: str) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        alpha = self.successes[change_name]
        beta = self.failures[change_name]
        return random.betavariate(alpha, beta)

    def expected_improvement(self, change_name: str) -> float:
        """Expected improvement based on historical performance."""
        improvements = self.change_improvements.get(change_name, [])
        if not improvements:
            return 0.0
        # Weighted average: more recent improvements count more
        weights = [1 + i * 0.1 for i in range(len(improvements))]
        total_weight = sum(weights)
        return sum(w * imp for w, imp in zip(weights, improvements)) / total_weight

    def diversity_score(self, change_name: str, already_tried: Set[str]) -> float:
        """Higher score for changes that are different from recently tried ones."""
        # How many of the recently tried changes is this similar to?
        if not already_tried:
            return 1.0

        # Count co-occurrence with recent changes
        total_similar = sum(
            self.co_occurrence.get(tuple(sorted([change_name, r])), 0)
            for r in already_tried
        )
        # Novelty bonus: penalize frequently paired changes
        novelty = 1.0 / (1.0 + total_similar * 0.3)
        return novelty

    def bandit_allocation(self, change_name: str) -> float:
        """Bandit algorithm: allocate more to promising directions."""
        total = self.successes[change_name] + self.failures[change_name]
        if total <= 1:
            return 0.5  # Unknown, give equal chance
        # UCB1-style: exploit + explore
        exploitation = self.successes[change_name] / total
        exploration_weight = math.sqrt(2 * math.log(max(sum(self.phase_counts.values()), 1)) / total)
        return exploitation + 0.1 * exploration_weight

    def rank_candidates(
        self,
        candidates: List[ExperimentConfig],
        already_tried: Optional[Set[str]] = None,
        top_n: int = 10,
    ) -> List[Tuple[float, ExperimentConfig]]:
        """
        Score and rank candidate experiments.
        Returns (score, config) pairs sorted by score descending.
        """
        already_tried = already_tried or set()
        scored = []

        for candidate in candidates:
            # Skip permanently broken changes
            if self.db.is_permanently_broken(candidate.name):
                continue

            # 1. Thompson Sampling score (probabilistic improvement likelihood)
            ts_score = self.thompson_sample(candidate.name)

            # 2. Expected improvement (historical average)
            ei_score = self.expected_improvement(candidate.name)

            # 3. Bandit allocation (exploitation + exploration)
            bandit_score = self.bandit_allocation(candidate.name)

            # 4. Diversity bonus
            diversity = self.diversity_score(candidate.name, already_tried)

            # 5. Base priority from config
            base_priority = candidate.expected_improvement

            # 6. Risk adjustment
            risk_factor = {"low": 1.1, "medium": 1.0, "high": 0.85}.get(candidate.risk_level, 1.0)

            # Combined score: weighted combination
            combined = (
                0.35 * ts_score          # Thompson Sampling: 35%
                + 0.25 * min(ei_score * 50, 1.0)   # Expected improvement (scaled): 25%
                + 0.20 * bandit_score    # Bandit exploration: 20%
                + 0.10 * diversity       # Diversity bonus: 10%
                + 0.10 * base_priority   # Manual expectation: 10%
            ) * risk_factor

            # Novelty bonus: changes never tried before get a small boost
            history_count = len(self.db.get_experiments_with_change(candidate.name))
            if history_count == 0:
                combined *= 1.15  # 15% novelty bonus

            scored.append((combined, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_n]


# ─── Cross-Pollination Engine ─────────────────────────────────────────────────


class CrossPollinationEngine:
    """
    When independent experiments show gains, try combining them.
    Detects synergistic combinations and proposes new experiments.
    """

    def __init__(self, db: ResultsDatabase):
        self.db = db

    def find_top_performers(
        self, phase: int = 1, top_n: int = 3
    ) -> List[ExperimentResult]:
        """Get top-performing single-factor experiments from a phase."""
        phase_results = [
            r for r in self.db.results
            if r.phase == phase
            and r.status == ExperimentStatus.SUCCESS.value
            and r.delta is not None
            and r.delta > 0
            and len(r.changes_applied) <= 2  # Single or near-single factor
        ]
        phase_results.sort(key=lambda x: x.delta, reverse=True)
        return phase_results[:top_n]

    def generate_combinations(
        self, phase: int = 1, top_n: int = 3
    ) -> List[ExperimentConfig]:
        """Generate combination experiments from top performers."""
        top_performers = self.find_top_performers(phase=phase, top_n=top_n)

        if len(top_performers) < 2:
            logger.warning(
                f"Not enough successful single-factor experiments in Phase {phase} "
                f"(found {len(top_performers)}, need >= 2)"
            )
            return []

        combinations = []
        already_tried = set()
        for r in self.db.results:
            if len(r.changes_applied) > 1:
                already_tried.add(tuple(sorted(r.changes_applied)))

        # Generate all pairs of top performers
        for i, r1 in enumerate(top_performers):
            for r2 in top_performers[i + 1:]:
                combo_changes = sorted(set(r1.changes_applied) | set(r2.changes_applied))
                combo_key = tuple(combo_changes)

                if combo_key in already_tried:
                    continue

                combined_params = {}
                combined_params.update(r1.configuration)
                combined_params.update(r2.configuration)

                combined_delta = (r1.delta or 0) + (r2.delta or 0)

                config = ExperimentConfig(
                    name=f"combo_{'_'.join(combo_changes)}",
                    change_type=ChangeType.HYPERPARAMETER,
                    description=f"Combine: {r1.changes_applied} + {r2.changes_applied}",
                    params=combined_params,
                    expected_improvement=max(0, combined_delta * 0.6),  # Expect some synergy loss
                    risk_level="medium",
                )
                combinations.append(config)

        # Also try triplets if we have enough top performers
        if len(top_performers) >= 3:
            for r1, r2, r3 in [top_performers[:3]]:
                combo_changes = sorted(
                    set(r1.changes_applied) | set(r2.changes_applied) | set(r3.changes_applied)
                )
                combo_key = tuple(combo_changes)
                if combo_key not in already_tried:
                    combined_params = {}
                    combined_params.update(r1.configuration)
                    combined_params.update(r2.configuration)
                    combined_params.update(r3.configuration)
                    combined_delta = (r1.delta or 0) + (r2.delta or 0) + (r3.delta or 0)

                    config = ExperimentConfig(
                        name=f"combo_{'_'.join(combo_changes)}",
                        change_type=ChangeType.HYPERPARAMETER,
                        description=f"Triplet combine: {' + '.join(r.changes_applied for r in [r1, r2, r3])}",
                        params=combined_params,
                        expected_improvement=max(0, combined_delta * 0.4),
                        risk_level="high",
                    )
                    combinations.append(config)

        logger.info(f"Generated {len(combinations)} combination experiments")
        return combinations


# ─── Change Registry ──────────────────────────────────────────────────────────


class ChangeRegistry:
    """
    Registry of all possible changes that can be applied.
    Each entry knows how to generate a patch for train.py.
    """

    def __init__(self):
        self.changes: Dict[str, ExperimentConfig] = {}
        self._initialize()

    def _initialize(self):
        """Define all possible experimental changes."""

        # ── Phase 1: Single-Factor Hyperparameters ──
        self._add_change("lr_1e3", "hyperparameter",
            "Learning rate 1e-3", {"learning_rate": 1e-3},
            expected_improvement=0.01, risk_level="low")
        self._add_change("lr_3e4", "hyperparameter",
            "Learning rate 3e-4", {"learning_rate": 3e-4},
            expected_improvement=0.005, risk_level="low")
        self._add_change("lr_1e4", "hyperparameter",
            "Learning rate 1e-4", {"learning_rate": 1e-4},
            expected_improvement=0.003, risk_level="low")
        self._add_change("lr_5e4_cosine", "hyperparameter",
            "Learning rate 5e-4 with cosine decay", {"learning_rate": 5e-4, "scheduler": "cosine"},
            expected_improvement=0.008, risk_level="medium")
        self._add_change("warmup_1000", "hyperparameter",
            "1000 step warmup", {"warmup_steps": 1000},
            expected_improvement=0.005, risk_level="low")
        self._add_change("warmup_5000", "hyperparameter",
            "5000 step warmup", {"warmup_steps": 5000},
            expected_improvement=0.006, risk_level="medium")

        self._add_change("depth_8", "hyperparameter",
            "8 layers", {"n_layers": 8},
            expected_improvement=0.02, risk_level="low")
        self._add_change("depth_16", "hyperparameter",
            "16 layers", {"n_layers": 16},
            expected_improvement=0.03, risk_level="medium")
        self._add_change("depth_20", "hyperparameter",
            "20 layers", {"n_layers": 20},
            expected_improvement=0.035, risk_level="high")

        self._add_change("dim_512", "hyperparameter",
            "Model dim 512", {"d_model": 512},
            expected_improvement=0.015, risk_level="low")
        self._add_change("dim_1024", "hyperparameter",
            "Model dim 1024", {"d_model": 1024},
            expected_improvement=0.025, risk_level="medium")

        self._add_change("heads_8", "hyperparameter",
            "8 attention heads", {"n_heads": 8},
            expected_improvement=0.005, risk_level="low")
        self._add_change("gqa_4_2", "hyperparameter",
            "Grouped-query attention 4/2", {"n_heads": 4, "n_kv_heads": 2},
            expected_improvement=0.01, risk_level="medium")

        self._add_change("batch_32", "hyperparameter",
            "Batch size 32", {"batch_size": 32},
            expected_improvement=0.005, risk_level="low")
        self._add_change("batch_128", "hyperparameter",
            "Batch size 128", {"batch_size": 128},
            expected_improvement=0.01, risk_level="medium")
        self._add_change("batch_256_grad_accum", "hyperparameter",
            "Effective batch 256 with grad accumulation", {"batch_size": 64, "grad_accum_steps": 4},
            expected_improvement=0.012, risk_level="medium")

        self._add_change("adamw_1e1", "hyperparameter",
            "AdamW eps 1e-1", {"adam_epsilon": 1e-1},
            expected_improvement=0.003, risk_level="low")
        self._add_change("wt_decay_01", "hyperparameter",
            "Weight decay 0.1", {"weight_decay": 0.1},
            expected_improvement=0.005, risk_level="low")
        self._add_change("wt_decay_001", "hyperparameter",
            "Weight decay 0.01", {"weight_decay": 0.01},
            expected_improvement=0.004, risk_level="low")

        self._add_change("dropout_1", "regularization",
            "Dropout 0.1", {"dropout": 0.1},
            expected_improvement=0.005, risk_level="low")
        self._add_change("dropout_0", "regularization",
            "No dropout", {"dropout": 0.0},
            expected_improvement=0.003, risk_level="low")

        self._add_change("rope_theta_5e5", "hyperparameter",
            "RoPE theta 5e5", {"rope_theta": 5e5},
            expected_improvement=0.005, risk_level="low")
        self._add_change("rope_theta_1e7", "hyperparameter",
            "RoPE theta 1e7", {"rope_theta": 1e7},
            expected_improvement=0.01, risk_level="medium")

        self._add_change("seq_4096", "hyperparameter",
            "Sequence length 4096", {"max_seq_len": 4096},
            expected_improvement=0.02, risk_level="medium")
        self._add_change("seq_8192", "hyperparameter",
            "Sequence length 8192", {"max_seq_len": 8192},
            expected_improvement=0.03, risk_level="high")

        # ── Phase 4: Architectural ──
        self._add_change("swiglu_ffn", "architectural",
            "SwiGLU feed-forward network", {"use_swiglu": True},
            expected_improvement=0.02, risk_level="high")
        self._add_change("no_bias", "architectural",
            "Remove bias from linear layers", {"use_bias": False},
            expected_improvement=0.005, risk_level="medium")
        self._add_change("pre_ln", "architectural",
            "Pre-LayerNorm architecture", {"pre_norm": True},
            expected_improvement=0.01, risk_level="medium")
        self._add_change("alibi", "architectural",
            "ALiBi positional embeddings", {"use_alibi": True},
            expected_improvement=0.01, risk_level="high")
        self._add_change("qk_norm", "architectural",
            "QK normalization in attention", {"qk_norm": True},
            expected_improvement=0.008, risk_level="medium")

        # ── Phase 1: Data pipeline ──
        self._add_change("packed_data", "data_pipeline",
            "Packed/token-packed dataset", {"use_packing": True},
            expected_improvement=0.015, risk_level="medium")
        self._add_change("fsl_shard", "data_pipeline",
            "FineWeb-edu-sharded dataset", {"dataset": "fineweb-edu-sharded"},
            expected_improvement=0.01, risk_level="medium")
        self._add_change("dclm_mix", "data_pipeline",
            "DCLM + fineWeb mix", {"dataset": "mixed_dclm_fineweb", "dclm_ratio": 0.3},
            expected_improvement=0.012, risk_level="high")

        # ── Phase 4: MoE ──
        self._add_change("moe_4_2", "architectural",
            "MoE 4 experts, 2 active", {"num_experts": 4, "num_active_experts": 2},
            expected_improvement=0.025, risk_level="high")
        self._add_change("moe_8_2", "architectural",
            "MoE 8 experts, 2 active", {"num_experts": 8, "num_active_experts": 2},
            expected_improvement=0.03, risk_level="high")

        # ── Training loop tweaks ──
        self._add_change("grad_clip_05", "training_loop",
            "Gradient clipping 0.5", {"max_grad_norm": 0.5},
            expected_improvement=0.005, risk_level="low")
        self._add_change("grad_clip_20", "training_loop",
            "Gradient clipping 2.0", {"max_grad_norm": 2.0},
            expected_improvement=0.003, risk_level="low")
        self._add_change("adam_beta2_95", "training_loop",
            "Adam beta2 0.95", {"adam_beta2": 0.95},
            expected_improvement=0.005, risk_level="medium")
        self._add_change("adam_beta2_999", "training_loop",
            "Adam beta2 0.999", {"adam_beta2": 0.999},
            expected_improvement=0.004, risk_level="low")

        logger.info(f"Registered {len(self.changes)} possible changes")

    def _add_change(self, name: str, change_type: str, description: str,
                    params: dict, expected_improvement: float, risk_level: str):
        self.changes[name] = ExperimentConfig(
            name=name,
            change_type=ChangeType(change_type),
            description=description,
            params=params,
            expected_improvement=expected_improvement,
            risk_level=risk_level,
        )

    def get_by_type(self, change_type: ChangeType) -> List[ExperimentConfig]:
        return [c for c in self.changes.values() if c.change_type == change_type]

    def get_all(self) -> List[ExperimentConfig]:
        return list(self.changes.values())

    def get_by_name(self, name: str) -> Optional[ExperimentConfig]:
        return self.changes.get(name)


# ─── Git Manager ──────────────────────────────────────────────────────────────


class GitManager:
    """Handles git branch creation, commits, and cleanup."""

    def __init__(self, base_branch: str = "main", results_dir: Optional[Path] = None):
        self.base_branch = base_branch
        self.results_dir = results_dir or RESULTS_DIR
        self.original_branch: Optional[str] = None

    def ensure_on_base(self):
        """Make sure we are on the base branch."""
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True
        )
        current = result.stdout.strip()
        result_clean = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
        )
        has_changes = bool(result_clean.stdout.strip())
        if current != self.base_branch:
            logger.info(f"Switching from {current} to {self.base_branch}")
            if has_changes:
                subprocess.run(["git", "stash"], capture_output=True)
            subprocess.run(["git", "checkout", self.base_branch], capture_output=True)
            if has_changes:
                subprocess.run(["git", "stash", "pop"], capture_output=True)

    def create_experiment_branch(self, exp_id: str, phase: int) -> str:
        """Create a new branch for an experiment."""
        branch_name = f"{self.base_branch}/{exp_id}"
        self.ensure_on_base()
        subprocess.run(["git", "checkout", "-b", branch_name], capture_output=True)
        return branch_name

    def get_head_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip()

    def get_diff_hash(self) -> str:
        """Hash of current unstaged changes."""
        result = subprocess.run(
            ["git", "diff", "--", "train.py"],
            capture_output=True, text=True
        )
        return hashlib.md5(result.stdout.encode()).hexdigest()[:8]

    def commit_changes(self, message: str) -> Optional[str]:
        """Stage and commit changes."""
        subprocess.run(["git", "add", "-A"], capture_output=True)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            return None
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return self.get_head_commit()
        return None

    def return_to_base(self):
        """Return to the base branch."""
        self.ensure_on_base()

    def cleanup_branch(self, branch_name: str):
        """Delete an experiment branch."""
        subprocess.run(["git", "checkout", self.base_branch], capture_output=True)
        subprocess.run(["git", "branch", "-D", branch_name], capture_output=True)


# ─── Change Applier ───────────────────────────────────────────────────────────


class ChangeApplier:
    """
    Applies experimental changes to train.py.
    Generates Python patches based on ExperimentConfig params.
    """

    def __init__(self, train_file: str = "train.py"):
        self.train_file = train_file
        self._applied_changes: Dict[str, dict] = {}

    def apply_changes(self, config: ExperimentConfig) -> str:
        """
        Apply changes to the training script.
        Returns a description of the diff.
        """
        changes_desc = []
        for key, value in config.params.items():
            desc = self._apply_param(key, value)
            changes_desc.append(desc)
        return "; ".join(changes_desc)

    def _apply_param(self, key: str, value: Any) -> str:
        """Apply a single parameter change to train.py via regex-based patching."""
        train_path = Path(self.train_file)
        if not train_path.exists():
            # Try to find train.py in parent directories
            for parent in train_path.resolve().parents:
                candidate = parent / self.train_file
                if candidate.exists():
                    train_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Cannot find {self.train_file}. "
                    "Make sure train.py exists in the project root."
                )

        content = train_path.read_text()

        if key == "learning_rate":
            content = re.sub(
                r'(learning_rate\s*[=:]\s*)([\d.e\-]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"learning_rate={value}"

        elif key == "scheduler":
            # Add or modify scheduler
            if re.search(r'scheduler\s*[=:]', content):
                content = re.sub(
                    r'(scheduler\s*[=:]\s*)(\w+)',
                    rf'\g<1>"{value}"',
                    content,
                )
            else:
                # Add at end of hyperparameter block
                content = content.replace(
                    "}", f', "scheduler": "{value}"\n}}'
                )
            return f"scheduler={value}"

        elif key == "warmup_steps":
            content = re.sub(
                r'(warmup_steps\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"warmup_steps={value}"

        elif key == "n_layers":
            content = re.sub(
                r'(n_layers\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"n_layers={value}"

        elif key == "d_model":
            content = re.sub(
                r'(d_model\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"d_model={value}"

        elif key == "n_heads":
            content = re.sub(
                r'(n_heads\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"n_heads={value}"

        elif key == "n_kv_heads":
            content = re.sub(
                r'(n_kv_heads\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"n_kv_heads={value}"

        elif key == "batch_size":
            content = re.sub(
                r'(batch_size\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"batch_size={value}"

        elif key == "grad_accum_steps":
            content = re.sub(
                r'(grad_accum_steps\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"grad_accum_steps={value}"

        elif key == "adam_epsilon":
            content = re.sub(
                r'(adam_epsilon\s*[=:]\s*)([\d.e\-]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"adam_epsilon={value}"

        elif key == "weight_decay":
            content = re.sub(
                r'(weight_decay\s*[=:]\s*)([\d.e\-]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"weight_decay={value}"

        elif key == "dropout":
            content = re.sub(
                r'(dropout\s*[=:]\s*)([\d.]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"dropout={value}"

        elif key == "rope_theta":
            content = re.sub(
                r'(rope_theta\s*[=:]\s*)([\d.e\-]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"rope_theta={value}"

        elif key == "max_seq_len":
            content = re.sub(
                r'(max_seq_len\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"max_seq_len={value}"

        elif key == "max_grad_norm":
            content = re.sub(
                r'(max_grad_norm\s*[=:]\s*)([\d.]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"max_grad_norm={value}"

        elif key == "adam_beta2":
            content = re.sub(
                r'(adam_beta2\s*[=:]\s*)([\d.e\-]+)',
                rf'\g<1>{value}',
                content,
            )
            return f"adam_beta2={value}"

        elif key in ("use_swiglu", "use_bias", "pre_norm", "use_alibi", "qk_norm"):
            content = re.sub(
                rf'({key}\s*[=:]\s*)(True|False)',
                rf'\g<1>{str(value)}',
                content,
            )
            return f"{key}={value}"

        elif key == "num_experts":
            content = re.sub(
                r'(num_experts\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"num_experts={value}"

        elif key == "num_active_experts":
            content = re.sub(
                r'(num_active_experts\s*[=:]\s*)(\d+)',
                rf'\g<1>{value}',
                content,
            )
            return f"num_active_experts={value}"

        elif key == "dataset":
            content = re.sub(
                r'(dataset\s*[=:]\s*)("|\')(\w+-?\w*?)("|\')',
                rf'\g<1>{value}\g<4>',
                content,
            )
            return f"dataset={value}"

        elif key == "use_packing":
            content = re.sub(
                rf'(use_packing\s*[=:]\s*)(True|False)',
                rf'\g<1>{str(value)}',
                content,
            )
            return f"use_packing={value}"

        else:
            logger.warning(f"Unknown parameter key: {key}")
            return f"{key}={value} (unknown)"

        train_path.write_text(content)
        return f"applied {key}={value}"

    def reset(self):
        """Reset train.py to original state via git checkout."""
        subprocess.run(
            ["git", "checkout", "--", self.train_file],
            capture_output=True, text=True
        )


# ─── Experiment Runner ────────────────────────────────────────────────────────


class ExperimentRunner:
    """
    Executes experiments: applies changes, trains, parses results.
    """

    def __init__(
        self,
        train_command: str = "python train.py --max-steps 100 --eval-every 50",
        train_timeout: int = DEFAULT_TRAIN_TIMEOUT,
        results_dir: Optional[Path] = None,
    ):
        self.train_command = train_command
        self.train_timeout = train_timeout
        self.results_dir = results_dir or RESULTS_DIR

    def run(self, exp_id: str, config: ExperimentConfig, phase: int,
            baseline_bpb: float, git_mgr: GitManager, applier: ChangeApplier) -> ExperimentResult:
        """
        Execute one experiment: branch -> apply -> train -> parse -> record.
        """
        start_time = time.time()
        notes_parts = []

        try:
            # Create branch
            branch_name = git_mgr.create_experiment_branch(exp_id, phase)
            logger.info(f"[{exp_id}] Created branch: {branch_name}")

            # Apply changes
            diff_desc = applier.apply_changes(config)
            logger.info(f"[{exp_id}] Applied changes: {diff_desc}")

            # Commit changes
            commit_hash = git_mgr.commit_changes(f"experiment {exp_id}: {config.description}")
            if not commit_hash:
                notes_parts.append("no actual diff produced")

            diff_hash = git_mgr.get_diff_hash() if commit_hash else None

            # Run training
            logger.info(f"[{exp_id}] Running training...")
            env = os.environ.copy()
            env["EXPERIMENT_ID"] = exp_id

            result = subprocess.run(
                self.train_command.split(),
                capture_output=True, text=True, timeout=self.train_timeout,
                env=env,
            )
            logger.info(f"[{exp_id}] Training complete: exit_code={result.returncode}")

            # Parse validation BPB from training output
            val_bpb = self._parse_val_bpb(result.stdout + "\n" + result.stderr)

            duration = time.time() - start_time

            if val_bpb is not None:
                delta = round(baseline_bpb - val_bpb, 6)
                status = ExperimentStatus.SUCCESS.value
                logger.info(f"[{exp_id}] val_bpb={val_bpb:.4f} delta={delta:+.6f}")
            else:
                # Failed to parse, check exit code
                if result.returncode != 0:
                    status = ExperimentStatus.CRASHED.value
                    val_bpb = None
                    delta = None
                    notes_parts.append(f"exit_code={result.returncode}")
                else:
                    status = ExperimentStatus.FAILED.value
                    val_bpb = None
                    delta = None
                    notes_parts.append("could not parse val_bpb")

                logger.warning(
                    f"[{exp_id}] {status}: {result.stderr[:200] if result.stderr else result.stdout[:200]}"
                )

            return ExperimentResult(
                experiment_id=exp_id,
                experiment_name=config.name,
                configuration=config.params,
                val_bpb=val_bpb,
                status=status,
                description=config.description,
                phase=phase,
                git_commit=commit_hash,
                git_diff_hash=diff_hash,
                baseline_bpb=baseline_bpb,
                delta=delta,
                duration_seconds=duration,
                changes_applied=[config.name],
                notes="; ".join(notes_parts),
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"[{exp_id}] Training timed out after {self.train_timeout}s")
            return ExperimentResult(
                experiment_id=exp_id,
                experiment_name=config.name,
                configuration=config.params,
                val_bpb=None,
                status=ExperimentStatus.FAILED.value,
                description=config.description,
                phase=phase,
                baseline_bpb=baseline_bpb,
                delta=None,
                duration_seconds=duration,
                changes_applied=[config.name],
                notes=f"timeout after {self.train_timeout}s",
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{exp_id}] Unexpected error: {e}")
            return ExperimentResult(
                experiment_id=exp_id,
                experiment_name=config.name,
                configuration=config.params,
                val_bpb=None,
                status=ExperimentStatus.CRASHED.value,
                description=config.description,
                phase=phase,
                baseline_bpb=baseline_bpb,
                delta=None,
                duration_seconds=duration,
                changes_applied=[config.name],
                notes=str(e),
            )

        finally:
            # Always return to base branch and reset file
            applier.reset()
            git_mgr.return_to_base()
            logger.info(f"[{exp_id}] Cleaned up, returned to {git_mgr.base_branch}")

    def _parse_val_bpb(self, output: str) -> Optional[float]:
        """
        Extract validation BPB from training logs.
        Tries multiple patterns for robustness.
        """
        # Pattern 1: val_bpb: 1.2345
        m = re.search(r'val_bpb\s*[=:]?\s*([0-9]+\.[0-9]+)', output)
        if m:
            return float(m.group(1))

        # Pattern 2: val_loss: 1.2345
        m = re.search(r'val_loss\s*[=:]?\s*([0-9]+\.[0-9]+)', output)
        if m:
            return float(m.group(1))

        # Pattern 3: validation.*bpb.*= 1.2345
        m = re.search(r'validation.*bpb.*?(\d+\.\d+)', output, re.IGNORECASE)
        if m:
            return float(m.group(1))

        # Pattern 4: step.*eval.*(\d+\.\d+)
        m = re.search(r'eval.*?(\d+\.\d+)', output, re.IGNORECASE)
        if m:
            return float(m.group(1))

        # Pattern 5: any number that looks like a loss (0.5 - 3.0)
        matches = re.findall(r'(\d+\.\d{3,})', output)
        for match in matches:
            val = float(match)
            if 0.1 <= val <= 5.0:
                return val

        return None

    def run_baseline(self, git_mgr: GitManager) -> ExperimentResult:
        """Run baseline experiment (no changes)."""
        exp_id = f"baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        baseline_result = self.run(
            exp_id=exp_id,
            config=ExperimentConfig(
                name="baseline",
                change_type=ChangeType.HYPERPARAMETER,
                description="Baseline - no changes",
                params={},
            ),
            phase=0,
            baseline_bpb=0.0,
            git_mgr=git_mgr,
            applier=ChangeApplier(),
        )
        baseline_result.status = ExperimentStatus.BASELINE.value
        baseline_result.baseline_bpb = None
        baseline_result.delta = None
        return baseline_result


# ─── Phase Executors ─────────────────────────────────────────────────────────


class PhaseExecutor:
    """
    Executes each research phase with its specific strategy.
    """

    def __init__(
        self,
        db: ResultsDatabase,
        registry: ChangeRegistry,
        runner: ExperimentRunner,
        git_mgr: GitManager,
        applier: ChangeApplier,
        prioritizer: PrioritizationEngine,
        cross_pollinator: CrossPollinationEngine,
        max_experiments: int = 100,
        train_timeout: int = DEFAULT_TRAIN_TIMEOUT,
    ):
        self.db = db
        self.registry = registry
        self.runner = runner
        self.git_mgr = git_mgr
        self.applier = applier
        self.prioritizer = prioritizer
        self.cross_pollinator = cross_pollinator
        self.max_experiments = max_experiments
        self.train_timeout = train_timeout
        self.experiment_counter = 0
        self.best_config: Optional[ExperimentConfig] = None
        self.best_bpb: Optional[float] = None

    def generate_experiment_id(self, phase: int) -> str:
        self.experiment_counter += 1
        return f"exp_p{phase}_{self.experiment_counter:04d}"

    def run_phase_1_single_factor(self) -> List[ExperimentResult]:
        """
        Phase 1: Test each hyperparameter independently.
        Cycles through all registered changes, prioritized by the engine.
        """
        phase = 1
        logger.info("=" * 60)
        logger.info("PHASE 1: Single-Factor Experiments")
        logger.info("=" * 60)

        # Get all available changes, excluding already-permanently-broken ones
        all_changes = [
            c for c in self.registry.get_all()
            if not self.db.is_permanently_broken(c.name)
        ]

        already_tried = self.db.get_all_changes()

        # Rank candidates
        ranked = self.prioritizer.rank_candidates(all_changes, already_tried, top_n=len(all_changes))

        results = []
        experiments_run = 0
        baseline_bpb = self.db.get_current_baseline() or DEFAULT_BASELINE_BPB

        # Check if we need baseline
        if not self.db.baseline_history:
            logger.info("Running initial baseline...")
            baseline_result = self.runner.run_baseline(self.git_mgr)
            self.db.add_result(baseline_result)
            if baseline_result.val_bpb:
                baseline_bpb = baseline_result.val_bpb
                logger.info(f"Baseline BPB: {baseline_bpb:.4f}")

        for _, config in ranked:
            if experiments_run >= self.max_experiments:
                logger.info(f"Reached max experiments ({self.max_experiments})")
                break

            # Skip if permanently broken
            if self.db.is_permanently_broken(config.name):
                logger.info(f"[Phase 1] Skipping {config.name} (permanently broken)")
                continue

            exp_id = self.generate_experiment_id(phase)
            logger.info(
                f"[Phase 1] {exp_id}: {config.name} - {config.description}"
            )

            result = self.runner.run(
                exp_id=exp_id,
                config=config,
                phase=phase,
                baseline_bpb=baseline_bpb,
                git_mgr=self.git_mgr,
                applier=self.applier,
            )

            self.db.add_result(result)
            results.append(result)
            experiments_run += 1

            # Update crash tracking
            if result.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value):
                self.db.update_crash_count(config.name)
                if self.db.is_permanently_broken(config.name):
                    logger.warning(
                        f"[Phase 1] {config.name} now permanently skipped "
                        f"({self.db.get_crash_count(config.name)} crashes)"
                    )

            # Re-check baseline periodically
            if experiments_run % BASELINE_RECHECK_INTERVAL == 0:
                logger.info("Re-checking baseline for drift detection...")
                baseline_result = self.runner.run_baseline(self.git_mgr)
                self.db.add_result(baseline_result)
                if baseline_result.val_bpb:
                    old_baseline = baseline_bpb
                    baseline_bpb = baseline_result.val_bpb
                    drift = abs(baseline_bpb - old_baseline)
                    if drift > 0.01:
                        logger.warning(
                            f"Baseline drift detected! {old_baseline:.4f} -> {baseline_bpb:.4f}"
                        )

            # Update prioritizer with new result
            self.prioritizer._initialize_from_db()

            # Track best result
            if result.val_bpb and (self.best_bpb is None or result.val_bpb < self.best_bpb):
                self.best_bpb = result.val_bpb
                self.best_config = config
                logger.info(
                    f"[Phase 1] New best! val_bpb={result.val_bpb:.4f} "
                    f"delta={result.delta:+.6f} ({config.name})"
                )

        logger.info(
            f"[Phase 1] Complete: {experiments_run} experiments, "
            f"best val_bpb={self.best_bpb}"
        )
        return results

    def run_phase_2_combinations(self, from_results_path: Optional[Path] = None) -> List[ExperimentResult]:
        """
        Phase 2: Combine top 3 improvements from Phase 1.
        """
        phase = 2
        logger.info("=" * 60)
        logger.info("PHASE 2: Combination Experiments")
        logger.info("=" * 60)

        if from_results_path:
            self.db.db_path = from_results_path
            self.db._load()
            logger.info(f"Loaded results from {from_results_path}")
            self.prioritizer = PrioritizationEngine(self.db)
            self.cross_pollinator = CrossPollinationEngine(self.db)

        # Generate combinations from Phase 1 results
        combos = self.cross_pollinator.generate_combinations(phase=1, top_n=3)

        if not combos:
            logger.warning("No combinations to try. Running Phase 1 first may help.")
            # Try to run Phase 1 if no successful results exist
            phase1_results = self.db.get_by_phase(1)
            if any(r.status == ExperimentStatus.SUCCESS.value for r in phase1_results):
                logger.info("Phase 1 results exist but no good combinations found.")
            else:
                logger.info("Auto-running Phase 1...")
                phase1_results = self.run_phase_1_single_factor()
                combos = self.cross_pollinator.generate_combinations(phase=1, top_n=3)
                if not combos:
                    return []

        results = []
        experiments_run = 0
        baseline_bpb = self.db.get_current_baseline() or DEFAULT_BASELINE_BPB

        for config in combos:
            if experiments_run >= self.max_experiments:
                break

            exp_id = self.generate_experiment_id(phase)
            logger.info(f"[Phase 2] {exp_id}: {config.name} - {config.description}")

            result = self.runner.run(
                exp_id=exp_id,
                config=config,
                phase=phase,
                baseline_bpb=baseline_bpb,
                git_mgr=self.git_mgr,
                applier=self.applier,
            )

            self.db.add_result(result)
            results.append(result)
            experiments_run += 1

            if result.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value):
                for change in config.params:
                    self.db.update_crash_count(config.name)

            if result.val_bpb and (self.best_bpb is None or result.val_bpb < self.best_bpb):
                self.best_bpb = result.val_bpb
                self.best_config = config
                logger.info(
                    f"[Phase 2] New best! val_bpb={result.val_bpb:.4f} "
                    f"delta={result.delta:+.6f}"
                )

            # Re-check baseline periodically
            if experiments_run % BASELINE_RECHECK_INTERVAL == 0:
                baseline_result = self.runner.run_baseline(self.git_mgr)
                self.db.add_result(baseline_result)
                if baseline_result.val_bpb:
                    baseline_bpb = baseline_result.val_bpb

        # Second wave: use prioritizer to rank any remaining combinations
        remaining_combos = self.cross_pollinator.generate_combinations(phase=1, top_n=5)
        ranked = self.prioritizer.rank_candidates(
            remaining_combos,
            top_n=self.max_experiments - experiments_run,
        )

        for _, config in ranked:
            if experiments_run >= self.max_experiments:
                break

            exp_id = self.generate_experiment_id(phase)
            logger.info(f"[Phase 2] {exp_id}: {config.name}")

            result = self.runner.run(
                exp_id=exp_id,
                config=config,
                phase=phase,
                baseline_bpb=baseline_bpb,
                git_mgr=self.git_mgr,
                applier=self.applier,
            )

            self.db.add_result(result)
            results.append(result)
            experiments_run += 1

            if result.val_bpb and (self.best_bpb is None or result.val_bpb < self.best_bpb):
                self.best_bpb = result.val_bpb
                self.best_config = config

        logger.info(f"[Phase 2] Complete: {experiments_run} experiments")
        return results

    def run_phase_3_fine_tuning(self) -> List[ExperimentResult]:
        """
        Phase 3: Micro-adjust around the best configuration found so far.
        """
        phase = 3
        logger.info("=" * 60)
        logger.info("PHASE 3: Fine-Tuning Experiments")
        logger.info("=" * 60)

        # Find the best experiment so far
        best_exp = None
        for r in self.db.results:
            if r.status == ExperimentStatus.SUCCESS.value and r.delta and r.delta > 0:
                if best_exp is None or (r.val_bpb is not None and best_exp.val_bpb is not None and r.val_bpb < best_exp.val_bpb):
                    best_exp = r

        if best_exp is None:
            logger.warning("No successful experiments yet. Run Phase 1 first.")
            return self.run_phase_1_single_factor()

        # Generate micro-adjustments around best config
        micro_adjustments = []
        best_params = best_exp.configuration.copy()

        # Learning rate micro-adjustments (+/- 10%, 20%, 50%)
        if "learning_rate" in best_params:
            base_lr = best_params["learning_rate"]
            for factor in [0.5, 0.8, 1.2, 2.0]:
                new_lr = base_lr * factor
                micro_adjustments.append(ExperimentConfig(
                    name=f"lr_micro_{new_lr:.2e}",
                    change_type=ChangeType.HYPERPARAMETER,
                    description=f"Fine-tune LR: {new_lr:.2e} (was {base_lr:.2e})",
                    params={**best_params, "learning_rate": new_lr},
                    expected_improvement=0.005,
                    risk_level="low",
                ))

        # Weight decay micro-adjustments
        if "weight_decay" in best_params:
            base_wd = best_params["weight_decay"]
            for factor in [0.5, 2.0, 5.0]:
                new_wd = base_wd * factor
                micro_adjustments.append(ExperimentConfig(
                    name=f"wd_micro_{new_wd:.4f}",
                    change_type=ChangeType.REGULARIZATION,
                    description=f"Fine-tune weight decay: {new_wd:.4f}",
                    params={**best_params, "weight_decay": new_wd},
                    expected_improvement=0.003,
                    risk_level="low",
                ))

        # Dropout micro-adjustments
        for dropout_val in [0.0, 0.05, 0.15, 0.2]:
            micro_adjustments.append(ExperimentConfig(
                name=f"dropout_micro_{dropout_val}",
                change_type=ChangeType.REGULARIZATION,
                description=f"Fine-tune dropout: {dropout_val}",
                params={**best_params, "dropout": dropout_val},
                expected_improvement=0.003,
                risk_level="low",
            ))

        # Batch size micro-adjustments
        if "batch_size" in best_params:
            base_bs = best_params["batch_size"]
            for mult in [0.5, 0.75, 1.5, 2.0]:
                new_bs = int(base_bs * mult)
                micro_adjustments.append(ExperimentConfig(
                    name=f"bs_micro_{new_bs}",
                    change_type=ChangeType.HYPERPARAMETER,
                    description=f"Fine-tune batch: {new_bs} (was {base_bs})",
                    params={**best_params, "batch_size": new_bs},
                    expected_improvement=0.005,
                    risk_level="low",
                ))

        logger.info(f"[Phase 3] Generated {len(micro_adjustments)} fine-tuning configs "
                     f"around best experiment {best_exp.experiment_id}")

        results = []
        experiments_run = 0
        baseline_bpb = self.db.get_current_baseline() or DEFAULT_BASELINE_BPB

        ranked = self.prioritizer.rank_candidates(micro_adjustments, top_n=self.max_experiments)

        for _, config in ranked:
            if experiments_run >= self.max_experiments:
                break

            exp_id = self.generate_experiment_id(phase)
            logger.info(f"[Phase 3] {exp_id}: {config.name}")

            result = self.runner.run(
                exp_id=exp_id,
                config=config,
                phase=phase,
                baseline_bpb=baseline_bpb,
                git_mgr=self.git_mgr,
                applier=self.applier,
            )

            self.db.add_result(result)
            results.append(result)
            experiments_run += 1

            if result.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value):
                self.db.update_crash_count(config.name)

            if result.val_bpb and (self.best_bpb is None or result.val_bpb < self.best_bpb):
                self.best_bpb = result.val_bpb
                self.best_config = config
                logger.info(
                    f"[Phase 3] New best! val_bpb={result.val_bpb:.4f}"
                )

            # Re-check baseline periodically
            if experiments_run % BASELINE_RECHECK_INTERVAL == 0:
                baseline_result = self.runner.run_baseline(self.git_mgr)
                self.db.add_result(baseline_result)
                if baseline_result.val_bpb:
                    baseline_bpb = baseline_result.val_bpb

        logger.info(f"[Phase 3] Complete: {experiments_run} experiments")
        return results

    def run_phase_4_radical(self) -> List[ExperimentResult]:
        """
        Phase 4: Radical architectural changes (SwiGLU, MoE, ALiBi, etc.)
        """
        phase = 4
        logger.info("=" * 60)
        logger.info("PHASE 4: Radical Architectural Experiments")
        logger.info("=" * 60)

        # Start with best config from previous phases, apply architectural changes
        best_params = None
        for r in sorted(
            self.db.results,
            key=lambda x: x.val_bpb if x.val_bpb is not None else float("inf"),
        ):
            if r.status == ExperimentStatus.SUCCESS.value and r.val_bpb is not None:
                best_params = r.configuration.copy()
                break

        if best_params is None:
            best_params = {}

        # Get architectural changes
        arch_changes = self.registry.get_by_type(ChangeType.ARCHITECTURAL)

        # Combine with best params
        radical_configs = []
        for arch in arch_changes:
            combined_params = {**best_params, **arch.params}
            radical_configs.append(ExperimentConfig(
                name=f"radical_{arch.name}",
                change_type=ChangeType.ARCHITECTURAL,
                description=f"RADICAL: {arch.description} on best config",
                params=combined_params,
                expected_improvement=arch.expected_improvement,
                risk_level="high",
            ))

        logger.info(f"[Phase 4] Testing {len(radical_configs)} architectural changes")

        results = []
        experiments_run = 0
        baseline_bpb = self.db.get_current_baseline() or DEFAULT_BASELINE_BPB

        # Also add radical combinations (arch + top hyperparameters)
        top_hyper = [
            c for c in self.registry.get_all()
            if c.change_type == ChangeType.HYPERPARAMETER
            and c.expected_improvement > 0.01
        ]
        for arch in arch_changes[:2]:  # Top 2 arch changes
            for hyper in top_hyper[:3]:  # Top 3 hyper changes
                combined_params = {**best_params, **arch.params, **hyper.params}
                radical_configs.append(ExperimentConfig(
                    name=f"radical_{arch.name}_{hyper.name}",
                    change_type=ChangeType.ARCHITECTURAL,
                    description=f"RADICAL combo: {arch.name} + {hyper.name}",
                    params=combined_params,
                    expected_improvement=arch.expected_improvement + hyper.expected_improvement * 0.5,
                    risk_level="high",
                ))

        ranked = self.prioritizer.rank_candidates(radical_configs, top_n=self.max_experiments)

        for _, config in ranked:
            if experiments_run >= self.max_experiments:
                break

            # Skip broken architectural changes
            if self.db.is_permanently_broken(config.name):
                continue

            exp_id = self.generate_experiment_id(phase)
            logger.info(f"[Phase 4] {exp_id}: {config.name}")

            # Give radical experiments more time
            original_timeout = self.runner.train_timeout
            self.runner.train_timeout = max(self.runner.train_timeout, 600)  # at least 10 min

            result = self.runner.run(
                exp_id=exp_id,
                config=config,
                phase=phase,
                baseline_bpb=baseline_bpb,
                git_mgr=self.git_mgr,
                applier=self.applier,
            )

            self.runner.train_timeout = original_timeout

            self.db.add_result(result)
            results.append(result)
            experiments_run += 1

            if result.status in (ExperimentStatus.FAILED.value, ExperimentStatus.CRASHED.value):
                self.db.update_crash_count(config.name)
                if self.db.is_permanently_broken(config.name):
                    logger.warning(f"[Phase 4] {config.name} permanently skipped")

            if result.val_bpb and (self.best_bpb is None or result.val_bpb < self.best_bpb):
                self.best_bpb = result.val_bpb
                self.best_config = config
                logger.info(f"[Phase 4] New best! val_bpb={result.val_bpb:.4f}")

        logger.info(f"[Phase 4] Complete: {experiments_run} experiments")
        return results

    def run_all_phases(self, start_phase: int = 1):
        """Run all phases sequentially starting from the given phase."""
        all_results = []

        if start_phase <= 1:
            results = self.run_phase_1_single_factor()
            all_results.extend(results)

        if start_phase <= 2:
            results = self.run_phase_2_combinations()
            all_results.extend(results)

        if start_phase <= 3:
            results = self.run_phase_3_fine_tuning()
            all_results.extend(results)

        if start_phase <= 4:
            results = self.run_phase_4_radical()
            all_results.extend(results)

        return all_results


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Systematic Research Orchestrator v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 (single-factor experiments on autoresearch/v2 branch)
  python research_orchestrator.py --phase 1 --branch autoresearch/v2 --max-experiments 20
  
  # Run Phase 2 (combinations) from existing results
  python research_orchestrator.py --phase 2 --from-results results/research_results.json
  
  # Analyze results
  python research_orchestrator.py --analyze --results-dir results/
  
  # Fine-tuning phase with custom training command
  python research_orchestrator.py --phase 3 --train-cmd "python train.py --max-steps 200"
  
  # Run all phases
  python research_orchestrator.py --phase 0 --max-experiments 100
  
  # Custom timeout per experiment
  python research_orchestrator.py --phase 1 --timeout 600
        """,
    )

    # Phase selection
    parser.add_argument(
        "--phase", type=int, choices=[0, 1, 2, 3, 4], default=1,
        help="Phase to run: 0=all, 1=single-factor, 2=combinations, "
             "3=fine-tuning, 4=radical. Default: 1",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=50,
        help="Maximum number of experiments per phase. Default: 50",
    )

    # Git/Branch config
    parser.add_argument(
        "--branch", type=str, default="main",
        help="Base git branch name. Default: main",
    )

    # Training configuration
    parser.add_argument(
        "--train-cmd", type=str, default=None,
        help="Training command to run. Default: python train.py --max-steps 100 --eval-every 50",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TRAIN_TIMEOUT,
        help="Timeout per experiment in seconds. Default: 300",
    )

    # Results loading
    parser.add_argument(
        "--from-results", type=str, default=None,
        help="Path to existing results JSON file to load",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing results files. Default: ./results/",
    )

    # Analysis mode
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze existing results and print report",
    )

    # Export format
    parser.add_argument(
        "--export-tsv", action="store_true",
        help="Also export results as TSV",
    )

    # Database path
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to results database JSON file",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Override results dir if specified ──
    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Analysis mode ──
    if args.analyze:
        db = ResultsDatabase(
            db_path=Path(args.db_path) if args.db_path else None
        )
        report = db.generate_report()
        print(report)

        if args.export_tsv:
            db.export_tsv()

        return

    # ── Initialize components ──
    db = ResultsDatabase(
        db_path=Path(args.db_path) if args.db_path else None
    )

    # Load external results if specified
    if args.from_results:
        from_path = Path(args.from_results)
        if from_path.exists():
            logger.info(f"Loading external results from {from_path}")
            external = ResultsDatabase(db_path=from_path)
            db.results.extend(external.results)
            db.baseline_history.extend(external.baseline_history)
            for k, v in external.crash_log.items():
                db.crash_log[k] = db.crash_log.get(k, 0) + v
            db._save()
        else:
            logger.error(f"Results file not found: {from_path}")
            sys.exit(1)

    registry = ChangeRegistry()

    git_mgr = GitManager(base_branch=args.branch)
    applier = ChangeApplier()

    train_cmd = args.train_cmd or "python train.py --max-steps 100 --eval-every 50"
    runner = ExperimentRunner(
        train_command=train_cmd,
        train_timeout=args.timeout,
    )

    prioritizer = PrioritizationEngine(db)
    cross_pollinator = CrossPollinationEngine(db)

    executor = PhaseExecutor(
        db=db,
        registry=registry,
        runner=runner,
        git_mgr=git_mgr,
        applier=applier,
        prioritizer=prioritizer,
        cross_pollinator=cross_pollinator,
        max_experiments=args.max_experiments,
        train_timeout=args.timeout,
    )

    # ── Run phases ──
    logger.info(f"Starting Research Orchestrator v2 (phase={args.phase})")
    logger.info(f"  max_experiments={args.max_experiments}")
    logger.info(f"  branch={args.branch}")
    logger.info(f"  train_cmd={train_cmd}")
    logger.info(f"  timeout={args.timeout}s")
    logger.info(f"  results database: {db.db_path}")

    try:
        if args.phase == 0:
            results = executor.run_all_phases(start_phase=1)
        elif args.phase == 1:
            results = executor.run_phase_1_single_factor()
        elif args.phase == 2:
            results = executor.run_phase_2_combinations(
                from_results_path=Path(args.from_results) if args.from_results else None
            )
        elif args.phase == 3:
            results = executor.run_phase_3_fine_tuning()
        elif args.phase == 4:
            results = executor.run_phase_4_radical()
        else:
            logger.error(f"Unknown phase: {args.phase}")
            sys.exit(1)

        # Final report
        print("")
        report = db.generate_report()
        print(report)

        # Export TSV
        if args.export_tsv or args.phase in (0, 1, 2, 3, 4):
            db.export_tsv()

        logger.info(f"Research orchestrator complete. {len(results)} experiments executed.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving current results...")
        db._save()
        if args.export_tsv:
            db.export_tsv()
        logger.info("Results saved.")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        db._save()
        logger.info("Partial results saved before crash.")
        sys.exit(1)


if __name__ == "__main__":
    main()
