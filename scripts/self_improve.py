#!/usr/bin/env python3
"""
SELF-IMPROVEMENT SYSTEM for autoresearch-v2
============================================
Recursive meta-reasoning engine that analyzes past research efficiency
and produces concrete improvements to the research process itself.

Think of it as a research meta-review: the system examines HOW it has been
improving the model and suggests ways to improve HOW it improves.

Output files:
  - analysis_report.md          (research process health)
  - suggested_program_v2.md     (better agent instructions)
  - updated_experiment_priorities.json (revised weights)
  - process_metrics.json        (quantitative metrics)

Usage:
    python self_improve.py --results-dir results/ --knowledge results/knowledge.json
    python self_improve.py --results-dir results/ --knowledge results/knowledge.json --experiments-done 100
"""

import argparse
import calendar
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_tsv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse TSV into headers + list of dicts. Returns empty if file missing."""
    if not os.path.exists(path):
        return [], []
    with open(path, "r") as f:
        raw = f.read().strip()
    if not raw:
        return [], []
    lines = raw.split("\n")
    headers = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if not parts or parts == [""]:
            continue
        row = {}
        for i, h in enumerate(headers):
            row[h] = parts[i].strip() if i < len(parts) else ""
        rows.append(row)
    return headers, rows


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, data: Any, indent: int = 2) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, default=str))
    return str(p)


def _save_text(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)


# ─── Data Models ───────────────────────────────────────────────────────────────

class ExperimentRecord:
    """Unified experiment record from TSV or JSON results."""
    def __init__(self, row: Dict[str, str] = None, json_obj: Dict = None):
        self.row = row or {}
        self.json_obj = json_obj or {}

    @property
    def experiment_id(self) -> str:
        return self.row.get("experiment_id", "") or self.json_obj.get("experiment_id", "")

    @property
    def experiment_name(self) -> str:
        return self.row.get("experiment_name", "") or self.json_obj.get("experiment_name", "")

    @property
    def val_bpb(self) -> float:
        return _safe_float(
            self.row.get("val_bpb"),
            self.json_obj.get("val_bpb", 1.45)
        )

    @property
    def delta(self) -> float:
        return _safe_float(
            self.row.get("delta"),
            self.json_obj.get("delta", 0.0)
        )

    @property
    def status(self) -> str:
        return self.row.get("status", "") or self.json_obj.get("status", "unknown")

    @property
    def timestamp(self) -> str:
        return self.row.get("timestamp", "") or self.json_obj.get("timestamp", "")

    @property
    def duration(self) -> float:
        return _safe_float(self.row.get("duration"), self.json_obj.get("duration", 0))

    @property
    def changes(self) -> List[str]:
        raw = self.row.get("changes", "")
        if raw:
            return [c.strip() for c in raw.split("|") if c.strip()]
        return self.json_obj.get("changes_applied", [])

    @property
    def configuration(self) -> Dict:
        return self.json_obj.get("configuration", {})

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def is_failed(self) -> bool:
        return self.status in ("failed", "crashed")

    @property
    def is_baseline(self) -> bool:
        return self.status == "baseline"

    @property
    def improved(self) -> bool:
        return self.delta > 0.0001

    @property
    def description(self) -> str:
        return self.json_obj.get("description", "") or self.experiment_name


# ─── ANALYSIS MODULE ───────────────────────────────────────────────────────────

class ResearchEfficiencyAnalyzer:
    """Analyzes research process efficiency across five dimensions."""

    def __init__(self, experiments: List[ExperimentRecord]):
        self.experiments = experiments
        self.successful = [e for e in experiments if e.is_success and not e.is_baseline]
        self.failed = [e for e in experiments if e.is_failed]
        self.baselines = [e for e in experiments if e.is_baseline]
        self.all_with_delta = [e for e in self.successful if e.delta != 0.0]
        self.best_bpb = min((e.val_bpb for e in self.successful), default=1.45)
        self.baseline_bpb = min(
            (e.val_bpb for e in self.baselines), default=1.45
        )
        if not self.baselines and self.successful:
            # Use oldest as baseline proxy
            oldest = min(self.successful, key=lambda x: x.timestamp) if self.successful else None
            if oldest:
                self.baseline_bpb = oldest.val_bpb

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all five efficiency dimensions."""
        return {
            "throughput": self.experiments_per_hour(),
            "improvement": self.improvement_percentage(),
            "success_rates": self.success_rate_by_category(),
            "stagnation": self.stagnation_detection(),
            "knowledge_utilization": self.knowledge_base_utilization(),
        }

    def experiments_per_hour(self) -> Dict[str, Any]:
        """Calculate experiment throughput based on time span of experiments."""
        if not self.experiments:
            return {"rate_per_hour": 0.0, "avg_duration_min": 0.0, "min_duration_min": 0.0,
                    "max_duration_min": 0.0, "total_experiments": 0, "successful": 0}

        durations = [e.duration for e in self.experiments if e.duration > 0]
        if not durations:
            durations = [120.0]  # default 2 min if no duration data

        avg_dur = mean(durations)

        timestamps = sorted(e.timestamp for e in self.experiments if e.timestamp)
        time_span_hours = 0.0
        if len(timestamps) >= 2:
            # Handle invalid dates (e.g. March 34) by extracting date components
            def parse_ts_safe(ts_str):
                """Parse ISO timestamp, handling invalid dates by clipping to max day."""
                try:
                    return datetime.fromisoformat(ts_str)
                except ValueError:
                    # Try to fix by clipping day to 28-31
                    parts = ts_str.split("T")
                    date_part = parts[0]
                    time_part = parts[1] if len(parts) > 1 else "00:00:00"
                    y, m, d = date_part.split("-")
                    y, m, d = int(y), int(m), int(d)
                    # Clip day to valid range
                    max_day = calendar.monthrange(y, m)[1]
                    if d > max_day:
                        d = max_day
                    return datetime(y, m, d,
                                    *[int(x) for x in time_part.split(":")[:3]])

            t0 = parse_ts_safe(timestamps[0])
            t1 = parse_ts_safe(timestamps[-1])
            time_span_hours = max((t1 - t0).total_seconds() / 3600.0, 0.01)
            empirical_rate = len(self.experiments) / time_span_hours
        else:
            # Fallback: use duration-based estimate
            empirical_rate = 60.0 / avg_dur if avg_dur > 0 else 0.0

        return {
            "rate_per_hour": round(empirical_rate, 2),
            "avg_duration_min": round(avg_dur, 1),
            "min_duration_min": round(min(durations), 1),
            "max_duration_min": round(max(durations), 1),
            "total_experiments": len(self.experiments),
            "successful": len(self.successful),
            "time_span_hours": round(time_span_hours, 2),
        }

    def improvement_percentage(self) -> Dict[str, Any]:
        """What percentage of experiments actually improved val_bpb."""
        non_baseline = [e for e in self.experiments if not e.is_baseline]
        if not non_baseline:
            return {"improvement_pct_all": 0.0, "improvement_pct_successful": 0.0,
                    "total_non_baseline": 0, "improvements": 0, "total_successful": 0,
                    "failed_no_improvement": 0, "total_bpb_reduction": 0.0,
                    "avg_improvement_delta": 0.0, "max_improvement_delta": 0.0,
                    "best_val_bpb": 0.0, "baseline_bpb": 0.0}

        improvements = [e for e in non_baseline if e.is_success and e.improved]
        successful_only = [e for e in non_baseline if e.is_success]

        pct_all = len(improvements) / len(non_baseline) * 100.0
        pct_successful = len(improvements) / len(successful_only) * 100.0 if successful_only else 0.0

        deltas = [e.delta for e in improvements]
        total_improvement = self.baseline_bpb - self.best_bpb

        return {
            "improvement_pct_all": round(pct_all, 1),
            "improvement_pct_successful": round(pct_successful, 1),
            "total_non_baseline": len(non_baseline),
            "total_successful": len(successful_only),
            "improvements": len(improvements),
            "total_bpb_reduction": round(total_improvement, 5),
            "avg_improvement_delta": round(mean(deltas), 5) if deltas else 0.0,
            "max_improvement_delta": round(max(deltas), 5) if deltas else 0.0,
            "best_val_bpb": round(self.best_bpb, 4),
            "baseline_bpb": round(self.baseline_bpb, 4),
        }

    def _categorize(self, record: ExperimentRecord) -> str:
        """Infer experiment category from name/description/changes."""
        name = (record.experiment_name or record.description).lower()
        if any(k in name for k in ("depth", "head", "arch", "layer", "dim")):
            return "architectural"
        if "dropout" in name:
            return "regularization"
        if "warmup" in name:
            return "training_loop"
        if any(k in name for k in ("lr", "batch", "optim")):
            return "hyperparameter"
        changes = record.changes
        if any(k in (c.lower() if isinstance(c, str) else str(c).lower()) for c in changes for k in ("depth", "head")):
            return "architectural"
        if any(k in (c.lower() if isinstance(c, str) else str(c).lower()) for c in changes for k in ("dropout", "regular")):
            return "regularization"
        return "hyperparameter"

    def success_rate_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Which research directions produce the most improvements."""
        cats: Dict[str, Dict[str, list]] = defaultdict(lambda: {"successes": [], "failures": [], "all_deltas": []})

        for e in self.experiments:
            if e.is_baseline:
                continue
            cat = self._categorize(e)
            info = cats[cat]
            if e.is_success:
                info["successes"].append(e)
                if e.improved:
                    info["all_deltas"].append(e.delta)
            elif e.is_failed:
                info["failures"].append(e)

        result = {}
        for cat, info in cats.items():
            s = len(info["successes"])
            f = len(info["failures"])
            total = s + f
            deltas = info["all_deltas"]
            result[cat] = {
                "total": total,
                "successes": s,
                "failures": f,
                "success_rate": round(s / total, 3) if total > 0 else 0.0,
                "improvements": len(deltas),
                "avg_delta": round(mean(deltas), 5) if deltas else 0.0,
                "best_delta": round(max(deltas), 5) if deltas else 0.0,
            }
        return result

    def stagnation_detection(self) -> Dict[str, Any]:
        """Are we stuck in a local optimum?"""
        if len(self.successful) < 3:
            return {"is_stagnating": False, "reason": "Not enough data", "data_points": len(self.successful)}

        sorted_by_time = sorted(self.successful, key=lambda e: e.timestamp)
        deltas = [e.delta for e in sorted_by_time]

        n = len(deltas)
        half = n // 2
        first_half = deltas[:half]
        second_half = deltas[half:]

        first_avg = mean(first_half) if first_half else 0
        second_avg = mean(second_half) if second_half else 0

        # Check diminishing returns
        diminishing = second_avg < first_avg * 0.5

        # Check if recent experiments are barely improving
        recent_threshold = 0.002
        recent_margin = [d for d in second_half if d < recent_threshold]
        recent_failure_pct = len(recent_margin) / len(second_half) if second_half else 0

        # Check delta trend using simple slope
        if n >= 3:
            x_mean = mean(range(n))
            y_mean = mean(deltas)
            num = sum((i - x_mean) * (d - y_mean) for i, d in enumerate(deltas))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0
        else:
            slope = 0

        is_stagnating = diminishing and (recent_failure_pct > 0.5 or slope < -0.001)

        return {
            "is_stagnating": is_stagnating,
            "first_half_avg_delta": round(first_avg, 5),
            "second_half_avg_delta": round(second_avg, 5),
            "delta_slope": round(slope, 6),
            "diminishing_returns": diminishing,
            "recent_margin_failure_pct": round(recent_failure_pct, 2),
            "data_points": n,
            "trend": "declining" if slope < -0.001 else ("stable" if abs(slope) <= 0.001 else "improving"),
        }

    def knowledge_base_utilization(self) -> Dict[str, Any]:
        """Is past knowledge being used in experiment design?"""
        if not self.json_objects:
            return {"kb_available": False, "utilization": 0.0}

        # Check if later experiments build on earlier improvements
        all_configs = []
        for e in self.experiments:
            if e.configuration:
                all_configs.append((e.experiment_id, e.configuration, e.timestamp))

        if not all_configs:
            return {"kb_available": True, "utilization": 0.0, "reason": "No configurations found"}

        # Knowledge utilization: fraction of experiments that combine
        # parameters from previous improvements
        improvement_params = set()
        utilizing_count = 0
        total_with_params = 0

        for exp_id, config, ts in sorted(all_configs, key=lambda x: x[2]):
            if not improvement_params:
                # First config initializes
                improvement_params.update(config.keys())
                continue
            total_with_params += 1
            config_keys = set(config.keys())
            # Check if any param comes from prior improvements
            overlap = config_keys & improvement_params
            if overlap:
                utilizing_count += 1
            improvement_params.update(config_keys)

        utilization = utilizing_count / total_with_params if total_with_params > 0 else 0.0

        return {
            "kb_available": True,
            "experiments_with_configs": total_with_params,
            "experiments_building_on_prior": utilizing_count,
            "utilization_ratio": round(utilization, 3),
            "distinct_params_explored": len(improvement_params),
        }

    def __init__(self, experiments: List[ExperimentRecord]):
        self.experiments = experiments
        self.successful = [e for e in experiments if e.is_success and not e.is_baseline]
        self.failed = [e for e in experiments if e.is_failed]
        self.baselines = [e for e in experiments if e.is_baseline]
        self.all_with_delta = [e for e in self.successful if e.delta != 0.0]

        # Collect JSON objects for KB analysis
        self.json_objects = [e.json_obj for e in experiments if e.json_obj]

        self.best_bpb = min((e.val_bpb for e in self.successful), default=1.45)
        self.baseline_bpb = min(
            (e.val_bpb for e in self.baselines), default=1.45
        )
        if not self.baselines and self.successful:
            oldest = min(self.successful, key=lambda x: x.timestamp) if self.successful else None
            if oldest:
                self.baseline_bpb = oldest.val_bpb


# ─── PROCESS IMPROVEMENT GENERATOR ─────────────────────────────────────────────

class ProcessImprovementGenerator:
    """Generates concrete process improvements from analysis."""

    def __init__(self, efficiency: Dict[str, Any], experiments: List[ExperimentRecord]):
        self.efficiency = efficiency
        self.experiments = experiments
        self.successful = [e for e in experiments if e.is_success and not e.is_baseline]
        self.failed = [e for e in experiments if e.is_failed]

    def generate(self) -> Dict[str, Any]:
        """Produce complete process improvement recommendations."""
        return {
            "more_attempts": self._recommend_more_attempts(),
            "fewer_attempts": self._recommend_fewer_attempts(),
            "search_ranges": self._evaluate_search_ranges(),
            "blind_spots": self._identify_blind_spots(),
            "orchestration_changes": self._recommend_orchestration_changes(),
            "priority_ranking": self._rank_experiment_types(),
        }

    def _recommend_more_attempts(self) -> List[Dict[str, Any]]:
        """Which experiment types deserve more attempts."""
        recommendations = []
        rates = self.efficiency.get("success_rates", {})

        # High success rate categories with room to explore
        for cat, info in rates.items():
            if info["success_rate"] >= 0.7 and info["total"] < 10:
                recommendations.append({
                    "category": cat,
                    "reason": f"High success rate ({info['success_rate']*100:.0f}%) but only {info['total']} experiments — likely more gains available",
                    "current_success_rate": info["success_rate"],
                    "current_count": info["total"],
                    "recommended_min_attempts": max(info["total"] * 2, 10),
                    "action": "INCREASE",
                })

        # Best delta categories
        cats_by_delta = sorted(
            [(k, v["avg_delta"]) for k, v in rates.items() if v["avg_delta"] > 0],
            key=lambda x: x[1], reverse=True
        )
        if cats_by_delta and cats_by_delta[0][1] > 0.005:
            top_cat = cats_by_delta[0][0]
            if not any(r["category"] == top_cat for r in recommendations):
                recommendations.insert(0, {
                    "category": top_cat,
                    "reason": f"Highest average improvement ({cats_by_delta[0][1]:.4f} bpb) — prioritize this direction",
                    "current_success_rate": rates.get(top_cat, {}).get("success_rate", 0),
                    "current_count": rates.get(top_cat, {}).get("total", 0),
                    "recommended_min_attempts": rates.get(top_cat, {}).get("total", 0) + 5,
                    "action": "INCREASE",
                })

        return recommendations

    def _recommend_fewer_attempts(self) -> List[Dict[str, Any]]:
        """Which experiment types deserve fewer attempts."""
        recommendations = []
        rates = self.efficiency.get("success_rates", {})

        for cat, info in rates.items():
            if info["success_rate"] <= 0.3 and info["total"] >= 3:
                recommendations.append({
                    "category": cat,
                    "reason": f"Low success rate ({info['success_rate']*100:.0f}%) across {info['total']} attempts — investigate before continuing",
                    "current_success_rate": info["success_rate"],
                    "current_count": info["total"],
                    "recommended_max_attempts": info["total"] + 1,  # One more to confirm
                    "action": "REDUCE",
                })

        return recommendations

    def _evaluate_search_ranges(self) -> List[Dict[str, Any]]:
        """Are search ranges appropriate."""
        evaluations = []

        # Analyze per-parameter value distributions
        param_values: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for e in self.experiments:
            if e.json_obj and "configuration" in e.json_obj:
                for key, val in e.json_obj["configuration"].items():
                    status = "success" if e.improved else "fail"
                    if isinstance(val, (int, float)):
                        param_values[key].append((val, status))

        for param, values in param_values.items():
            success_vals = [v for v, s in values if s == "success"]
            fail_vals = [v for v, s in values if s == "fail"]

            if not success_vals:
                evaluations.append({
                    "param": param,
                    "recommendation": "EXPAND",
                    "reason": "No successful values found — try drastically different range",
                    "successful_range": [],
                    "tested_range": [v for v, _ in values],
                })
                continue

            s_min, s_max = min(success_vals), max(success_vals)
            tested_min = min(v for v, _ in values)
            tested_max = max(v for v, _ in values)

            # If all successes are clustered in a narrow band
            success_spread = s_max - s_min
            total_spread = tested_max - tested_min

            if total_spread > 0 and success_spread / total_spread < 0.3:
                evaluations.append({
                    "param": param,
                    "recommendation": "NARROW",
                    "reason": f"Successes cluster in [{s_min}, {s_max}] — focus search in this region",
                    "successful_range": [s_min, s_max],
                    "tested_range": [tested_min, tested_max],
                })

        return evaluations

    def _identify_blind_spots(self) -> List[str]:
        """What areas have we not explored at all."""
        blind_spots = []
        known_categories = {
            "architectural", "hyperparameter", "regularization",
            "training_loop", "data_pipeline", "optimization"
        }
        explored = set(self.efficiency.get("success_rates", {}).keys())

        unexplored = known_categories - explored
        if unexplored:
            blind_spots.append(f"Completely unexplored categories: {', '.join(sorted(unexplored))}")

        # Check for combo experiments
        combos = [e for e in self.experiments if len(e.configuration) >= 3]
        if not combos and len(self.experiments) > 10:
            blind_spots.append("No multi-parameter combination experiments despite having 10+ single-factor runs")

        # Check for extreme value testing
        param_extremes = defaultdict(lambda: {"min": None, "max": None})
        for e in self.experiments:
            for k, v in e.configuration.items():
                if isinstance(v, (int, float)):
                    if param_extremes[k]["min"] is None or v < param_extremes[k]["min"]:
                        param_extremes[k]["min"] = v
                    if param_extremes[k]["max"] is None or v > param_extremes[k]["max"]:
                        param_extremes[k]["max"] = v

        if not blind_spots:
            blind_spots.append(
                "No obvious blind spots detected — consider cross-category interaction experiments"
            )

        return blind_spots

    def _recommend_orchestration_changes(self) -> List[Dict[str, str]]:
        """Should orchestration parameters change."""
        changes = []
        throughput = self.efficiency.get("throughput", {})
        stagnation = self.efficiency.get("stagnation", {})
        improvement = self.efficiency.get("improvement", {})

        # Adjust batch parallelism based on success rate
        success_rate = improvement.get("improvement_pct_all", improvement.get("improvement_pct_successful", 50))
        if success_rate < 30:
            changes.append({
                "param": "exploration_strategy",
                "current": "broad_sampling",
                "recommended": "focused_refinement",
                "reason": f"Low success rate ({success_rate}%) — narrow exploration, intensify around known-good configurations",
            })
        elif success_rate > 70:
            changes.append({
                "param": "exploration_strategy",
                "current": "broad_sampling",
                "recommended": "aggressive_exploration",
                "reason": f"High success rate ({success_rate}%) — system is finding improvements easily, expand search space",
            })

        # Stagnation response
        if stagnation.get("is_stagnating"):
            changes.append({
                "param": "stagnation_response",
                "current": "continue",
                "recommended": "reset_and_explore",
                "reason": "Stagnation detected — try radical departures from current best configuration",
            })

        return changes

    def _rank_experiment_types(self) -> Dict[str, float]:
        """Rank experiment types by expected value."""
        rates = self.efficiency.get("success_rates", {})
        ranking = {}
        for cat, info in rates.items():
            # Expected value = success_rate * avg_improvement * reliability_weight
            reliability = min(info["total"] / 5.0, 1.0)  # More data = more reliable
            ev = info["success_rate"] * info["avg_delta"] * (0.5 + 0.5 * reliability)
            ranking[cat] = round(ev, 6)

        return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))


# ─── PROGRAM.MD EVOLUTION ────────────────────────────────────────────────────

class ProgramMdEvolver:
    """Analyzes what parts of the research process worked and suggests improvements."""

    # Default process instructions (simulating typical program.md content)
    DEFAULT_PROCESS_SECTIONS = [
        "1. Run baseline to establish current best val_bpb",
        "2. Test individual hyperparameter changes (LR, batch size, warmup)",
        "3. Test architectural changes (depth, heads, attention patterns)",
        "4. Test regularization strategies (dropout, weight decay)",
        "5. Combine top-performing changes into multi-factor experiments",
        "6. Run confirmation experiments on best configuration",
        "7. Update knowledge base with results",
    ]

    def __init__(self, efficiency: Dict[str, Any],
                 improvements: Dict[str, Any],
                 experiments: List[ExperimentRecord]):
        self.efficiency = efficiency
        self.improvements = improvements
        self.experiments = experiments
        self.successful = [e for e in experiments if e.is_success and not e.is_baseline]

    def generate_v2(self) -> str:
        """Produce suggested_program_v2.md."""
        rates = self.efficiency.get("success_rates", {})
        stagnation = self.efficiency.get("stagnation", {})
        throughput = self.efficiency.get("throughput", {})
        improvement = self.efficiency.get("improvement", {})

        # Determine what worked
        best_categories = sorted(
            [(k, v) for k, v in rates.items()],
            key=lambda x: x[1]["avg_delta"], reverse=True
        )
        worst_categories = sorted(
            [(k, v) for k, v in rates.items()],
            key=lambda x: x[1]["success_rate"]
        )

        now = _now()
        version = "1.0"  # Track version

        md = []
        md.append(f"# Suggested ReSEARCH Process V2")
        md.append(f"")
        md.append(f"Generated: {now}")
        md.append(f"Based on: {len(self.experiments)} experiments")
        md.append(f"Current best val_bpb: {improvement.get('best_val_bpb', 'N/A')}")
        md.append(f"Baseline val_bpb: {improvement.get('baseline_bpb', 'N/A')}")
        md.append(f"Total improvement: {improvement.get('total_bpb_reduction', 0):.4f} bpb")
        md.append(f"")

        md.append("## What Worked")
        md.append("")
        for cat, info in best_categories[:3]:
            md.append(f"- **{cat}**: {info['success_rate']*100:.0f}% success rate, avg delta {info['avg_delta']:.4f} ({info['total']} experiments)")
        md.append("")

        md.append("## What Didn't Work")
        md.append("")
        for cat, info in worst_categories:
            if info["success_rate"] < 0.5:
                md.append(f"- **{cat}**: {info['success_rate']*100:.0f}% success rate ({info['failures']} failures out of {info['total']})")
        if not any(v["success_rate"] < 0.5 for _, v in worst_categories):
            md.append("- No category has a failure rate above 50%")
        md.append("")

        md.append("## Stagnation Status")
        md.append("")
        if stagnation.get("is_stagnating"):
            md.append("**WARNING: Stagnation detected** — recent experiments show diminishing returns.")
            md.append(f"Trend: {stagnation.get('trend', 'unknown')}")
            md.append(f"Slope: {stagnation.get('delta_slope', 0):.6f}")
        else:
            md.append("**No stagnation detected** — improvements are continuing.")
            md.append(f"Trend: {stagnation.get('trend', 'unknown')}")
        md.append("")

        md.append("## V2 Process Instructions")
        md.append("")
        md.append("### Phase 1: Focused Exploration")
        md.append("")
        # Recommend starting with highest-ROI categories
        top_cats = [c for c, _ in best_categories[:2]] if best_categories else ["architectural", "hyperparameter"]
        md.append(f"1. Start with **{top_cats[0]}** experiments (highest historical success rate)")
        md.append(f"   - This category has a {rates.get(top_cats[0], {}).get('success_rate', 0)*100:.0f}% success rate")
        md.append(f"   - Run at least {max(rates.get(top_cats[0], {}).get('total', 3) * 2, 5)} experiments before moving on")
        md.append("")
        md.append(f"2. Test **{top_cats[1]}** experiments")
        md.append(f"   - Run targeted experiments, avoid blind parameter sweeps")
        md.append("")

        md.append("### Phase 2: Combination & Interaction")
        md.append("")
        md.append("3. Combine top parameters from Phase 1 into multi-factor experiments")
        md.append("   - Test 2-parameter combinations first")
        md.append("   - Then test 3+ parameter combinations with proven-good values")
        md.append("   - Track interaction effects (synergy vs conflict)")
        md.append("")

        md.append("### Phase 3: Refinement & Confirmation")
        md.append("")
        md.append("4. Zoom in on best configuration region")
        md.append("   - Reduce search range to +/- 20% of best values")
        md.append("   - Run 3x confirmation on the single best config")
        md.append("")

        md.append("5. Update knowledge base with ALL results (success, failure, and crash)")
        md.append("   - Record dead ends to avoid repeating")
        md.append("   - Track parameter interactions")
        md.append("")

        md.append("### Orchestration Parameters")
        md.append("")
        md.append(f"- Current throughput: {throughput.get('rate_per_hour', 'N/A')} experiments/hour")
        md.append(f"- Target success rate: >60%")
        imp_pct = improvement.get("improvement_pct_all", improvement.get("improvement_pct_successful", 0))
        if imp_pct < 60:
            md.append(f"- Current success rate: {imp_pct:.0f}% — **BELOW TARGET**")
            md.append("  -> Reduce exploration breadth, increase exploitation of known-good regions")
        md.append("")

        md.append("### Self-Improvement Loop")
        md.append("")
        md.append(f"6. After every N experiments, re-run self_improve.py to update this document")
        md.append(f"7. Compare current process against previous version")
        md.append(f"8. Archive versions of this document for tracking")
        md.append("")

        md.append("---")
        md.append(f"*Auto-generated by self_improve.py at {now}*")

        return "\n".join(md)


# ─── EXPERIMENT CATALOG EVOLVER ───────────────────────────────────────────────

class ExperimentCatalogEvolver:
    """Updates experiment designer priorities based on results."""

    # Default prioritis from experiment_designer.py patterns
    DEFAULT_PRIORITIES = {
        "architecture": {"weight": 0.30, "min_attempts": 3, "max_attempts": 20},
        "hyperparameter": {"weight": 0.25, "min_attempts": 3, "max_attempts": 25},
        "regularization": {"weight": 0.15, "min_attempts": 2, "max_attempts": 10},
        "training_loop": {"weight": 0.15, "min_attempts": 2, "max_attempts": 10},
        "novel": {"weight": 0.15, "min_attempts": 1, "max_attempts": 8},
    }

    def __init__(self, efficiency: Dict[str, Any],
                 experiments: List[ExperimentRecord]):
        self.efficiency = efficiency
        self.experiments = experiments
        self.successful = [e for e in experiments if e.is_success and not e.is_baseline]
        self.failed = [e for e in experiments if e.is_failed]

    def _categorize(self, record: ExperimentRecord) -> str:
        name = (record.experiment_name or record.description).lower()
        if any(k in name for k in ("depth", "head", "arch", "layer")):
            return "architecture"
        if "dropout" in name:
            return "regularization"
        if "warmup" in name:
            return "training_loop"
        if any(k in name for k in ("lr", "batch", "optim")):
            return "hyperparameter"
        return "hyperparameter"

    def evolve_priorities(self) -> Dict[str, Any]:
        """Return updated priority weights."""
        rates = self.efficiency.get("success_rates", {})

        # Category name mapping
        cat_map = {
            "architectural": "architecture",
            "regularization": "regularization",
            "training_loop": "training_loop",
            "hyperparameter": "hyperparameter",
        }

        new_priorities = {}

        for raw_cat, info in rates.items():
            mapped = cat_map.get(raw_cat, raw_cat)
            current = self.DEFAULT_PRIORITIES.get(mapped, {"weight": 0.15, "min_attempts": 1, "max_attempts": 10})

            # Adjust weight based on performance
            performance_score = info["success_rate"] * info["avg_delta"] * 100  # Expected value
            # Normalize: higher = better
            weight_factor = 0.5 + 0.5 * (performance_score / max(max(v.get("avg_delta", 0.001), 0.001) for v in rates.values()) if rates else 1.0)

            # Clamp weight
            new_weight = round(current["weight"] * weight_factor, 3)
            new_weight = max(0.05, min(0.50, new_weight))

            # Adjust attempt ranges
            if info["success_rate"] >= 0.7:
                new_min = int(current.get("min_attempts", 3) * 1.5)
                new_max = int(current.get("max_attempts", 20) * 1.5)
            elif info["success_rate"] <= 0.3:
                new_min = current.get("min_attempts", 3)
                new_max = current.get("max_attempts", 20) + 1  # One more to confirm
            else:
                new_min = current.get("min_attempts", 3)
                new_max = current.get("max_attempts", 20)

            new_priorities[mapped] = {
                "weight": new_weight,
                "min_attempts": new_min,
                "max_attempts": new_max,
                "previous_weight": current["weight"],
                "change_reason": (
                    f"success_rate={info['success_rate']}, avg_delta={info['avg_delta']:.4f}"
                ),
            }

        # Normalize weights to sum to 1.0
        total_weight = sum(v["weight"] for v in new_priorities.values())
        if total_weight > 0:
            for cat in new_priorities:
                new_priorities[cat]["weight"] = round(
                    new_priorities[cat]["weight"] / total_weight, 3
                )

        # Add categories that were never tried
        for cat, defaults in self.DEFAULT_PRIORITIES.items():
            if cat not in new_priorities:
                new_priorities[cat] = {
                    **defaults,
                    "note": "Untested — default priority applied",
                }

        return new_priorities

    def get_archived_types(self) -> List[str]:
        """Types that should be permanently archived (consistently failing)."""
        archived = []
        rates = self.efficiency.get("success_rates", {})

        for cat, info in rates.items():
            if info["total"] >= 3 and info["success_rate"] <= 0.2:
                archived.append({
                    "category": cat,
                    "reason": f"Only {info['success_rate']*100:.0f}% success rate over {info['total']} attempts",
                    "recommendation": "Archive — do not attempt without major redesign",
                })

        # Check specific experiment names
        failing_names = [
            e.experiment_name
            for e in self.failed
            if e.experiment_name
        ]

        return archived

    def get_new_experiment_types(self) -> List[Dict[str, Any]]:
        """Newly-discovered experiment types to add."""
        new_types = []

        # Look for combo experiments that succeeded
        combo_exps = [
            e for e in self.successful
            if len(e.configuration) >= 2 and e.improved
        ]

        seen_combos = set()
        for e in combo_exps:
            keys = tuple(sorted(e.configuration.keys()))
            if keys not in seen_combos:
                seen_combos.add(keys)
                new_types.append({
                    "type": f"combo_{'_'.join(keys)}",
                    "description": f"Multi-factor: {', '.join(f'{k}={v}' for k, v in e.configuration.items())}",
                    "val_bpb": e.val_bpb,
                    "delta": e.delta,
                    "priority_boost": 1.5 if e.delta > 0.005 else 1.0,
                    "recommendation": "EXPLORE" if e.delta > 0.01 else "MONITOR",
                })

        return new_types


# ─── REPORT GENERATORS ────────────────────────────────────────────────────────

def generate_analysis_report(efficiency: Dict, improvements: Dict,
                             experiments: List[ExperimentRecord]) -> str:
    """Generate analysis_report.md."""
    throughput = efficiency.get("throughput", {})
    improvement = efficiency.get("improvement", {})
    rates = efficiency.get("success_rates", {})
    stagnation = efficiency.get("stagnation", {})
    kb_util = efficiency.get("knowledge_utilization", {})

    md = []
    md.append("# Research Process Analysis Report")
    md.append("")
    md.append(f"Generated: {_now()}")
    md.append(f"Total experiments analyzed: {len(experiments)}")
    md.append("")

    md.append("## 1. Throughput")
    md.append("")
    md.append(f"- **Rate**: {throughput.get('rate_per_hour', 0):.1f} experiments/hour")
    md.append(f"- **Avg duration**: {throughput.get('avg_duration_min', 0):.1f} min")
    md.append(f"- **Time span**: {throughput.get('time_span_hours', 0):.1f} hours")
    md.append(f"- **Total experiments**: {throughput.get('total_experiments', 0)}")
    md.append(f"  - Successful: {throughput.get('successful', 0)}")
    md.append(f"  - Failed: {len([e for e in experiments if e.is_failed])}")
    md.append("")

    md.append("## 2. Improvement Rate")
    md.append("")
    md.append(f"- **Improvement rate**: {improvement.get('improvement_pct_all', 0):.1f}% of all experiments (non-baseline) improved val_bpb")
    md.append(f"- **Among successful**: {improvement.get('improvement_pct_successful', 0):.1f}% of successful experiments showed improvement")
    md.append(f"- **Total BPB reduction**: {improvement.get('total_bpb_reduction', 0):.5f}")
    md.append(f"- **Best val_bpb**: {improvement.get('best_val_bpb', 0):.4f}")
    md.append(f"- **Baseline val_bpb**: {improvement.get('baseline_bpb', 0):.4f}")
    md.append(f"- **Avg improvement delta**: {improvement.get('avg_improvement_delta', 0):.5f}")
    md.append(f"- **Max improvement delta**: {improvement.get('max_improvement_delta', 0):.5f}")
    md.append("")

    md.append("## 3. Success Rate by Category")
    md.append("")
    md.append("| Category | Attempts | Successes | Failures | Success Rate | Avg Delta | Best Delta |")
    md.append("|----------|----------|-----------|----------|--------------|-----------|------------|")
    for cat, info in sorted(rates.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        md.append(
            f"| {cat} | {info['total']} | {info['successes']} | "
            f"{info['failures']} | {info['success_rate']*100:.0f}% | "
            f"{info['avg_delta']:.5f} | {info['best_delta']:.5f} |"
        )
    md.append("")

    md.append("## 4. Stagnation Detection")
    md.append("")
    md.append(f"- **Stagnating**: {'YES' if stagnation.get('is_stagnating') else 'No'}")
    md.append(f"- **Trend**: {stagnation.get('trend', 'unknown')}")
    md.append(f"- **Delta slope**: {stagnation.get('delta_slope', 0):.6f}")
    md.append(f"- **First half avg delta**: {stagnation.get('first_half_avg_delta', 0):.5f}")
    md.append(f"- **Second half avg delta**: {stagnation.get('second_half_avg_delta', 0):.5f}")
    md.append(f"- **Diminishing returns**: {'Yes' if stagnation.get('diminishing_returns') else 'No'}")
    md.append("")

    md.append("## 5. Knowledge Base Utilization")
    md.append("")
    md.append(f"- **KB available**: {'Yes' if kb_util.get('kb_available') else 'No'}")
    md.append(f"- **Utilization ratio**: {kb_util.get('utilization_ratio', 0):.1%}")
    md.append(f"- **Experiments building on prior**: {kb_util.get('experiments_building_on_prior', 0)}")
    md.append(f"- **Distinct params explored**: {kb_util.get('distinct_params_explored', 0)}")
    md.append("")

    md.append("## 6. Process Improvement Recommendations")
    md.append("")

    more = improvements.get("more_attempts", [])
    fewer = improvements.get("fewer_attempts", [])

    if more:
        md.append("### Increase Effort On:")
        for m in more:
            md.append(f"- **{m['category']}**: {m['reason']}")
        md.append("")

    if fewer:
        md.append("### Reduce Effort On:")
        for f_item in fewer:
            md.append(f"- **{f_item['category']}**: {f_item['reason']}")
        md.append("")

    md.append("### Orchestration Adjustments:")
    for oc in improvements.get("orchestration_changes", []):
        md.append(f"- **{oc['param']}**: {oc['current']} -> {oc['recommended']} ({oc['reason']})")
    md.append("")

    # Priority ranking
    ranking = improvements.get("priority_ranking", {})
    if ranking:
        md.append("### Experiment Type Rankings (by expected value):")
        for i, (cat, ev) in enumerate(ranking.items(), 1):
            md.append(f"  {i}. {cat}: {ev:.4f}")
        md.append("")

    md.append("---")
    md.append(f"*Auto-generated by self_improve.py*")
    return "\n".join(md)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Improvement System for autoresearch-v2"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/",
        help="Directory containing results.tsv and research_results.json"
    )
    parser.add_argument(
        "--knowledge", type=str, default=None,
        help="Path to knowledge.json"
    )
    parser.add_argument(
        "--experiments-done", type=int, default=None,
        help="Override: number of experiments completed (for tracking)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write output files (default: results/)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────────────────

    tsv_headers, tsv_rows = _load_tsv(str(results_dir / "results.tsv"))
    json_data = _load_json(str(results_dir / "research_results.json"))
    kb_data = _load_json(args.knowledge) if args.knowledge else None

    print(f"Loaded {len(tsv_rows)} experiments from results.tsv")
    print(f"Loaded JSON results: {'Yes' if json_data else 'No'}")
    print(f"Loaded knowledge base: {'Yes' if kb_data else 'No'}")

    # Parse all experiment records
    experiments = []
    exp_ids_seen = set()

    # Add TSV records
    for row in tsv_rows:
        exp_id = row.get("experiment_id", "")
        if exp_id not in exp_ids_seen:
            experiments.append(ExperimentRecord(row=row))
            exp_ids_seen.add(exp_id)

    # Add JSON records (enrich with config data)
    if json_data and "results" in json_data:
        for obj in json_data["results"]:
            exp_id = obj.get("experiment_id", "")
            if exp_id not in exp_ids_seen:
                experiments.append(ExperimentRecord(json_obj=obj))
                exp_ids_seen.add(exp_id)
            else:
                # Enrich: find matching TSV record and add JSON data
                for exp in experiments:
                    if exp.row.get("experiment_id") == exp_id:
                        exp.json_obj = obj
                        break

    # Override experiment count if specified
    n_done = args.experiments_done or len(experiments)

    print(f"Total unique experiments: {len(experiments)}")

    if not experiments:
        print("ERROR: No experiment data found. Check --results-dir and file contents.")
        sys.exit(1)

    # ── Analyze ────────────────────────────────────────────────────────────

    print("\nRunning research efficiency analysis...")
    analyzer = ResearchEfficiencyAnalyzer(experiments)
    efficiency = analyzer.compute_metrics()

    throughput = efficiency.get("throughput", {})
    improvement = efficiency.get("improvement", {})
    rates = efficiency.get("success_rates", {})
    stagnation = efficiency.get("stagnation", {})
    kb_util = efficiency.get("knowledge_utilization", {})

    print(f"  Throughput: {throughput.get('rate_per_hour', 0):.1f} exp/hr")
    print(f"  Improvement rate: {improvement.get('improvement_pct_all', improvement.get('improvement_pct_successful', 0)):.1f}%")
    print(f"  Stagnating: {stagnation.get('is_stagnating', False)}")
    print(f"  KB utilization: {kb_util.get('utilization_ratio', 0):.1%}")

    # ── Process Improvements ───────────────────────────────────────────────

    print("\nGenerating process improvements...")
    gen = ProcessImprovementGenerator(efficiency, experiments)
    improvements = gen.generate()

    more = improvements.get("more_attempts", [])
    fewer = improvements.get("fewer_attempts", [])
    ranges = improvements.get("search_ranges", [])
    spots = improvements.get("blind_spots", [])
    orchest = improvements.get("orchestration_changes", [])
    ranking = improvements.get("priority_ranking", {})

    print(f"  Recommended increases: {len(more)}")
    print(f"  Recommended reductions: {len(fewer)}")
    print(f"  Search range adjustments: {len(ranges)}")
    print(f"  Blind spots: {len(spots)}")

    # ── Program.md Evolution ───────────────────────────────────────────────

    print("\nGenerating V2 process instructions...")
    evolver = ProgramMdEvolver(efficiency, improvements, experiments)
    program_v2 = evolver.generate_v2()

    # ── Experiment Catalog Evolution ───────────────────────────────────────

    print("Evolving experiment catalog priorities...")
    cat_evolver = ExperimentCatalogEvolver(efficiency, experiments)
    new_priorities = cat_evolver.evolve_priorities()
    archived_types = cat_evolver.get_archived_types()
    new_types = cat_evolver.get_new_experiment_types()

    print(f"  Updated {len(new_priorities)} category priorities")
    print(f"  Archived types: {len(archived_types)}")
    print(f"  New combo types discovered: {len(new_types)}")

    # ── Write Outputs ──────────────────────────────────────────────────────

    print("\nWriting output files...")

    # 1. analysis_report.md
    report_md = generate_analysis_report(efficiency, improvements, experiments)
    report_path = _save_text(str(output_dir / "analysis_report.md"), report_md)
    print(f"  Written: {report_path}")

    # 2. suggested_program_v2.md
    prog_path = _save_text(str(output_dir / "suggested_program_v2.md"), program_v2)
    print(f"  Written: {prog_path}")

    # 3. updated_experiment_priorities.json
    priorities_output = {
        "generated_at": _now(),
        "priorities": new_priorities,
        "archived_types": archived_types,
        "new_experiment_types": new_types,
        "metadata": {
            "experiments_analyzed": len(experiments),
            "success_rate_by_category": rates,
        }
    }
    pri_path = _save_json(str(output_dir / "updated_experiment_priorities.json"), priorities_output)
    print(f"  Written: {pri_path}")

    # 4. process_metrics.json
    metrics_output = {
        "generated_at": _now(),
        "experiments_done": n_done,
        "efficiency": efficiency,
        "process_improvements": improvements,
        "catalog_evolution": {
            "archived_types": archived_types,
            "new_types": new_types,
            "priority_changes": {
                cat: {"old": p.get("previous_weight"), "new": p["weight"]}
                for cat, p in new_priorities.items()
                if "previous_weight" in p
            }
        }
    }
    metrics_path = _save_json(str(output_dir / "process_metrics.json"), metrics_output)
    print(f"  Written: {metrics_path}")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"Experiments analyzed: {len(experiments)}")
    print(f"Throughput: {throughput.get('rate_per_hour', 0):.1f} experiments/hour")
    print(f"Improvement rate: {improvement.get('improvement_pct_all', improvement.get('improvement_pct_successful', 0)):.1f}%")
    print(f"Best val_bpb: {improvement.get('best_val_bpb', 'N/A')}")
    print(f"Total reduction: {improvement.get('total_bpb_reduction', 0):.5f} bpb")
    print(f"Stagnating: {'YES' if stagnation.get('is_stagnating') else 'No'}")
    print(f"")
    if more:
        print("DO MORE of:")
        for m in more[:3]:
            print(f"  - {m['category']}: {m['reason']}")
    if fewer:
        print("DO LESS of:")
        for f_item in fewer[:3]:
            print(f"  - {f_item['category']}: {f_item['reason']}")
    print("")
    print(f"Files written to: {output_dir}/")
    print("  - analysis_report.md")
    print("  - suggested_program_v2.md")
    print("  - updated_experiment_priorities.json")
    print("  - process_metrics.json")


if __name__ == "__main__":
    main()
