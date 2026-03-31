#!/usr/bin/env python3
"""
META-ANALYSIS ENGINE for autoresearch-v2

Analyzes experiment results.tsv to produce insights, hypotheses, 
and prioritized next-experiment recommendations.

Usage:
    python meta_analyzer.py --results results/results.tsv --output analysis.md
    python meta_analyzer.py --results results/results.tsv --hypotheses
    python meta_analyzer.py --results results/results.tsv --next-experiments 10
    python meta_analyzer.py --results results/results.tsv --output analysis.md --next-experiments 5

Uses only Python stdlib + numpy.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ─── Known change metadata ────────────────────────────────────────────────────
# Maps change names to their category, description, and parameter keys.
# Extracted from research_orchestrator.py Register.
CHANGE_CATALOG: Dict[str, Dict[str, Any]] = {
    # Learning rate
    "lr_1e3":   {"category": "lr",      "desc": "LR 1e-3",             "keys": ["learning_rate"]},
    "lr_3e4":   {"category": "lr",      "desc": "LR 3e-4",             "keys": ["learning_rate"]},
    "lr_1e4":   {"category": "lr",      "desc": "LR 1e-4",             "keys": ["learning_rate"]},
    "lr_5e4_cosine": {"category": "lr", "desc": "LR 5e-4 cosine",      "keys": ["learning_rate", "scheduler"]},
    "warmup_1000":   {"category": "lr",  "desc": "Warmup 1000",          "keys": ["warmup_steps"]},
    "warmup_5000":   {"category": "lr",  "desc": "Warmup 5000",          "keys": ["warmup_steps"]},
    # Architecture depth/dim
    "depth_8":  {"category": "arch",    "desc": "8 layers",             "keys": ["n_layers"]},
    "depth_16": {"category": "arch",    "desc": "16 layers",            "keys": ["n_layers"]},
    "depth_20": {"category": "arch",    "desc": "20 layers",            "keys": ["n_layers"]},
    "dim_512":  {"category": "arch",    "desc": "dim 512",              "keys": ["d_model"]},
    "dim_1024": {"category": "arch",    "desc": "dim 1024",             "keys": ["d_model"]},
    # Attention
    "heads_8":  {"category": "arch",    "desc": "8 heads",              "keys": ["n_heads"]},
    "gqa_4_2":  {"category": "arch",    "desc": "GQA 4/2",              "keys": ["n_heads", "n_kv_heads"]},
    # Batch size
    "batch_32":            {"category": "opt",  "desc": "Batch 32",              "keys": ["batch_size"]},
    "batch_128":           {"category": "opt",  "desc": "Batch 128",             "keys": ["batch_size"]},
    "batch_256_grad_accum":{"category": "opt",  "desc": "Batch 256 accum",       "keys": ["batch_size", "grad_accum_steps"]},
    # Adam
    "adamw_1e1":  {"category": "opt",  "desc": "AdamW eps 1e-1",       "keys": ["adam_epsilon"]},
    "adam_beta2_95":  {"category": "opt",  "desc": "Adam b2 0.95",       "keys": ["adam_beta2"]},
    "adam_beta2_999": {"category": "opt",  "desc": "Adam b2 0.999",      "keys": ["adam_beta2"]},
    # Weight decay
    "wt_decay_01":  {"category": "reg",   "desc": "WD 0.1",              "keys": ["weight_decay"]},
    "wt_decay_001": {"category": "reg",   "desc": "WD 0.01",             "keys": ["weight_decay"]},
    # Dropout
    "dropout_1": {"category": "reg",   "desc": "Dropout 0.1",           "keys": ["dropout"]},
    "dropout_0": {"category": "reg",   "desc": "No dropout",            "keys": ["dropout"]},
    # RoPE
    "rope_theta_5e5": {"category": "arch",  "desc": "RoPE 5e5",         "keys": ["rope_theta"]},
    "rope_theta_1e7": {"category": "arch",  "desc": "RoPE 1e7",         "keys": ["rope_theta"]},
    # Sequence length
    "seq_4096": {"category": "arch",    "desc": "Seq 4096",             "keys": ["max_seq_len"]},
    "seq_8192": {"category": "arch",    "desc": "Seq 8192",             "keys": ["max_seq_len"]},
    # Architecture tweaks
    "swiglu_ffn": {"category": "arch",  "desc": "SwiGLU FFN",           "keys": ["use_swiglu"]},
    "no_bias":    {"category": "arch",  "desc": "Remove bias",          "keys": ["use_bias"]},
    "pre_ln":     {"category": "arch",  "desc": "Pre-LayerNorm",        "keys": ["pre_norm"]},
    "alibi":      {"category": "arch",  "desc": "ALiBi pos emb",        "keys": ["use_alibi"]},
    "qk_norm":    {"category": "arch",  "desc": "QK norm",              "keys": ["qk_norm"]},
    # Data pipeline
    "packed_data": {"category": "data",  "desc": "Packed data",         "keys": ["use_packing"]},
    "fsl_shard":   {"category": "data",  "desc": "FW-edu-sharded",      "keys": ["dataset"]},
    "dclm_mix":    {"category": "data",  "desc": "DCLM+FW mix",         "keys": ["dataset", "dclm_ratio"]},
    # Gradient clipping
    "grad_clip_05": {"category": "opt",  "desc": "Grad clip 0.5",       "keys": ["max_grad_norm"]},
    "grad_clip_20": {"category": "opt",  "desc": "Grad clip 2.0",       "keys": ["max_grad_norm"]},
    # MoE
    "moe_4_2": {"category": "arch",     "desc": "MoE 4/2",             "keys": ["num_experts", "num_active_experts"]},
    "moe_8_2": {"category": "arch",     "desc": "MoE 8/2",             "keys": ["num_experts", "num_active_experts"]},
}

CATEGORY_LABELS = {
    "lr":   "Learning Rate",
    "arch": "Architecture",
    "opt":  "Optimizer",
    "reg":  "Regularization",
    "data": "Data Pipeline",
}

# Change name prefixes that map to categories (for unknown changes)
PREFIX_CATEGORY_MAP = {
    "lr_": "lr",
    "warmup_": "lr",
    "depth_": "arch",
    "dim_": "arch",
    "heads_": "arch",
    "gqa_": "arch",
    "seq_": "arch",
    "swiglu_": "arch",
    "no_bias": "arch",
    "pre_ln": "arch",
    "alibi": "arch",
    "qk_norm": "arch",
    "moe_": "arch",
    "rope_": "arch",
    "batch_": "opt",
    "adamw_": "opt",
    "adam_": "opt",
    "grad_clip_": "opt",
    "wt_decay_": "reg",
    "dropout_": "reg",
    "packed_": "data",
    "fsl_": "data",
    "dclm_": "data",
}


# ─── Data Loading ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentRow:
    """Parsed row from results.tsv."""
    experiment_id: str
    experiment_name: str
    val_bpb: Optional[float]
    delta: Optional[float]
    phase: int
    status: str
    timestamp: str
    duration: float
    changes: List[str]

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
    def improved(self) -> Optional[bool]:
        """True if delta > 0 (improvement over baseline)."""
        if self.delta is None:
            return None
        return self.delta > 0.0001  # small epsilon for floating point


def parse_tsv(path: str) -> List[ExperimentRow]:
    """Parse results.tsv into ExperimentRow list."""
    rows: List[ExperimentRow] = []
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")

    if not lines:
        return rows

    # Detect header
    header = lines[0].lower()
    has_header = "experiment_id" in header or "commit" in header
    start = 1 if has_header else 0

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")

        # Determine format: could be the orchestrator TSV format
        # experiment_id, experiment_name, val_bpb, delta, phase, status, timestamp, duration, changes
        def safe_float(v: str) -> Optional[float]:
            v = v.strip()
            if v in ("N/A", "", "None", "null"):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def safe_int(v: str) -> int:
            try:
                return int(float(v.strip()))
            except (ValueError, TypeError):
                return 0

        try:
            if len(parts) >= 9:
                row = ExperimentRow(
                    experiment_id=parts[0].strip(),
                    experiment_name=parts[1].strip(),
                    val_bpb=safe_float(parts[2]),
                    delta=safe_float(parts[3]),
                    phase=safe_int(parts[4]),
                    status=parts[5].strip(),
                    timestamp=parts[6].strip(),
                    duration=safe_float(parts[7]) or 0.0,
                    changes=[c.strip() for c in parts[8].split("|") if c.strip()],
                )
                rows.append(row)
            elif len(parts) >= 5:
                # Fallback: commit, val_bpb, memory_gb, status, description
                row = ExperimentRow(
                    experiment_id=parts[0].strip(),
                    experiment_name=parts[0].strip(),
                    val_bpb=safe_float(parts[1]),
                    delta=None,
                    phase=0,
                    status=parts[3].strip() if len(parts) > 3 else "unknown",
                    timestamp="",
                    duration=0.0,
                    changes=[],
                )
                # Try to extract change name from experiment_name/description
                if len(parts) > 4 and parts[4].strip():
                    desc = parts[4].strip()
                    # Try to extract change names
                    for change_name in CHANGE_CATALOG:
                        if change_name in desc:
                            row.changes.append(change_name)
                rows.append(row)
        except (IndexError, ValueError) as e:
            # Skip malformed rows
            continue

    return rows


# ─── Analysis Engine ──────────────────────────────────────────────────────────

class AnalysisEngine:
    """Core statistical analysis of experiment results."""

    def __init__(self, rows: List[ExperimentRow]):
        self.rows = rows
        self.successful = [r for r in rows if r.is_success]
        self.failed = [r for r in rows if r.is_failed]
        self.baselines = [r for r in rows if r.is_baseline]
        self.baseline_bpb = self._compute_baseline()

    def _compute_baseline(self) -> float:
        """Get baseline val_bpb: average of baseline rows, or median of all."""
        if self.baselines:
            vals = [r.val_bpb for r in self.baselines if r.val_bpb is not None]
            if vals:
                return float(np.mean(vals))
        # Fallback: median of all successful rows without changes (single-experiment baselines)
        all_vals = [r.val_bpb for r in self.successful if r.val_bpb is not None]
        if all_vals:
            return float(np.median(all_vals))
        return 1.45  # default fallback

    # ── 1. Parameter Correlations ─────────────────────────────────────────

    def parameter_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Correlation between each change and val_bpb improvement.
        For each unique change, compute:
          - mean delta when present
          - mean delta when absent
          - correlation coefficient (point-biserial)
          - sample counts
        """
        results: Dict[str, Dict[str, float]] = {}
        change_vals: Dict[str, List[float]] = defaultdict(list)   # delta when change present
        no_change_vals: Dict[str, List[float]] = defaultdict(list)  # delta when change absent

        # Collect all unique changes
        all_changes: Set[str] = set()
        for r in self.successful:
            if r.delta is not None:
                for c in r.changes:
                    all_changes.add(c)
                # Rows without changes don't contribute individually
                if not r.changes:
                    for c in all_changes:
                        no_change_vals[c].append(r.delta)

        # If we only have multi-change experiments, use different approach
        for r in self.successful:
            if r.delta is None:
                continue
            present_changes = set(r.changes)
            for c in all_changes:
                if c in present_changes:
                    change_vals[c].append(r.delta)
                else:
                    no_change_vals[c].append(r.delta)

        for change in sorted(all_changes):
            has = np.array(change_vals.get(change, []))
            not_has = np.array(no_change_vals.get(change, []))

            if len(has) < 2 and len(not_has) < 2:
                continue

            mean_with = float(np.mean(has)) if len(has) > 0 else 0.0
            mean_without = float(np.mean(not_has)) if len(not_has) > 0 else 0.0
            std_with = float(np.std(has)) if len(has) > 0 else 0.0

            # Point-biserial correlation
            try:
                # Create binary variable
                all_deltas = list(has) + list(not_has)
                binary = [1] * len(has) + [0] * len(not_has)
                if len(set(binary)) < 2 or np.std(all_deltas) == 0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(binary, all_deltas)[0, 1])
                    if math.isnan(corr):
                        corr = 0.0
            except Exception:
                corr = 0.0

            success_count = sum(1 for d in has if d > 0)
            total = len(has)

            results[change] = {
                "mean_delta_present": round(mean_with, 6),
                "mean_delta_absent": round(mean_without, 6),
                "delta_diff": round(mean_with - mean_without, 6),
                "correlation": round(corr, 4),
                "n_present": len(has),
                "n_absent": len(not_has),
                "success_rate": round(success_count / total, 3) if total > 0 else 0.0,
                "std_present": round(std_with, 6),
            }

        # Sort by absolute delta_diff descending
        return dict(sorted(results.items(), key=lambda x: abs(x[1]["delta_diff"]), reverse=True))

    # ── 2. Best Individual Changes ────────────────────────────────────────

    def best_individual_changes(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank changes by their average delta improvement."""
        deltas: Dict[str, List[float]] = defaultdict(list)
        successes: Dict[str, int] = Counter()
        attempts: Dict[str, int] = Counter()

        for r in self.successful:
            if r.delta is None:
                continue
            for c in r.changes:
                deltas[c].append(r.delta)
                successes[c] += 1 if r.delta > 0 else 0
            if not r.changes:
                continue  # skip pure baselines
            for c in r.changes:
                attempts[c] += 1

        results = []
        for change, d_values in deltas.items():
            if not d_values:
                continue
            results.append({
                "change": change,
                "desc": CHANGE_CATALOG.get(change, {}).get("desc", change),
                "category": CHANGE_CATALOG.get(change, {}).get("category",
                           _guess_category(change)),
                "avg_delta": round(float(np.mean(d_values)), 6),
                "max_delta": round(float(np.max(d_values)), 6),
                "min_delta": round(float(np.min(d_values)), 6),
                "std": round(float(np.std(d_values)), 6),
                "n": len(d_values),
                "success_rate": round(successes.get(change, 0) / len(d_values), 3),
            })

        results.sort(key=lambda x: x["avg_delta"], reverse=True)
        return results[:top_n]

    # ── 3. Interaction Analysis ───────────────────────────────────────────

    def interaction_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze which changes work together (synergy) or conflict.
        For each pair of changes, compare their combined delta vs individual deltas.
        Returns: change_a -> change_b -> {synergy, n_combined, avg_individual, avg_combined}
        """
        # Track individual and combined deltas
        individual_deltas: Dict[str, List[float]] = defaultdict(list)
        pair_deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for r in self.successful:
            if r.delta is None:
                continue
            changes = r.changes
            # Record individual contributions
            for c in changes:
                individual_deltas[c].append(r.delta)
            # Record pairs
            for i in range(len(changes)):
                for j in range(i + 1, len(changes)):
                    pair = tuple(sorted([changes[i], changes[j]]))
                    pair_deltas[pair].append(r.delta)

        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (a, b), combined_list in sorted(pair_deltas.items()):
            avg_combined = float(np.mean(combined_list))
            avg_a = float(np.mean(individual_deltas.get(a, [0]))) if individual_deltas.get(a) else 0
            avg_b = float(np.mean(individual_deltas.get(b, [0]))) if individual_deltas.get(b) else 0
            # Expected additive effect
            expected = avg_a + avg_b
            synergy = avg_combined - expected if expected != 0 else 0.0
            # Interaction type
            if synergy > 0.005:
                interaction = "synergistic"
            elif synergy < -0.005:
                interaction = "conflicting"
            else:
                interaction = "additive"

            if a not in result:
                result[a] = {}
            if b not in result:
                result[b] = {}
            result[a][b] = {
                "synergy": round(synergy, 6),
                "avg_combined": round(avg_combined, 6),
                "avg_a": round(avg_a, 6),
                "avg_b": round(avg_b, 6),
                "n_combined": len(combined_list),
                "interaction": interaction,
            }
            result[b][a] = {
                "synergy": round(synergy, 6),
                "avg_combined": round(avg_combined, 6),
                "avg_a": round(avg_a, 6),
                "avg_b": round(avg_b, 6),
                "n_combined": len(combined_list),
                "interaction": interaction,
            }

        return result

    # ── 4. Trend Analysis ─────────────────────────────────────────────────

    def trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze whether improvements are diminishing over time.
        Splits experiments into temporal buckets and tracks best delta,
        success rate, and average delta per phase.
        """
        # Sort by timestamp
        sorted_rows = sorted(
            [r for r in self.successful if r.delta is not None],
            key=lambda x: x.timestamp
        )

        if not sorted_rows:
            return {"error": "No successful experiments with deltas"}

        # Phase-based analysis
        phase_data: Dict[int, List[float]] = defaultdict(list)
        for r in sorted_rows:
            phase_data[r.phase].append(r.delta)

        phase_stats = {}
        for phase in sorted(phase_data.keys()):
            vals = phase_data[phase]
            phase_stats[phase] = {
                "n": len(vals),
                "avg_delta": round(float(np.mean(vals)), 6),
                "max_delta": round(float(np.max(vals)), 6),
                "min_delta": round(float(np.min(vals)), 6),
                "success_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
            }

        # Diminishing returns: compare first half vs second half
        mid = len(sorted_rows) // 2
        first_half = [r.delta for r in sorted_rows[:mid]]
        second_half = [r.delta for r in sorted_rows[mid:]]

        trend = {
            "phase_stats": phase_stats,
            "timeline": {
                "first_half_avg": round(float(np.mean(first_half)), 6) if first_half else 0,
                "second_half_avg": round(float(np.mean(second_half)), 6) if second_half else 0,
                "first_half_max": round(float(np.max(first_half)), 6) if first_half else 0,
                "second_half_max": round(float(np.max(second_half)), 6) if second_half else 0,
            },
            "total_experiments": len(sorted_rows),
            "overall_success_rate": round(
                sum(1 for r in sorted_rows if r.delta > 0) / len(sorted_rows), 3
            ) if sorted_rows else 0,
        }

        # Detect diminishing returns
        if first_half and second_half:
            trend["diminishing_returns"] = (
                float(np.mean(second_half)) < float(np.mean(first_half))
            )
            trend["delta_degradation"] = round(
                float(np.mean(first_half)) - float(np.mean(second_half)), 6
            )

        return trend

    # ── 5. Success Rate by Category ───────────────────────────────────────

    def category_success_rates(self) -> Dict[str, Dict[str, Any]]:
        """Compute success rate per experiment category (LR, arch, opt, etc.)."""
        category_deltas: Dict[str, List[float]] = defaultdict(list)

        for r in self.successful:
            if r.delta is None:
                continue
            for c in r.changes:
                cat = CHANGE_CATALOG.get(c, {}).get("category", _guess_category(c))
                category_deltas[cat].append(r.delta)

        result = {}
        for cat in sorted(category_deltas.keys()):
            vals = category_deltas[cat]
            result[cat] = {
                "label": CATEGORY_LABELS.get(cat, cat),
                "n": len(vals),
                "avg_delta": round(float(np.mean(vals)), 6),
                "max_delta": round(float(np.max(vals)), 6),
                "success_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
                "pct_improving": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            }

        return result


# ─── Hypothesis Generator ─────────────────────────────────────────────────────

class HypothesisGenerator:
    """Generates research hypotheses from analysis patterns."""

    def __init__(self, analysis: AnalysisEngine):
        self.analysis = analysis
        self.rows = analysis.rows
        self.successful = analysis.successful

    def generate(self) -> List[Dict[str, Any]]:
        """Generate hypotheses from all available signal."""
        hypotheses: List[Dict[str, Any]] = []

        hypotheses.extend(self._hypotheses_single_changes())
        hypotheses.extend(self._hypotheses_conditionals())
        hypotheses.extend(self._hypotheses_interactions())
        hypotheses.extend(self._hypotheses_trends())
        hypotheses.extend(self._hypotheses_surprises())

        # Sort by confidence
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        return hypotheses

    def _hypotheses_single_changes(self) -> List[Dict[str, Any]]:
        """Generate hypotheses from individual change performance."""
        hypotheses = []
        correlations = self.analysis.parameter_correlations()

        for change, stats in correlations.items():
            n = stats["n_present"]
            if n < 2:
                continue

            success_rate = stats["success_rate"]
            mean_delta = stats["mean_delta_present"]
            correlation = stats["correlation"]
            desc = CHANGE_CATALOG.get(change, {}).get("desc", change)
            category = CHANGE_CATALOG.get(change, {}).get("category", _guess_category(change))

            # High success rate hypothesis
            if success_rate >= 0.75 and n >= 3:
                confidence = "high" if success_rate >= 0.8 and n >= 4 else "medium"
                hypotheses.append({
                    "type": "reliable_improver",
                    "confidence": confidence,
                    "hypothesis": (
                        f"{desc} is a reliable improvement ({stats['n_present']} attempts, "
                        f"{success_rate*100:.0f}% success rate, avg +{mean_delta:.4f} delta). "
                        f"Should be included in future configs."
                    ),
                    "change": change,
                    "category": category,
                    "evidence": {
                        "n": n,
                        "success_rate": success_rate,
                        "avg_delta": mean_delta,
                        "correlation": correlation,
                    }
                })

            # Consistent negative performer
            if success_rate <= 0.25 and n >= 3:
                hypotheses.append({
                    "type": "underperformer",
                    "confidence": "high" if success_rate <= 0.15 else "medium",
                    "hypothesis": (
                        f"{desc} consistently underperforms ({stats['n_present']} attempts, "
                        f"{success_rate*100:.0f}% success rate, avg {mean_delta:+.4f} delta). "
                        f"Consider abandoning or combining differently."
                    ),
                    "change": change,
                    "category": category,
                    "evidence": {
                        "n": n,
                        "success_rate": success_rate,
                        "avg_delta": mean_delta,
                    }
                })

            # Promising but under-tested
            if 0.5 <= success_rate < 0.75 and n >= 2 and n < 5:
                hypotheses.append({
                    "type": "promising_under_tested",
                    "confidence": "low",
                    "hypothesis": (
                        f"{desc} is promising but under-tested ({stats['n_present']} attempts, "
                        f"{success_rate*100:.0f}% success, avg +{mean_delta:.4f}). "
                        f"Warrants more trials to confirm."
                    ),
                    "change": change,
                    "category": category,
                    "evidence": {"n": n, "success_rate": success_rate, "avg_delta": mean_delta},
                })

        return hypotheses

    def _hypotheses_conditionals(self) -> List[Dict[str, Any]]:
        """Generate conditional hypotheses (X helps when Y is present)."""
        hypotheses = []

        # Build data structures for conditional analysis
        change_deltas: Dict[str, List[Tuple[float, Set[str]]]] = defaultdict(list)
        for r in self.successful:
            if r.delta is None:
                continue
            change_set = set(r.changes)
            for c in r.changes:
                change_deltas[c].append((r.delta, change_set - {c}))

        # Check if certain changes perform differently depending on what else is present
        for change, entries in change_deltas.items():
            if len(entries) < 3:
                continue

            # Group by presence of other changes
            for other, other_stats in change_deltas.items():
                if other == change or len(other_stats) < 2:
                    continue

                # Split entries for this change based on presence of 'other'
                with_other = [d for d, ctx in entries if other in ctx]
                without_other = [d for d, ctx in entries if other not in ctx]

                if len(with_other) < 2 and len(without_other) < 2:
                    continue

                if len(with_other) < 2 or len(without_other) < 2:
                    continue

                mean_with = float(np.mean(with_other))
                mean_without = float(np.mean(without_other))
                diff = mean_with - mean_without

                if abs(diff) > 0.003:  # Meaningful difference
                    cond_change = CHANGE_CATALOG.get(change, {}).get("desc", change)
                    other_change = CHANGE_CATALOG.get(other, {}).get("desc", other)

                    if diff > 0:
                        hypotheses.append({
                            "type": "conditional_positive",
                            "confidence": "medium" if min(len(with_other), len(without_other)) >= 3 else "low",
                            "hypothesis": (
                                f"{cond_change} performs better with {other_change} "
                                f"(+{diff:.4f} improvement, {len(with_other)} with vs "
                                f"{len(without_other)} without)."
                            ),
                            "change": change,
                            "condition": other,
                            "category": CHANGE_CATALOG.get(change, {}).get("category", _guess_category(change)),
                            "evidence": {
                                "n_with": len(with_other),
                                "n_without": len(without_other),
                                "mean_with": round(mean_with, 6),
                                "mean_without": round(mean_without, 6),
                                "diff": round(diff, 6),
                            }
                        })
                    else:
                        hypotheses.append({
                            "type": "conditional_negative",
                            "confidence": "medium" if min(len(with_other), len(without_other)) >= 3 else "low",
                            "hypothesis": (
                                f"{cond_change} performs worse with {other_change} "
                                f"({diff:+.4f}, {len(with_other)} with vs "
                                f"{len(without_other)} without). "
                                f"May be conflicting changes."
                            ),
                            "change": change,
                            "condition": other,
                            "category": CHANGE_CATALOG.get(change, {}).get("category", _guess_category(change)),
                            "evidence": {
                                "n_with": len(with_other),
                                "n_without": len(without_other),
                                "diff": round(diff, 6),
                            }
                        })

        return hypotheses

    def _hypotheses_interactions(self) -> List[Dict[str, Any]]:
        """Generate hypotheses from interaction analysis."""
        hypotheses = []
        matrix = self.analysis.interaction_matrix()

        for a, others in matrix.items():
            for b, stats in others.items():
                if a >= b:  # Avoid duplicates
                    continue
                n = stats["n_combined"]
                if n < 2:
                    continue

                synergy = stats["synergy"]
                a_desc = CHANGE_CATALOG.get(a, {}).get("desc", a)
                b_desc = CHANGE_CATALOG.get(b, {}).get("desc", b)

                if synergy > 0.005:
                    hypotheses.append({
                        "type": "synergy",
                        "confidence": "high" if synergy > 0.01 and n >= 3 else "medium",
                        "hypothesis": (
                            f"{a_desc} + {b_desc} shows synergistic interaction "
                            f"(synergy +{synergy:.4f}, {n} combined trials, "
                            f"avg +{stats['avg_combined']:.4f}). "
                            f"Better together than individually."
                        ),
                        "changes": [a, b],
                        "category": "interaction",
                        "evidence": stats,
                    })
                elif synergy < -0.005:
                    hypotheses.append({
                        "type": "conflict",
                        "confidence": "high" if synergy < -0.01 and n >= 3 else "medium",
                        "hypothesis": (
                            f"{a_desc} + {b_desc} shows negative interaction "
                            f"(conflict {synergy:.4f}, {n} combined trials). "
                            f"These changes may conflict and should not be combined."
                        ),
                        "changes": [a, b],
                        "category": "interaction",
                        "evidence": stats,
                    })

        return hypotheses

    def _hypotheses_trends(self) -> List[Dict[str, Any]]:
        """Generate hypotheses from trend analysis."""
        hypotheses = []
        trend = self.analysis.trend_analysis()

        if "error" in trend:
            return hypotheses

        timeline = trend.get("timeline", {})
        if timeline:
            first_avg = timeline.get("first_half_avg", 0)
            second_avg = timeline.get("second_half_avg", 0)

            if first_avg > 0 and second_avg > 0:
                if second_avg < first_avg * 0.7:
                    hypotheses.append({
                        "type": "diminishing_returns",
                        "confidence": "high",
                        "hypothesis": (
                            f"Diminishing returns detected: first half avg delta "
                            f"+{first_avg:.4f} dropped to +{second_avg:.4f} in second half. "
                            f"Easier improvements have been found; harder gains remain."
                        ),
                        "category": "meta",
                        "evidence": {"first_avg": first_avg, "second_avg": second_avg},
                    })
                elif second_avg > first_avg * 1.3:
                    hypotheses.append({
                        "type": "accelerating_progress",
                        "confidence": "medium",
                        "hypothesis": (
                            f"Progress is accelerating: first half avg delta "
                            f"+{first_avg:.4f} improved to +{second_avg:.4f} in second half. "
                            f"Recent experiment types are more productive."
                        ),
                        "category": "meta",
                        "evidence": {"first_avg": first_avg, "second_avg": second_avg},
                    })

        # Per-phase degradation
        phase_stats = trend.get("phase_stats", {})
        phases = sorted(phase_stats.keys())
        if len(phases) >= 3:
            phase_avgs = [phase_stats[p]["avg_delta"] for p in phases]
            if all(a > 0 for a in phase_avgs):
                diffs = [phase_avgs[i] - phase_avgs[i+1] for i in range(len(phase_avgs)-1)]
                if all(d > 0 for d in diffs):
                    hypotheses.append({
                        "type": "phase_degradation",
                        "confidence": "medium",
                        "hypothesis": (
                            f"Phase-by-phase degradation: avg delta dropped each phase "
                            f"({', '.join(f'P{p}: +{phase_avgs[i]:.4f}' for i, p in enumerate(phases))}). "
                            f"Low-hanging fruit exhausted in earlier phases."
                        ),
                        "category": "meta",
                        "evidence": {"phase_avgs": dict(zip(phases, phase_avgs))},
                    })

        return hypotheses

    def _hypotheses_surprises(self) -> List[Dict[str, Any]]:
        """Find surprising findings that contradict expectations."""
        hypotheses = []

        # Compare actual performance vs expected_improvement from catalog
        correlations = self.analysis.parameter_correlations()
        for change, stats in correlations.items():
            catalog = CHANGE_CATALOG.get(change, {})
            expected = None  # No hardcoded expected values in catalog format
            actual = stats["mean_delta_present"]

            # If the change has a known risk level that mismatches outcome
            risk = "medium"  # default
            if catalog:
                risk = "medium"  # Would need the full config

            # Surprise: change that was expected to fail succeeded
            if stats["correlation"] < -0.3 and stats["n_present"] >= 3:
                desc = CHANGE_CATALOG.get(change, {}).get("desc", change)
                # Check if this change should theoretically not work
                cat = CHANGE_CATALOG.get(change, {}).get("category", "")
                if cat:
                    hypotheses.append({
                        "type": "surprise",
                        "confidence": "medium",
                        "hypothesis": (
                            f"SURPRISE: {desc} has negative correlation ({stats['correlation']:.2f}) "
                            f"with improvement despite being a canonical technique. "
                            f"Avg delta: {actual:+.4f} across {stats['n_present']} trials. "
                            f"May be dataset-specific or needs different tuning."
                        ),
                        "change": change,
                        "category": "surprise",
                        "evidence": stats,
                    })

        # Check for successful "impossible" combinations
        for r in self.successful:
            if r.delta is not None and r.delta > 0 and len(r.changes) >= 3:
                # High-risk combinations that worked
                risky = [c for c in r.changes
                         if CHANGE_CATALOG.get(c, {}).get("category") == "arch"]
                if len(risky) >= 2:
                    risky_desc = [CHANGE_CATALOG.get(c, {}).get("desc", c) for c in risky]
                    hypotheses.append({
                        "type": "surprise",
                        "confidence": "low",
                        "hypothesis": (
                            f"SURPRISE: High-risk architectural combo worked: "
                            f"{'+'.join(risky_desc)} with delta +{r.delta:.4f} "
                            f"(experiment {r.experiment_name}). "
                            f"Contradicts assumption that architectural changes conflict."
                        ),
                        "changes": r.changes,
                        "category": "surprise",
                        "evidence": {"delta": r.delta, "n_changes": len(r.changes)},
                    })

        return hypotheses


# ─── Experiment Prioritizer ───────────────────────────────────────────────────

class ExperimentPrioritizer:
    """Generates prioritized next-experiment recommendations."""

    def __init__(
        self,
        analysis: AnalysisEngine,
        hypotheses: List[Dict[str, Any]],
        all_registered_changes: Optional[Set[str]] = None,
    ):
        self.analysis = analysis
        self.hypotheses = hypotheses
        # All changes that have been tried
        self.tried_changes: Set[str] = set()
        for r in analysis.rows:
            self.tried_changes.update(r.changes)

        # All changes that have been attempted as COMBINATIONS
        self.tried_combinations: Set[Tuple[str, ...]] = set()
        for r in analysis.rows:
            if len(r.changes) >= 2:
                self.tried_combinations.add(tuple(sorted(r.changes)))

        # Registered changes from orchestrator
        self.all_registered = all_registered_changes or set(CHANGE_CATALOG.keys())

    def get_next_experiments(self, n: int = 20) -> List[Dict[str, Any]]:
        """Generate prioritized list of next experiments."""
        experiments: List[Dict[str, Any]] = []

        # 1. High-confidence: follow proven patterns
        experiments.extend(self._high_confidence())

        # 2. Medium-confidence: extrapolate from partial patterns
        experiments.extend(self._medium_confidence())

        # 3. Low-confidence/high-reward: radical combinations
        experiments.extend(self._low_confidence_radical())

        # 4. Fill gaps: untested changes
        experiments.extend(self._fill_gaps())

        # Remove already-tried experiments and deduplicate
        experiments = self._deduplicate_and_filter(experiments, n)
        return experiments[:n]

    def _high_confidence(self) -> List[Dict[str, Any]]:
        """Experiments with strong precedent."""
        experiments = []
        correlations = self.analysis.parameter_correlations()

        # Find changes with high success rate
        for change, stats in correlations.items():
            if stats["n_present"] >= 3 and stats["success_rate"] >= 0.7 and stats["delta_diff"] > 0:
                desc = CHANGE_CATALOG.get(change, {}).get("desc", change)
                experiments.append({
                    "priority": "high",
                    "confidence": "high",
                    "name": f"reconfirm_{change}",
                    "changes": [change],
                    "rationale": (
                        f"Reconfirm {desc}: {stats['n_present']} prior attempts, "
                        f"{stats['success_rate']*100:.0f}% success, avg +{stats['mean_delta_present']:.4f}. "
                        f"Strong precedent suggests repeating with slightly different params."
                    ),
                    "evidence": stats,
                })

        # Top performing combination to re-run with validation
        interaction = self.analysis.interaction_matrix()
        for a, others in interaction.items():
            for b, stats in others.items():
                if a >= b:
                    continue
                if stats["interaction"] == "synergistic" and stats["n_combined"] >= 2:
                    experiments.append({
                        "priority": "high",
                        "confidence": "high",
                        "name": f"reconfirm_{a}_{b}",
                        "changes": [a, b],
                        "rationale": (
                            f"Reconfirm synergistic pair: {CHANGE_CATALOG.get(a, {}).get('desc', a)} + "
                            f"{CHANGE_CATALOG.get(b, {}).get('desc', b)}. "
                            f"Synergy +{stats['synergy']:.4f}, avg +{stats['avg_combined']:.4f}."
                        ),
                        "evidence": stats,
                    })

        return experiments

    def _medium_confidence(self) -> List[Dict[str, Any]]:
        """Extrapolations from partial patterns."""
        experiments = []
        best_changes = self.analysis.best_individual_changes()

        # Combine top individual performers that haven't been tried together
        top_individuals = [b["change"] for b in best_changes if b["avg_delta"] > 0 and b["n"] >= 2]

        for i, a in enumerate(top_individuals):
            for b in top_individuals[i+1:]:
                combo = tuple(sorted([a, b]))
                if combo in self.tried_combinations:
                    continue
                # Check if they conflict
                interaction = self.analysis.interaction_matrix()
                is_conflict = False
                if a in interaction and b in interaction.get(a, {}):
                    if interaction[a][b]["interaction"] == "conflicting":
                        is_conflict = True
                if is_conflict:
                    continue

                a_desc = CHANGE_CATALOG.get(a, {}).get("desc", a)
                b_desc = CHANGE_CATALOG.get(b, {}).get("desc", b)
                experiments.append({
                    "priority": "medium",
                    "confidence": "medium",
                    "name": f"extrapolate_{a}_{b}",
                    "changes": [a, b],
                    "rationale": (
                        f"Extrapolate: {a_desc} (avg +{self._get_avg(a):.4f}) + "
                        f"{b_desc} (avg +{self._get_avg(b):.4f}). "
                        f"Both individually effective; combination untested."
                    ),
                })

        # Conditional hypotheses: try change X under condition Y
        for h in self.hypotheses:
            if h["type"] == "conditional_positive" and h.get("condition"):
                change = h["change"]
                condition = h["condition"]
                combo = tuple(sorted([change, condition]))
                if combo not in self.tried_combinations:
                    experiments.append({
                        "priority": "medium",
                        "confidence": "medium",
                        "name": f"conditional_{change}_{condition}",
                        "changes": [change, condition],
                        "rationale": f"Test conditional hypothesis: {h['hypothesis']}",
                        "source_hypothesis": h["hypothesis"],
                    })

        return experiments

    def _low_confidence_radical(self) -> List[Dict[str, Any]]:
        """High-risk, high-reward experiments."""
        experiments = []
        best_changes = self.analysis.best_individual_changes()

        # Combine top performers with under-tested but promising changes
        correlations = self.analysis.parameter_correlations()
        untested_changes = [
            c for c in correlations
            if correlations[c]["n_present"] < 2 and correlations[c]["delta_diff"] > 0
        ]

        top_proven = [b["change"] for b in best_changes[:3] if b["avg_delta"] > 0]

        for proven in top_proven:
            for untested in untested_changes[:3]:
                combo = tuple(sorted([proven, untested]))
                if combo in self.tried_combinations:
                    continue
                proven_desc = CHANGE_CATALOG.get(proven, {}).get("desc", proven)
                untested_desc = CHANGE_CATALOG.get(untested, {}).get("desc", untested)
                experiments.append({
                    "priority": "low",
                    "confidence": "low",
                    "name": f"radical_{proven}_{untested}",
                    "changes": [proven, untested],
                    "rationale": (
                        f"High-reward: {proven_desc} + {untested_desc}. "
                        f"{proven_desc} is proven ({self._get_avg(proven):.4f} avg), "
                        f"{untested_desc} is untested ({correlations[untested]['n_present']} trials). "
                        f"Potential for large gains if they synergize."
                    ),
                })

        # Triple combinations of top performers
        if len(top_proven) >= 3:
            combo = tuple(sorted(top_proven[:3]))
            if combo not in self.tried_combinations:
                descs = [CHANGE_CATALOG.get(c, {}).get("desc", c) for c in combo]
                experiments.append({
                    "priority": "low",
                    "confidence": "low",
                    "name": f"triple_{'_'.join(top_proven[:3])}",
                    "changes": list(combo),
                    "rationale": (
                        f"Triple combination: {' + '.join(descs)}. "
                        f"All individually proven; combination could capture cumulative gains."
                    ),
                })

        return experiments

    def _fill_gaps(self) -> List[Dict[str, Any]]:
        """Fill in untested registered changes."""
        experiments = []
        untested = self.all_registered - self.tried_changes

        for change in sorted(untested):
            desc = CHANGE_CATALOG.get(change, {}).get("desc", change)
            experiments.append({
                "priority": "medium",
                "confidence": "medium",
                "name": f"untested_{change}",
                "changes": [change],
                "rationale": (
                    f"Untested: {desc} has never been tried. "
                    f"Should be evaluated to build complete picture."
                ),
            })

        return experiments

    def _deduplicate_and_filter(
        self, experiments: List[Dict[str, Any]], n: int
    ) -> List[Dict[str, Any]]:
        """Remove duplicates and already-tried combinations."""
        seen: Set[str] = set()
        unique = []

        for exp in experiments:
            key = "|".join(sorted(exp["changes"]))
            if key in seen:
                continue
            if key in ("",):
                continue
            # Skip if already tried (single changes that are already tried get lower priority)
            if tuple(sorted(exp["changes"])) in self.tried_combinations and len(exp["changes"]) >= 2:
                continue
            seen.add(key)
            unique.append(exp)

        return unique

    def _get_avg(self, change: str) -> float:
        """Get average delta for a change."""
        for r in self.successful:
            pass  # Placeholder - use correlations
        correlations = self.analysis.parameter_correlations()
        if change in correlations:
            return correlations[change]["mean_delta_present"]
        return 0.0

    @property
    def successful(self):
        return self.analysis.successful


# ─── Report Generator ─────────────────────────────────────────────────────────

class ReportGenerator:
    """Produces formatted markdown reports."""

    def __init__(
        self,
        analysis: AnalysisEngine,
        hypotheses: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        rows: List[ExperimentRow],
        output_path: Optional[str] = None,
    ):
        self.analysis = analysis
        self.hypotheses = hypotheses
        self.experiments = experiments
        self.rows = rows
        self.output_path = output_path

    def generate_full_report(self) -> str:
        """Generate comprehensive analysis report."""
        sections = [
            self._header(),
            self._summary_stats(),
            self._experiment_summary_table(),
            self._top_impactful_changes(),
            self._interaction_matrix_report(),
            self._category_analysis(),
            self._trend_analysis_report(),
            self._correlations_report(),
            self._hypotheses_report(),
            self._next_experiments_report(),
            self._surprises_report(),
        ]
        return "\n".join(sections)

    def generate_hypotheses_report(self) -> str:
        """Generate hypotheses-only report."""
        sections = [
            "# Hypotheses Report\n",
            f"Generated from {len(self.rows)} experiments "
            f"({len(self.analysis.successful)} successful)\n",
        ]
        sections.append(self._hypotheses_report())
        return "\n".join(sections)

    def generate_experiments_report(self, n: int) -> str:
        """Generate next-experiments report."""
        sections = [
            f"# Next {n} Experiment Recommendations\n",
            f"Based on analysis of {len(self.rows)} experiments\n",
        ]
        experiments = self.experiments[:n] if self.experiments else []
        sections.append(self._render_next_experiments(experiments))
        return "\n".join(sections)

    # ── Section Renderers ──────────────────────────────────────────────────

    def _header(self) -> str:
        return f"""# Meta-Analysis Report

**Experiments analyzed:** {len(self.rows)}
**Successful:** {len(self.analysis.successful)}
**Failed/Crashed:** {len(self.analysis.failed)}
**Baseline BPB:** {self.analysis.baseline_bpb:.4f}
**Best delta:** {max((r.delta or 0) for r in self.analysis.successful):.4f}
**Overall success rate:** {len([r for r in self.analysis.successful if r.delta and r.delta > 0]) / max(len(self.analysis.successful), 1) * 100:.1f}%
"""

    def _summary_stats(self) -> str:
        successful = self.analysis.successful
        improving = [r for r in successful if r.delta and r.delta > 0]

        return f"""## Summary Statistics

| Metric | Value |
|--------|-------|
| Total experiments | {len(self.rows)} |
| Successful | {len(successful)} |
| Failed/Crashed | {len(self.analysis.failed)} |
| Improving | {len(improving)} |
| Best delta | {max((r.delta or 0) for r in successful):.6f} |
| Worst delta | {min((r.delta or 0) for r in successful):.6f} |
| Median delta | {float(np.median([r.delta for r in successful if r.delta is not None])):.6f} |
| Mean delta | {float(np.mean([r.delta for r in successful if r.delta is not None])):.6f} |
| Baseline val_bpb | {self.analysis.baseline_bpb:.4f} |
"""

    def _experiment_summary_table(self) -> str:
        lines = ["## All Experiments\n"]
        lines.append("| # | ID | Name | val_bpb | delta | phase | status | changes |")
        lines.append("|---|----|------|---------|-------|-------|--------|---------|")

        for i, r in enumerate(self.rows, 1):
            val = f"{r.val_bpb:.4f}" if r.val_bpb is not None else "N/A"
            delta = f"{r.delta:+.4f}" if r.delta is not None else "N/A"
            changes = " | ".join(r.changes) if r.changes else "-"
            lines.append(f"| {i} | {r.experiment_id[:12]} | {r.experiment_name[:25]} | {val} | {delta} | {r.phase} | {r.status} | {changes} |")

        return "\n".join(lines)

    def _top_impactful_changes(self) -> str:
        best = self.analysis.best_individual_changes(5)
        lines = ["## Top 5 Most Impactful Changes\n"]

        if not best:
            lines.append("*No data available*\n")
            return "\n".join(lines)

        lines.append("| Rank | Change | Category | Avg Delta | Max Delta | N | Success Rate |")
        lines.append("|------|--------|----------|-----------|-----------|---|-------------|")
        for i, b in enumerate(best, 1):
            lines.append(
                f"| {i} | {b['desc']} | {b['category']} | "
                f"{b['avg_delta']:+.4f} | {b['max_delta']:+.4f} | "
                f"{b['n']} | {b['success_rate']*100:.0f}% |"
            )

        return "\n".join(lines)

    def _interaction_matrix_report(self) -> str:
        matrix = self.analysis.interaction_matrix()
        lines = ["## Interaction Matrix\n"]

        if not matrix:
            lines.append("*No interaction data (need multi-change experiments)*\n")
            return "\n".join(lines)

        # Find all significant interactions
        interactions = []
        seen = set()
        for a, others in matrix.items():
            for b, stats in others.items():
                pair = tuple(sorted([a, b]))
                if pair in seen:
                    continue
                seen.add(pair)
                a_desc = CHANGE_CATALOG.get(a, {}).get("desc", a)
                b_desc = CHANGE_CATALOG.get(b, {}).get("desc", b)
                interactions.append((a_desc, b_desc, stats))

        # Sort by absolute synergy
        interactions.sort(key=lambda x: abs(x[2]["synergy"]), reverse=True)

        lines.append("| Change A | Change B | Synergy | Combined Avg | N | Interaction |")
        lines.append("|----------|----------|---------|-------------|---|-------------|")
        for a_desc, b_desc, stats in interactions[:15]:
            lines.append(
                f"| {a_desc} | {b_desc} | {stats['synergy']:+.4f} | "
                f"{stats['avg_combined']:+.4f} | {stats['n_combined']} | "
                f"{stats['interaction']} |"
            )

        lines.append("")
        return "\n".join(lines)

    def _category_analysis(self) -> str:
        cats = self.analysis.category_success_rates()
        lines = ["## Success Rate by Category\n"]

        if not cats:
            lines.append("*No category data*\n")
            return "\n".join(lines)

        lines.append("| Category | N | Avg Delta | Max Delta | Success Rate |")
        lines.append("|----------|---|-----------|-----------|-------------|")
        for cat, stats in sorted(cats.items(), key=lambda x: x[1]["avg_delta"], reverse=True):
            lines.append(
                f"| {stats['label']} | {stats['n']} | "
                f"{stats['avg_delta']:+.4f} | {stats['max_delta']:+.4f} | "
                f"{stats['success_rate']*100:.1f}% |"
            )

        return "\n".join(lines)

    def _trend_analysis_report(self) -> str:
        trend = self.analysis.trend_analysis()
        lines = ["## Trend Analysis\n"]

        if "error" in trend:
            lines.append(f"*{trend['error']}*\n")
            return "\n".join(lines)

        dim = trend.get("diminishing_returns", False)
        degradation = trend.get("delta_degradation", 0)

        if dim:
            lines.append(
                f"**Diminishing returns detected!** Average delta degraded by "
                f"{degradation:.4f} between first and second half of experiments.\n"
            )
        elif trend.get("timeline", {}).get("second_half_avg", 0) > 0:
            lines.append("Progress is steady or accelerating - no diminishing returns detected.\n")

        # Phase stats
        phase_stats = trend.get("phase_stats", {})
        if phase_stats:
            lines.append("### Per-Phase Performance\n")
            lines.append("| Phase | N | Avg Delta | Max Delta | Success Rate |")
            lines.append("|-------|---|-----------|-----------|-------------|")
            for phase in sorted(phase_stats.keys()):
                s = phase_stats[phase]
                lines.append(
                    f"| {phase} | {s['n']} | {s['avg_delta']:+.4f} | "
                    f"{s['max_delta']:+.4f} | {s['success_rate']*100:.1f}% |"
                )

        return "\n".join(lines)

    def _correlations_report(self) -> str:
        correlations = self.analysis.parameter_correlations()
        lines = ["## Parameter Correlations\n"]

        if not correlations:
            lines.append("*No correlation data*\n")
            return "\n".join(lines)

        lines.append("| Change | Category | Correlation | Avg Delta (with) | Avg Delta (without) | Diff | N | Success Rate |")
        lines.append("|--------|----------|-------------|------------------|---------------------|------|---|-------------|")
        for change, stats in list(correlations.items())[:20]:
            desc = CHANGE_CATALOG.get(change, {}).get("desc", change)
            cat = CHANGE_CATALOG.get(change, {}).get("category", _guess_category(change))
            lines.append(
                f"| {desc} | {cat} | {stats['correlation']:+.3f} | "
                f"{stats['mean_delta_present']:+.4f} | "
                f"{stats['mean_delta_absent']:+.4f} | "
                f"{stats['delta_diff']:+.4f} | "
                f"{stats['n_present']} | {stats['success_rate']*100:.0f}% |"
            )

        return "\n".join(lines)

    def _hypotheses_report(self) -> str:
        lines = ["## Generated Hypotheses\n"]

        if not self.hypotheses:
            lines.append("*No hypotheses generated (need more data)*\n")
            return "\n".join(lines)

        # Group by type
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for h in self.hypotheses:
            by_type[h["type"]].append(h)

        type_labels = {
            "reliable_improver": "Reliable Improvers",
            "underperformer": "Underperformers",
            "promising_under_tested": "Promising but Under-Tested",
            "conditional_positive": "Conditional Improvements",
            "conditional_negative": "Conditional Conflicts",
            "synergy": "Synergistic Combinations",
            "conflict": "Conflicting Combinations",
            "diminishing_returns": "Trend Analysis",
            "surprise": "Surprise Findings",
            "phase_degradation": "Phase Analysis",
            "accelerating_progress": "Progress Trends",
        }

        for htype, label in type_labels.items():
            items = by_type.get(htype, [])
            if not items:
                continue
            lines.append(f"\n### {label}\n")
            for item in items:
                conf_badge = {
                    "high": "[HIGH]",
                    "medium": "[MED]",
                    "low": "[LOW]",
                }.get(item["confidence"], "[?]")
                lines.append(f"- {conf_badge} {item['hypothesis']}")

        return "\n".join(lines)

    def _next_experiments_report(self, experiments: Optional[List[Dict]] = None) -> str:
        experiments = experiments or self.experiments
        lines = []

        lines.append("## Next Experiment Recommendations\n")

        if not experiments:
            lines.append("*No recommendations available*\n")
            return "\n".join(lines)

        lines.append("| # | Priority | Confidence | Changes | Rationale |")
        lines.append("|---|----------|------------|---------|-----------|")
        for i, exp in enumerate(experiments[:20], 1):
            changes_str = ", ".join(
                CHANGE_CATALOG.get(c, {}).get("desc", c) for c in exp["changes"]
            )
            rationale = (exp.get("rationale", "")[:120] + "...") if len(exp.get("rationale", "")) > 120 else exp.get("rationale", "")
            lines.append(
                f"| {i} | {exp['priority']} | {exp['confidence']} | "
                f"{changes_str} | {rationale} |"
            )

        return "\n".join(lines)

    def _surprises_report(self) -> str:
        surprises = [h for h in self.hypotheses if h["type"] == "surprise"]
        lines = ["## Surprise Findings\n"]

        if not surprises:
            lines.append("*No surprising findings detected*\n")
            return "\n".join(lines)

        for s in surprises:
            lines.append(f"- {s['hypothesis']}")
        lines.append("")

        return "\n".join(lines)

    def write(self, content: str):
        """Write report to file or stdout."""
        if self.output_path:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                f.write(content)
        else:
            print(content)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _guess_category(change_name: str) -> str:
    """Guess category from change name prefix."""
    for prefix, cat in PREFIX_CATEGORY_MAP.items():
        if change_name.startswith(prefix):
            return cat
    return "unknown"


def _load_registered_changes(orchestrator_path: Optional[str] = None) -> Set[str]:
    """Try to extract registered change names from orchestrator source."""
    changes = set(CHANGE_CATALOG.keys())
    if orchestrator_path and os.path.exists(orchestrator_path):
        try:
            with open(orchestrator_path, "r") as f:
                content = f.read()
            # Find all _add_change calls
            pattern = r'self\._add_change\(["\']([\w_]+)["\']'
            matches = re.findall(pattern, content)
            changes.update(matches)
        except Exception:
            pass
    return changes


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Meta-Analysis Engine for autoresearch-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results", "-r",
        type=str,
        default="results/results.tsv",
        help="Path to results.tsv file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--hypotheses",
        action="store_true",
        help="Output hypotheses only",
    )
    parser.add_argument(
        "--next-experiments", "-n",
        type=int,
        default=0,
        help="Output top N next experiment recommendations",
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default=None,
        help="Path to research_orchestrator.py (for extracting registered changes)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format instead of markdown",
    )

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    rows = parse_tsv(args.results)
    if not rows:
        print("Error: No experiments found in results file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} experiments from {args.results}", file=sys.stderr)

    # Load registered changes
    orchestrator_path = args.orchestrator
    if not orchestrator_path:
        # Try default locations
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "research_orchestrator.py",
            script_dir.parent / "scripts" / "research_orchestrator.py",
            Path("scripts/research_orchestrator.py"),
        ]
        for c in candidates:
            if c.exists():
                orchestrator_path = str(c)
                break

    registered = _load_registered_changes(orchestrator_path)

    # Run analysis
    analysis = AnalysisEngine(rows)
    hypotheses_gen = HypothesisGenerator(analysis)
    hypotheses = hypotheses_gen.generate()

    prioritizer = ExperimentPrioritizer(analysis, hypotheses, registered)
    next_experiments = prioritizer.get_next_experiments(20)

    # Output
    if args.json:
        output = {
            "experiments_analyzed": len(rows),
            "successful": len(analysis.successful),
            "failed": len(analysis.failed),
            "baseline_bpb": analysis.baseline_bpb,
            "correlations": analysis.parameter_correlations(),
            "top_changes": analysis.best_individual_changes(10),
            "interactions": analysis.interaction_matrix(),
            "trends": analysis.trend_analysis(),
            "category_rates": analysis.category_success_rates(),
            "hypotheses": hypotheses,
            "next_experiments": next_experiments[:args.next_experiments] if args.next_experiments else next_experiments,
        }
        content = json.dumps(output, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(content)
        else:
            print(content)
        return

    if args.hypotheses:
        reporter = ReportGenerator(analysis, hypotheses, next_experiments, rows, args.output)
        content = reporter.generate_hypotheses_report()
        reporter.write(content)
        return

    if args.next_experiments > 0:
        reporter = ReportGenerator(analysis, hypotheses, next_experiments, rows, args.output)
        content = reporter.generate_experiments_report(args.next_experiments)
        reporter.write(content)
        return

    # Full report (default)
    reporter = ReportGenerator(analysis, hypotheses, next_experiments, rows, args.output)
    content = reporter.generate_full_report()
    reporter.write(content)


if __name__ == "__main__":
    main()
