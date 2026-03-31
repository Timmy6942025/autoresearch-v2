#!/usr/bin/env python3
"""
Knowledge Base for Autoresearch V2
===================================
Persistent, structured store of research learnings that survives across sessions.
Replaces the human researcher's accumulated experience with machine-readable knowledge.

Usage:
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase(path="/tmp/autoresearch-v2/knowledge.json")
    
    # Record experiment outcomes
    kb.record_result("exp_042", {"DEPTH": 8, "LR": 0.003}, 0.982, "confirmed", "Parallel ATTN+MLP")
    
    # Get suggestions
    best = kb.get_best_config()
    next_exp = kb.suggest_next_experiment()
    
    # Analyze
    trends = kb.analyze_trends()
    combos = kb.get_combination_suggestions()
    
    # Export report
    kb.export_markdown("/tmp/autoresearch-v2/reports/kb_summary.md")
"""

import json
import os
import re
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Default / empty knowledge base
# ---------------------------------------------------------------------------

_EMPTY_KB = {
    "improvements": [],
    "dead_ends": [],
    "configurations": {
        "base": {"params": {}, "val_bpb": None},
        "best": {"params": {}, "val_bpb": None, "diff": ""},
    },
    "experiments": [],          # flat log of every experiment run
    "meta": {
        "total_experiments": 0,
        "total_nights": 0,
        "best_improvement": 0.0,
        "current_branch": "unknown",
        "first_session": None,
        "last_session": None,
        "categories_success_rate": {},   # category -> {successes, attempts, rate}
    },
}


# ---------------------------------------------------------------------------
# Helpers used across methods
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _categorize_change(param_name: str) -> str:
    """Map a param key to a high-level category."""
    _MAP = {
        # learning rate / schedule
        "LR": "lr",
        "LR_WARMUP": "lr",
        "LR_COOL_DOWN": "lr",
        "LR_SCHEDULE": "lr",
        "WARMUP_STEPS": "lr",
        "WEIGHT_DECAY": "lr",
        "EPS": "lr",

        # architecture
        "DEPTH": "architecture",
        "DIM": "architecture",
        "ASPECT_RATIO": "architecture",
        "HEAD_DIM": "architecture",
        "GQA_KV_GROUPS": "architecture",
        "VOCAB_SIZE": "architecture",
        "MLP_RATIO": "architecture",
        "ACTIVATION": "architecture",
        "NORM_TYPE": "architecture",
        "PARALLEL_ATTN_MLP": "architecture",
        "ROPE_THETA": "architecture",
        "USE_QKNORM": "architecture",
        "USE_SWIGLU": "architecture",
        "SLIDING_WINDOW": "architecture",
        "ROPE_SCALING_FACTOR": "architecture",
        "FFN_EXPANSION_FACTOR": "architecture",

        # training dynamics
        "BATCH_SIZE": "training",
        "SEQ_LEN": "training",
        "GRAD_ACCUM": "training",
        "CLIP_GRAD_NORM": "training",
        "DROPOUT": "training",
        "LABEL_SMOOTHING": "training",
        "BETA_1": "training",
        "BETA_2": "training",

        # optimization
        "OPTIMIZER": "optimization",
        "FUSED": "optimization",
        "COMPILE": "optimization",
        "AMP": "optimization",
        "FUSED_CROSS_ENTROPY": "optimization",

        # tokenizer / data
        "TOKENIZER": "data",
        "DATA_MIX": "data",
    }
    return _MAP.get(param_name.upper(), "other")


def _diff_configs(a: dict, b: dict) -> list[str]:
    """Return list of param names that differ between two config dicts."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _is_improvement(val_bpb: float, baseline: float) -> bool:
    """Lower BPB is better.  A difference >= 0.001 is considered real."""
    return (baseline - val_bpb) >= 0.001


def _is_significant_surprise(val_bpb: float, baseline: float, expected_delta: float) -> bool:
    """An improvement much larger than expected = surprise."""
    actual = baseline - val_bpb
    return actual > (expected_delta + 0.010)  # >10m better than expected


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Persistent knowledge base for autoresearch experiment learnings."""

    def __init__(self, path: str = "knowledge.json"):
        self.path = Path(path)
        self._data = deepcopy(_EMPTY_KB)

        # Load existing KB if available
        self._loaded = False
        if self.path.exists():
            self._load()
            self._loaded = True
        else:
            # Seed with minimal structure so meta timestamps exist
            self._data["meta"]["first_session"] = _timestamp()
            self._data["meta"]["last_session"] = _timestamp()
            self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load KB from disk.  Backwards-compatible with older schemas."""
        raw = json.loads(self.path.read_text())
        self._data = deepcopy(_EMPTY_KB)
        self._data.update(raw)
        # Ensure new fields exist even in old KB files
        self._data.setdefault("experiments", [])
        self._data["meta"].setdefault("first_session", _timestamp())
        self._data["meta"].setdefault("categories_success_rate", {})
        self._data["meta"]["last_session"] = _timestamp()
        self.save()

    def save(self):
        """Persist current KB to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, default=str))

    # ------------------------------------------------------------------
    # Core API — record results
    # ------------------------------------------------------------------

    def record_result(
        self,
        experiment_id: str,
        config: dict,
        val_bpb: float,
        status: str = "confirmed",
        notes: str = "",
    ) -> dict:
        """
        Record an experiment outcome and update KB accordingly.

        Args:
            experiment_id: Unique experiment identifier.
            config: Parameter dict used for the experiment.
            val_bpb: Achieved validation BPB.
            status: One of 'confirmed', 'tentative', 'failed', 'cancelled'.
            notes: Free-form notes about the run.
        Returns:
            The experiment record as stored.
        """
        baseline_cfg = self._data["configurations"].get("base", {})
        baseline_bpb = (self._data["configurations"]["base"].get("val_bpb")
                        or baseline_cfg.get("val_bpb") or float("inf"))
        best_cfg = self._data["configurations"].get("best", {})
        best_bpb = (best_cfg.get("val_bpb")
                     or float("inf"))

        # Compute which params differ from the BASE (not the running best).
        # This isolates the actual tested change.
        base_params = baseline_cfg.get("params", {})
        changed_keys = _diff_configs(base_params, config)

        delta = baseline_bpb - val_bpb if baseline_bpb != float("inf") else 0.0
        best_delta = best_bpb - val_bpb if best_bpb != float("inf") else 0.0

        # Build experiment record
        record = {
            "id": experiment_id,
            "config": dict(config),
            "val_bpb": val_bpb,
            "delta": round(delta, 5),
            "improves_on_baseline": delta > 0.001,
            "improves_on_best": best_delta > 0.001,
            "surprise": False,
            "status": status,
            "notes": notes,
            "timestamp": _timestamp(),
        }

        # ---- Surprise detection ----
        # Compare against average improvement for the change type(s).
        for key in changed_keys:
            cat = _categorize_change(key)
            cat_info = self._data["meta"]["categories_success_rate"].get(cat, {})
            avg_improvement = cat_info.get("avg_improvement", 0)
            if delta > (avg_improvement + 0.010):
                record["surprise"] = True

        # ---- Store in experiment log ----
        self._data["experiments"].append(record)
        self._data["meta"]["total_experiments"] += 1

        # ---- Update configuration tracking ----
        if status not in ("failed", "cancelled"):
            # If this is the first experiment, set it as base
            if not self._data["configurations"]["base"]["params"]:
                self._data["configurations"]["base"] = {
                    "params": dict(config),
                    "val_bpb": val_bpb,
                }
            # If better than best, update best
            if best_bpb == float("inf") or val_bpb < best_bpb:
                diff_str = ", ".join(
                    _diff_configs(self._data["configurations"]["best"]["params"], config)
                ) if self._data["configurations"]["best"]["params"] else "initial_best"
                self._data["configurations"]["best"] = {
                    "params": dict(config),
                    "val_bpb": val_bpb,
                    "diff": diff_str,
                }
                self._data["meta"]["best_improvement"] = round(
                    baseline_bpb - val_bpb if baseline_bpb != float("inf") else 0.0, 4
                )
                best_delta = delta  # for improvement recording

            # ---- Improve or dead-end tracking ----
            if delta > 0.001:
                self._record_improvement(experiment_id, config, delta, status, notes, changed_keys)
            else:
                self._record_dead_end(experiment_id, config, delta, status, notes)

            # Update category success rates
            self._update_category_stats(changed_keys, delta > 0.001, delta)

        self.save()
        return record

    def _record_improvement(self, exp_id, config, delta, status, notes, changed_keys):
        """Add a new improvement entry or update an existing one."""
        # Check if we already have an improvement with a similar signature
        existing = None
        for imp in self._data["improvements"]:
            # Same experiment already recorded
            if exp_id in imp.get("source_experiments", []):
                existing = imp
                break

        if existing is None:
            new_id = f"imp_{len(self._data['improvements']) + 1:03d}"
            name = self._generate_improvement_name(config, changed_keys)
            entry = {
                "id": new_id,
                "name": name,
                "description": notes or f"Improved val_bpb by {delta:+.4f}",
                "val_bpb_delta": round(-delta, 4),   # negative delta = good
                "confirmed": status == "confirmed",
                "applied_to": ["base_config"],
                "changed_params": changed_keys,
                "interactions": [],
                "source_experiments": [exp_id],
                "first_seen": _timestamp(),
            }
            self._data["improvements"].append(entry)
        else:
            # Update counts if re-confirmed
            existing["confirmed"] = existing["confirmed"] or status == "confirmed"
            if exp_id not in existing["source_experiments"]:
                existing["source_experiments"].append(exp_id)

    def _record_dead_end(self, exp_id, config, delta, status, notes):
        """Add or update a dead-end entry."""
        name = self._generate_improvement_name(config, _diff_configs(
            self._data["configurations"]["best"].get("params", {}), config
        ))
        # Check if we've already tried this change
        for de in self._data["dead_ends"]:
            if de["name"] == name or de.get("status") == "discarded":
                if name in de.get("aliases", []):
                    de["attempts"] = de.get("attempts", 1) + 1
                    if delta > de.get("best_delta", float("-inf")):
                        de["best_delta"] = round(delta, 4)
                    if exp_id not in de.get("source_experiments", []):
                        de.setdefault("source_experiments", []).append(exp_id)
                    return
            # Match by changed params similarity >70%
            de_params = de.get("tested_params", [])
            cur_params = _diff_configs(
                self._data["configurations"]["best"].get("params", {}), config
            )
            if de_params and len(set(de_params) & set(cur_params)) / max(len(set(de_params) | set(cur_params)), 1) > 0.7:
                de["attempts"] = de.get("attempts", 1) + 1
                if exp_id not in de.get("source_experiments", []):
                    de.setdefault("source_experiments", []).append(exp_id)
                if delta > de.get("best_delta", float("-inf")):
                    de["best_delta"] = round(delta, 4)
                return

        new_id = f"dead_{len(self._data['dead_ends']) + 1:03d}"
        entry = {
            "id": new_id,
            "name": name,
            "attempts": 1,
            "best_delta": round(delta, 4),
            "status": "discarded" if status != "tentative" else "suspended",
            "reason": notes or f"No improvement (delta={delta:+.4f})",
            "tested_params": _diff_configs(
                self._data["configurations"]["best"].get("params", {}), config
            ),
            "source_experiments": [exp_id],
        }
        self._data["dead_ends"].append(entry)

    def _update_category_stats(self, changed_keys, is_improvement, delta):
        """Update per-category success/attempt counters."""
        if not changed_keys:
            return
        categories = set(_categorize_change(k) for k in changed_keys)
        for cat in categories:
            info = self._data["meta"]["categories_success_rate"].setdefault(cat, {
                "successes": 0,
                "attempts": 0,
                "improvements": [],
            })
            info["attempts"] += 1
            if is_improvement:
                info["successes"] += 1
            info["improvements"].append(round(delta, 5))
            info["avg_improvement"] = round(
                statistics.mean(info["improvements"]), 5
            )
            info["rate"] = round(
                info["successes"] / info["attempts"], 3
            )

    @staticmethod
    def _generate_improvement_name(config: dict, changed_keys: list[str]) -> str:
        """Generate a descriptive name from the config changes."""
        if changed_keys == ["PARALLEL_ATTN_MLP"]:
            return "Parallel Attention+MLP"
        if changed_keys == ["GQA_KV_GROUPS"]:
            return f"GQA (kv_groups={config.get('GQA_KV_GROUPS', '?')})"
        if changed_keys == ["ACTIVATION"]:
            act = config.get('ACTIVATION', config.get('USE_SWIGLU', '?'))
            return f"Activation={act}"
        if changed_keys == ["NORM_TYPE"]:
            return f"Norm -> {config.get('NORM_TYPE', '?')}"
        if changed_keys == ["USE_SWIGLU"]:
            return "SwiGLU activation"
        if changed_keys == ["USE_QKNORM"]:
            return "QK-Norm"
        if changed_keys == ["ROPE_SCALING_FACTOR"]:
            return f"RoPE scaling ({config.get('ROPE_SCALING_FACTOR', '?')}x)"
        if changed_keys == ["LR"]:
            return f"LR -> {config.get('LR', '?')}"
        if changed_keys == ["LR_SCHEDULE"]:
            return f"LR schedule -> {config.get('LR_SCHEDULE', '?')}"
        if not changed_keys:
            return "No changes detected"
        # Multi-param changes: use a short label
        short = []
        for k in sorted(changed_keys):
            v = config.get(k)
            if isinstance(v, bool):
                short.append(v)
            elif isinstance(v, (int, float)):
                short.append(f"{k}={v}")
            else:
                short.append(v)
        return f"Multi-change ({', '.join(str(s) for s in short[:3])})"

    # ------------------------------------------------------------------
    # Interaction tracking
    # ------------------------------------------------------------------

    def track_interaction(
        self,
        improvement_id: str,
        with_improvement_id: str,
        outcome: str,
        description: str = "",
    ):
        """
        Record an interaction between two improvements.
        outcome: 'synergistic', 'neutral', 'antagonistic'
        """
        for imp in self._data["improvements"]:
            if imp["id"] == improvement_id:
                desc = f"{outcome} with {with_improvement_id}"
                if description:
                    desc += f": {description}"
                imp.setdefault("interactions", []).append(desc)
                break
        self.save()

    def set_current_branch(self, branch: str):
        self._data["meta"]["current_branch"] = branch
        self.save()

    def increment_night(self, count: int = 1):
        self._data["meta"]["total_nights"] += count
        self._data["meta"]["last_session"] = _timestamp()
        self.save()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_best_config(self) -> Optional[dict]:
        """Return the best-known configuration with its metrics."""
        best = self._data["configurations"].get("best", {})
        if not best.get("params"):
            return None
        return {
            "params": dict(best["params"]),
            "val_bpb": best["val_bpb"],
            "diff": best.get("diff", ""),
        }

    def get_base_config(self) -> Optional[dict]:
        """Return the baseline configuration."""
        base = self._data["configurations"].get("base", {})
        if not base.get("params"):
            return None
        return {"params": dict(base["params"]), "val_bpb": base["val_bpb"]}

    def suggest_next_experiment(self, num_suggestions: int = 5) -> list[dict]:
        """
        Suggest promising next experiments based on:
        1. Unconfirmed improvements (need re-testing)
        2. Untried combinations of confirmed improvements
        3. Under-explored categories with decent success rates
        4. Escalation (try a slightly larger version of a minor improvement)
        """
        suggestions = []
        best_cfg = self.get_best_config()
        base_params = best_cfg["params"] if best_cfg else {}

        # Strategy 1: Re-test unconfirmed improvements
        for imp in self._data["improvements"]:
            if not imp.get("confirmed"):
                suggestions.append({
                    "strategy": "confirm_improvement",
                    "name": imp["name"],
                    "expected_delta": imp["val_bpb_delta"],
                    "source_id": imp["id"],
                    "priority": "high",
                    "reason": "Unconfirmed improvement — needs verification",
                })

        # Strategy 2: Combinations of confirmed, non-interfering improvements
        combos = self.get_combination_suggestions(max_combos=3)
        for combo in combos:
            suggestions.append({
                "strategy": "combination",
                "name": f"Combine: {combo['name']}",
                "params": combo["params"],
                "expected_delta": combo["expected_delta"],
                "confidence": combo["confidence"],
                "reason": combo["reason"],
                "priority": "high" if combo["confidence"] > 0.7 else "medium",
            })

        # Strategy 3: Escalate minor improvements
        for imp in self._data["improvements"]:
            if imp.get("confirmed") and -0.010 <= imp["val_bpb_delta"] <= -0.003:
                suggestions.append({
                    "strategy": "escalate",
                    "name": f"Stronger version of {imp['name']}",
                    "base_improvement": imp["id"],
                    "expected_delta": round(imp["val_bpb_delta"] * 1.5, 4),
                    "reason": "Small confirmed improvement — try amplified version",
                    "priority": "medium",
                })

        # Strategy 4: Explore categories with good success rates but few attempts
        cat_stats = self._data["meta"]["categories_success_rate"]
        for cat, info in cat_stats.items():
            rate = info.get("rate", 0)
            attempts = info.get("attempts", 0)
            if rate >= 0.5 and attempts <= 3:
                suggestions.append({
                    "strategy": "explore_category",
                    "category": cat,
                    "success_rate": rate,
                    "attempts_so_far": attempts,
                    "expected_delta": -info.get("avg_improvement", 0),
                    "reason": f"{cat} has {rate:.0%} success rate with only {attempts} attempts",
                    "priority": "medium",
                })

        # Strategy 5: If no improvements exist yet, try a grid over high-value changes
        if not self._data["improvements"]:
            suggestions.extend(self._cold_start_suggestions(base_params))

        # Deduplicate and rank
        seen_names = set()
        unique = []
        for s in sorted(suggestions, key=lambda s: {"high": 0, "medium": 1, "low": 2}.get(s.get("priority", "low"), 3)):
            name = s.get("name", str(s))
            if name not in seen_names:
                seen_names.add(name)
                unique.append(s)

        return unique[:num_suggestions]

    def _cold_start_suggestions(self, base_params: dict) -> list[dict]:
        """When we have no improvements yet, suggest a sensible exploration order."""
        return [
            {
                "strategy": "cold_start",
                "name": "Try parallel attention+MLP",
                "params": {**base_params, "PARALLEL_ATTN_MLP": True},
                "expected_delta": -0.010,
                "reason": "Low-risk architectural change, commonly beneficial",
                "priority": "high",
            },
            {
                "strategy": "cold_start",
                "name": "Try GQA (kv_groups=2)",
                "params": {**base_params, "GQA_KV_GROUPS": 2},
                "expected_delta": -0.005,
                "reason": "Reduces KV cache, often improves efficiency",
                "priority": "high",
            },
            {
                "strategy": "cold_start",
                "name": "Try LR sweep (0.5x, 1.5x baseline)",
                "params": {**base_params},
                "expected_delta": -0.008,
                "reason": "Learning rate rarely optimal on first try",
                "priority": "medium",
            },
            {
                "strategy": "cold_start",
                "name": "Try SwiGLU activation",
                "params": {**base_params, "USE_SWIGLU": True},
                "expected_delta": -0.006,
                "reason": "SwiGLU consistently outperforms standard activations",
                "priority": "medium",
            },
        ]

    def analyze_trends(self) -> dict:
        """
        Analyze experiment history for patterns.
        Returns dict with:
          - category_rankings: which param categories yield most improvements
          - diminishing_returns: categories where success rate is declining
          - interaction_summary: known synergies and conflicts
          - surprising_finds: experiments that beat expectations
          - overall_trajectory: are we still improving?
        """
        result = {}

        # --- Category rankings ---
        cat_stats = self._data["meta"]["categories_success_rate"]
        rankings = sorted(
            [(cat, info) for cat, info in cat_stats.items()],
            key=lambda x: (x[1].get("rate", 0), x[1].get("avg_improvement", 0)),
            reverse=True,
        )
        result["category_rankings"] = [
            {
                "category": cat,
                "success_rate": info.get("rate", 0),
                "avg_improvement": info.get("avg_improvement", 0),
                "attempts": info.get("attempts", 0),
                "improvements": info.get("successes", 0),
            }
            for cat, info in rankings
        ]

        # --- Diminishing returns detection ---
        diminishing = []
        for cat, info in cat_stats.items():
            improvements = info.get("improvements", [])
            if len(improvements) >= 4:
                first_half = improvements[:len(improvements) // 2]
                second_half = improvements[len(improvements) // 2:]
                avg_first = statistics.mean(first_half) if first_half else 0
                avg_second = statistics.mean(second_half) if second_half else 0
                if avg_second < avg_first * 0.5:
                    diminishing.append({
                        "category": cat,
                        "early_avg": round(avg_first, 5),
                        "late_avg": round(avg_second, 5),
                        "decline_pct": round(100 * (1 - avg_second / max(avg_first, 1e-9)), 1),
                    })
        result["diminishing_returns"] = diminishing

        # --- Interaction summary ---
        interactions = []
        for imp in self._data["improvements"]:
            for interaction in imp.get("interactions", []):
                interactions.append({
                    "improvement": imp["name"],
                    "interaction": interaction,
                })
        result["interaction_summary"] = interactions

        # --- Surprising findings ---
        surprises = [
            {
                "experiment": exp["id"],
                "val_bpb": exp["val_bpb"],
                "delta": exp["delta"],
                "config_keys": list(exp["config"].keys()),
                "notes": exp.get("notes", ""),
            }
            for exp in self._data["experiments"]
            if exp.get("surprise")
        ]
        result["surprising_finds"] = surprises

        # --- Overall trajectory ---
        exps = self._data["experiments"]
        if len(exps) >= 3:
            recent = exps[-5:]
            improvements_count = sum(1 for e in recent if e.get("improves_on_baseline"))
            result["overall_trajectory"] = {
                "last_5_experiments": {
                    "improvements": improvements_count,
                    "stagnations": 5 - improvements_count,
                },
                "trend": ("improving" if improvements_count >= 3
                          else "declining" if improvements_count == 0
                          else "plateauing"),
            }
        else:
            result["overall_trajectory"] = {
                "trend": "insufficient_data",
                "experiments_so_far": len(exps),
            }

        return result

    def get_combination_suggestions(
        self,
        max_combos: int = 5,
        max_params: int = 3,
    ) -> list[dict]:
        """
        Suggest combinations of improvements to try together.
        Uses interaction history to avoid antagonistic pairings.
        """
        confirmed = [imp for imp in self._data["improvements"] if imp.get("confirmed")]
        if len(confirmed) < 2:
            return []

        best_params = self._data["configurations"]["best"].get("params", {})
        # Build set of known antagonistic pairs
        antagonistic = set()
        for imp in confirmed:
            for interaction in imp.get("interactions", []):
                if "conflict" in interaction.lower() or "antagonistic" in interaction.lower():
                    # Parse: "antagonistic with imp_002"
                    match = re.search(r"(imp_\d+)", interaction)
                    if match:
                        antagonistic.add((imp["id"], match.group(1)))

        # Synergistic bonuses
        synergy_bonus = {}
        for imp in confirmed:
            for interaction in imp.get("interactions", []):
                if "synergistic" in interaction.lower() or "works well" in interaction.lower():
                    match = re.search(r"(imp_\d+)", interaction)
                    if match:
                        pair = tuple(sorted([imp["id"], match.group(1)]))
                        synergy_bonus[pair] = synergy_bonus.get(pair, 0) + 0.003

        # Generate combinations
        from itertools import combinations as itertools_combinations
        suggestions = []

        for size in range(2, min(len(confirmed) + 1, max_params + 1)):
            for combo in itertools_combinations(confirmed, size):
                ids = [c["id"] for c in combo]

                # Check no antagonistic pairs
                has_conflict = False
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        if (ids[i], ids[j]) in antagonistic or (ids[j], ids[i]) in antagonistic:
                            has_conflict = True
                if has_conflict:
                    continue

                # Merge params, checking for actual differences
                merged = dict(best_params)
                for imp in combo:
                    # Find source experiment config
                    for exp in self._data["experiments"]:
                        if exp["id"] in imp.get("source_experiments", []):
                            merged.update(exp["config"])
                            break

                # Calc expected delta with synergy bonus
                total_delta = sum(c["val_bpb_delta"] for c in combo)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pair_key = tuple(sorted([ids[i], ids[j]]))
                        total_delta -= synergy_bonus.get(pair_key, 0)

                # Confidence score
                confidence = min(len(combo) * 0.25 + 0.4, 0.95)
                if any(sid in synergy_bonus for sid in itertools_combinations(ids, 2)):
                    confidence += 0.1
                confidence = min(confidence, 0.95)

                suggestions.append({
                    "improvements": [{"id": c["id"], "name": c["name"], "delta": c["val_bpb_delta"]}
                                     for c in combo],
                    "params": merged,
                    "expected_delta": round(total_delta, 4),
                    "confidence": round(confidence, 2),
                    "name": " + ".join(c["name"] for c in combo),
                    "reason": f"{'Synergistic pairing' if synergy_bonus else 'Independent improvements'} — "
                              f"expected combined effect: {total_delta:+.4f}",
                })

        return sorted(suggestions, key=lambda s: (-s["confidence"], s["expected_delta"]))[:max_combos]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_markdown(self, output_path: str = "kb_report.md") -> str:
        """Generate a comprehensive markdown summary of all learnings."""
        best = self.get_best_config()
        base = self.get_base_config()
        trends = self.analyze_trends()
        meta = self._data["meta"]
        improvements = self._data["improvements"]
        dead_ends = self._data["dead_ends"]
        experiments = self._data["experiments"]

        lines = []
        def ln(s: str = ""):
            lines.append(s)

        ln(f"# Autoresearch V2 — Knowledge Base Report")
        ln(f"Generated: {_timestamp()}")
        ln()
        ln("---")
        ln()

        # Meta summary
        ln("## Executive Summary")
        ln(f"- **Total experiments**: {meta['total_experiments']}")
        ln(f"- **Total nights of research**: {meta['total_nights']}")
        ln(f"- **Best improvement over baseline**: {meta['best_improvement']:+.4f} BPB")
        ln(f"- **Confirmed improvements**: {sum(1 for i in improvements if i.get('confirmed'))}")
        ln(f"- **Discarded dead-ends**: {sum(1 for d in dead_ends if d.get('status') == 'discarded')}")
        ln(f"- **Current branch**: {meta.get('current_branch', 'unknown')}")
        if best:
            ln(f"- **Best val_bpb achieved**: {best['val_bpb']:.4f}")
        if base:
            ln(f"- **Base val_bpb**: {base['val_bpb']:.4f}")
        ln()

        # Best configuration
        ln("## Best Configuration")
        if best:
            ln(f"```json")
            ln(json.dumps(best, indent=2, default=str))
            ln("```")
        else:
            ln("*No configuration recorded yet.*")
        ln()

        # Improvements
        ln("## Confirmed Improvements")
        if improvements:
            for imp in sorted(improvements, key=lambda x: x.get("val_bpb_delta", 0)):
                conf = "✅" if imp.get("confirmed") else "⏳"
                ln(f"### {conf} {imp['name']} (`{imp['id']}`)")
                ln(f"- **Delta**: {imp['val_bpb_delta']:+.4f} BPB")
                ln(f"- **Description**: {imp.get('description', 'N/A')}")
                if imp.get("changed_params"):
                    ln(f"- **Changed params**: {', '.join(imp['changed_params'])}")
                if imp.get("interactions"):
                    ln(f"- **Interactions**:")
                    for i in imp["interactions"]:
                        ln(f"  - {i}")
                if imp.get("source_experiments"):
                    ln(f"- **Source experiments**: {', '.join(imp['source_experiments'])}")
                ln()
        else:
            ln("*No improvements discovered yet.*")
            ln()

        # Dead ends
        ln("## Dead Ends")
        if dead_ends:
            for de in dead_ends:
                discard = "❌" if de.get("status") == "discarded" else "⏸️"
                ln(f"### {discard} {de['name']} (`{de['id']}`)")
                ln(f"- **Attempts**: {de.get('attempts', 1)}")
                ln(f"- **Best delta**: {de.get('best_delta', 0):+.4f} BPB")
                ln(f"- **Status**: {de.get('status', 'unknown')}")
                ln(f"- **Reason**: {de.get('reason', 'N/A')}")
                if de.get("tested_params"):
                    ln(f"- **Tested params**: {', '.join(de['tested_params'])}")
                ln()
        else:
            ln("*No dead ends recorded.*")
            ln()

        # Category rankings
        ln("## Parameter Category Rankings")
        if trends.get("category_rankings"):
            ln("| Category | Success Rate | Avg Improvement | Attempts |")
            ln("|----------|-------------|-----------------|----------|")
            for cat in trends["category_rankings"]:
                ln(f"| {cat['category']} | {cat['success_rate']:.0%} | "
                   f"{cat['avg_improvement']:+.4f} | {cat['attempts']} |")
            ln()
        else:
            ln("*Insufficient data for category rankings.*")
            ln()

        # Diminishing returns
        if trends.get("diminishing_returns"):
            ln("## ⚠️  Diminishing Returns Detected")
            for dr in trends["diminishing_returns"]:
                ln(f"- **{dr['category']}**: avg {dr['early_avg']:+.4f} → "
                   f"{dr['late_avg']:+.4f} ({dr['decline_pct']:.0f}% decline)")
            ln()

        # Surprises
        if trends.get("surprising_finds"):
            ln("## 🎉 Surprising Findings")
            for s in trends["surprising_finds"]:
                ln(f"- **{s['experiment']}**: delta {s['delta']:+.4f} BPB"
                   f" — {s.get('notes', '')}")
            ln()

        # Interaction summary
        if trends.get("interaction_summary"):
            ln("## Improvement Interactions")
            for ix in trends["interaction_summary"]:
                ln(f"- **{ix['improvement']}**: {ix['interaction']}")
            ln()

        # Overall trajectory
        if trends.get("overall_trajectory"):
            ln("## Overall Trajectory")
            traj = trends["overall_trajectory"]
            ln(f"- **Trend**: **{traj.get('trend', 'unknown')}**")
            if "last_5_experiments" in traj:
                ln(f"- **Last 5 experiments**: "
                   f"{traj['last_5_experiments']['improvements']} improvements, "
                   f"{traj['last_5_experiments']['stagnations']} stagnations")
            ln()

        # Combination suggestions
        combos = self.get_combination_suggestions()
        if combos:
            ln("## Suggested Improvement Combinations")
            for c in combos:
                ln(f"### {c['name']}")
                ln(f"- **Expected delta**: {c['expected_delta']:+.4f} BPB")
                ln(f"- **Confidence**: {c['confidence']:.0%}")
                ln(f"- **Reason**: {c['reason']}")
            ln()

        # Next experiments
        next_exp = self.suggest_next_experiment(num_suggestions=5)
        if next_exp:
            ln("## Suggested Next Experiments")
            for i, s in enumerate(next_exp, 1):
                prio = s.get("priority", "medium").upper()
                ln(f"{i}. [{prio}] **{s['name']}**")
                ln(f"   - Strategy: {s['strategy']}")
                if "expected_delta" in s:
                    ln(f"   - Expected delta: {s['expected_delta']:+.4f} BPB")
                ln(f"   - Reason: {s.get('reason', 'N/A')}")
            ln()

        # Experiment history
        ln("---")
        ln("## Full Experiment History")
        if experiments:
            ln("| ID | val_bpb | Δ | Status | Notes |")
            ln("|----|---------|---|--------|-------|")
            for exp in experiments:
                ln(f"| {exp['id']} | {exp['val_bpb']:.4f} | "
                   f"{exp.get('delta', 0):+.4f} | "
                   f"{exp['status']} | "
                   f"{exp.get('notes', '')} |")
        else:
            ln("*No experiments recorded.*")
        ln()

        ln("---")
        ln(f"*Report generated by KnowledgeBase at {_timestamp()}*")

        content = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content)
        return content

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a concise KB summary dict (useful for agent prompts)."""
        best = self.get_best_config()
        meta = self._data["meta"]
        return {
            "total_experiments": meta["total_experiments"],
            "total_nights": meta["total_nights"],
            "best_val_bpb": best["val_bpb"] if best else None,
            "best_improvement": meta["best_improvement"],
            "confirmed_improvements": len([i for i in self._data["improvements"] if i.get("confirmed")]),
            "discarded_dead_ends": len([d for d in self._data["dead_ends"] if d.get("status") == "discarded"]),
            "top_category": (
                trends["category_rankings"][0]["category"]
                if (trends := self.analyze_trends()).get("category_rankings")
                else "none"
            ),
            "trajectory": self.analyze_trends().get("overall_trajectory", {}).get("trend", "unknown"),
        }


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base for Autoresearch V2")
    parser.add_argument("--path", default="knowledge.json", help="Path to KB JSON file")
    parser.add_argument("--export", help="Export markdown report to this path")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--best", action="store_true", help="Print best config")
    parser.add_argument("--suggest", type=int, default=0, help="Print N suggestions")
    parser.add_argument("--trends", action="store_true", help="Print trend analysis")
    parser.add_argument("--combos", action="store_true", help="Print combination suggestions")
    parser.add_argument("--record", metavar="JSON", help="Record experiment from JSON string")
    args = parser.parse_args()

    kb = KnowledgeBase(path=args.path)

    if args.record:
        data = json.loads(args.record)
        record = kb.record_result(
            experiment_id=data.get("id", "exp_auto"),
            config=data.get("config", {}),
            val_bpb=data["val_bpb"],
            status=data.get("status", "confirmed"),
            notes=data.get("notes", ""),
        )
        print(json.dumps(record, indent=2))
    elif args.export:
        kb.export_markdown(args.export)
        print(f"Report exported to {args.export}")
    elif args.summary:
        print(json.dumps(kb.summary(), indent=2))
    elif args.best:
        print(json.dumps(kb.get_best_config(), indent=2))
    elif args.suggest:
        print(json.dumps(kb.suggest_next_experiment(args.suggest), indent=2))
    elif args.trends:
        print(json.dumps(kb.analyze_trends(), indent=2))
    elif args.combos:
        print(json.dumps(kb.get_combination_suggestions(), indent=2))
    else:
        parser.print_help()
