#!/usr/bin/env python3
"""
Experiment Designer - Systematic experimental design autoresearch-v2
Generates, manages, and optimizes experiment plans across multiple categories (Architecture, Optimization, Training, Regularization, Novel).
"""

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set


# --- Property Assignment Heuristics ---
def compute_properties(
    category: str,
    param: str,
    baseline: Any,
    target: Any,
    is_combo: bool = False,
) -> Dict[str, Any]:
    """
    Compute estimated_impact, risk, effort, category, dependencies, compatibility
    for a given change.
    """
    impact_scores = {
        "architecture": {
            "DEPTH": 0.65,
            "ASPECT_RATIO": 0.45,
            "HEAD_DIM": 0.40,
            "WINDOW_PATTERN": 0.55,
            "BLOCK_TYPE": 0.60,
            "MLP_TYPE": 0.45,
            "N_KV_HEAD": 0.40,
        },
        "optimization": {
            "LEARNING_RATE": 0.75,
            "WARMUP_PCT": 0.35,
            "WARM_DOWN_PCT": 0.30,
            "WEIGHT_DECAY": 0.45,
            "MOMENTUM": 0.40,
        },
        "training": {
            "BATCH_SIZE": 0.50,
            "GRAD_CLIP": 0.25,
            "SEQ_LEN": 0.55,
        },
        "regularization": {
            "DROPOUT": 0.30,
            "STOCHASTIC_DEPTH": 0.35,
            "LABEL_SMOOTHING": 0.20,
        },
        "novel": {
            "ADAPTIVE_SOFTCAP": 0.50,
            "RESIDUAL_LOGIT_BIAS": 0.45,
            "DYNAMIC_WINDOW": 0.55,
        },
    }

    risk_scores = {
        "architecture": {
            "DEPTH": 0.30,
            "ASPECT_RATIO": 0.20,
            "HEAD_DIM": 0.20,
            "WINDOW_PATTERN": 0.30,
            "BLOCK_TYPE": 0.30,
            "MLP_TYPE": 0.20,
            "N_KV_HEAD": 0.15,
        },
        "optimization": {
            "LEARNING_RATE": 0.40,
            "WARMUP_PCT": 0.15,
            "WARM_DOWN_PCT": 0.15,
            "WEIGHT_DECAY": 0.20,
            "MOMENTUM": 0.15,
        },
        "training": {
            "BATCH_SIZE": 0.25,
            "GRAD_CLIP": 0.10,
            "SEQ_LEN": 0.20,
        },
        "regularization": {
            "DROPOUT": 0.15,
            "STOCHASTIC_DEPTH": 0.15,
            "LABEL_SMOOTHING": 0.10,
        },
        "novel": {
            "ADAPTIVE_SOFTCAP": 0.40,
            "RESIDUAL_LOGIT_BIAS": 0.35,
            "DYNAMIC_WINDOW": 0.40,
        },
    }

    # Dependencies between parameters
    deps_map: Dict[str, List[str]] = {
        "DEPTH": ["ASPECT_RATIO", "HEAD_DIM"],
        "BLOCK_TYPE": ["DEPTH"],
        "MLP_TYPE": [],
        "WINDOW_PATTERN": ["DEPTH"],
        "N_KV_HEAD": ["HEAD_DIM"],
        "LEARNING_RATE": ["BATCH_SIZE"],
        "WARMUP_PCT": ["LEARNING_RATE"],
        "WARM_DOWN_PCT": ["LEARNING_RATE"],
        "MOMENTUM": ["LEARNING_RATE"],
        "WEIGHT_DECAY": [],
        "BATCH_SIZE": [],
        "GRAD_CLIP": [],
        "SEQ_LEN": ["BATCH_SIZE"],
        "DROPOUT": [],
        "STOCHASTIC_DEPTH": ["DEPTH"],
        "LABEL_SMOOTHING": [],
        "ADAPTIVE_SOFTCAP": ["LEARNING_RATE"],
        "RESIDUAL_LOGIT_BIAS": [],
        "DYNAMIC_WINDOW": ["WINDOW_PATTERN"],
    }

    cat = category.lower()
    p = impact_scores.get(cat, {}).get(param, 0.3)
    r = risk_scores.get(cat, {}).get(param, 0.2)
    
    # Combo boosts for risk/effort but also potential impact
    combo_boost = 0.15 if is_combo else 0.0
    effort = 2 if not is_combo else 3
    if "DEPTH" in param or "SEQ_LEN" in param:
        effort += 1
    
    impact = min(1.0, p + combo_boost)
    risk = min(1.0, r + combo_boost * 0.5)

    # Priority: impact / (risk * effort)
    priority = int(max(1, min(10, (impact / max(0.05, risk * effort)) * 3)))

    deps = deps_map.get(param, [])
    compatibility = ["baseline"] + [d for d in deps if d in deps_map]

    return {
        "estimated_impact": round(impact, 2),
        "risk": round(risk, 2),
        "effort": effort,
        "category": cat,
        "dependencies": deps,
        "compatibility": compatibility,
        "priority": priority,
    }


# --- Parameter Space Enumeration ---
class ParameterSpace:
    """Defines possible ranges and values for each experimental parameter."""
    
    @staticmethod
    def architecture() -> Dict[str, List[Any]]:
        return {
            "DEPTH": list(range(4, 17)),
            "ASPECT_RATIO": list(range(32, 129, 8)),
            "HEAD_DIM": [64, 96, 128, 192, 256],
            "N_KV_HEAD": [1, 2, 4, 8, 16],
            "BLOCK_TYPE": ["standard", "parallel", "sandwich"],
            "MLP_TYPE": ["standard", "moe", "swiglu", "geglu"],
            "WINDOW_PATTERN": [{"short": 256, "long": 2048}],  # Special handling for S/L combos
        }
    
    @staticmethod
    def window_patterns_for_depth(depth: int) -> List[Dict[str, int]]:
        """Generate window pattern combos for a given depth."""
        short = [128, 256, 512]
        long = [1024, 2048, 4096]
        patterns = []
        for s, l in itertools.product(short, long):
            # Alternate or stack patterns
            patterns.append({"short": s, "long": l})
        return patterns
    
    @staticmethod
    def optimization() -> Dict[str, List[Any]]:
        return {
            "LEARNING_RATE": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1],
            "WARMUP_PCT": [0.0, 0.01, 0.05, 0.10],
            "WARM_DOWN_PCT": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            "WEIGHT_DECAY": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            "MOMENTUM": [0.8, 0.85, 0.9, 0.95, 0.99],
        }
    
    @staticmethod
    def training() -> Dict[str, List[Any]]:
        return {
            "BATCH_SIZE": [16, 32, 64, 128, 256],
            "GRAD_CLIP": [0.5, 1.0, 1.5, 2.0, 5.0],
            "SEQ_LEN": [512, 1024, 2048, 4096, 8192],
        }
    
    @staticmethod
    def regularization() -> Dict[str, List[Any]]:
        return {
            "DROPOUT": [0.02, 0.05, 0.1, 0.15, 0.2],
            "STOCHASTIC_DEPTH": [0.1, 0.2, 0.3],
            "LABEL_SMOOTHING": [0.0, 0.1, 0.2, 0.3],
        }
    
    @staticmethod
    def novel() -> Dict[str, List[Any]]:
        return {
            "ADAPTIVE_SOFTCAP": [0.01, 0.05, 0.1, 0.2],
            "RESIDUAL_LOGIT_BIAS": [0.001, 0.01, 0.05, 0.1],
            "DYNAMIC_WINDOW": [0.5, 1.0, 1.5, 2.0], 
        }


# --- Experiment Catalog & Generation ---
class ExperimentCatalog:
    """Manages the catalog of all possible experiments."""
    
    def __init__(self):
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._generate_catalog()
    
    def _generate_catalog(self):
        """Generate all individual parameter experiments."""
        
        # Architecture
        arch = ParameterSpace.architecture()
        self._add_experiments("architecture", arch)
        
        # Window patterns for specific depths
        for depth in [8, 12, 16]:
            for i, wp in enumerate(ParameterSpace.window_patterns_for_depth(depth)):
                eid = f"exp_{self._next_id():03d}"
                baseline = {"short": 256, "long": 2048}
                props = compute_properties("architecture", "WINDOW_PATTERN", baseline, wp)
                self._catalog[eid] = {
                    "experiment_id": eid,
                    "changes": {"DEPTH": depth, "WINDOW_PATTERN": wp},
                    "category": "architecture",
                    "param": "WINDOW_PATTERN",
                    "properties": props,
                    "hypothesis": f"Using S={wp['short']}/L={wp['long']} window pattern at D={depth}",
                    "priority": props["priority"],
                    "expected_delta": self._expected_delta("architecture", props["estimated_impact"]),
                }
        
        # Optimization
        opt = ParameterSpace.optimization()
        self._add_experiments("optimization", opt)
        
        # Training
        train = ParameterSpace.training()
        self._add_experiments("training", train)
        
        # Regularization
        reg = ParameterSpace.regularization()
        self._add_experiments("regularization", reg)
        
        # Novel
        novel = ParameterSpace.novel()
        self._add_experiments("novel", novel)
    
    def _next_id(self) -> int:
        self._counter += 1
        return self._counter
    
    def _add_experiments(self, category: str, params: Dict[str, List[Any]]):
        for param, values in params.items():
            baselines = {
                "DEPTH": 8,
                "ASPECT_RATIO": 64,
                "HEAD_DIM": 128,
                "BLOCK_TYPE": "standard",
                "MLP_TYPE": "standard",
                "N_KV_HEAD": 4,
                "LEARNING_RATE": 0.03,
                "WARMUP_PCT": 0.05,
                "WARM_DOWN_PCT": 0.10,
                "WEIGHT_DECAY": 0.1,
                "MOMENTUM": 0.9,
                "BATCH_SIZE": 64,
                "SEQ_LEN": 2048,
                "GRAD_CLIP": 1.0,
                "DROPOUT": 0.1,
                "STOCHASTIC_DEPTH": 0.0,
                "LABEL_SMOOTHING": 0.0,
            }
            
            baseline_val = baselines.get(param, values[0])
            for val in values:
                if val == baseline_val:
                    continue  # Skip baseline
                eid = f"exp_{self._next_id():03d}"
                props = compute_properties(category, param, baseline_val, val)
                self._catalog[eid] = {
                    "experiment_id": eid,
                    "changes": {param: val},
                    "category": category,
                    "param": param,
                    "properties": props,
                    "hypothesis": self._generate_hypothesis(category, param, baseline_val, val),
                    "priority": props["priority"],
                    "expected_delta": self._expected_delta(category, props["estimated_impact"]),
                }
    
    def _generate_hypothesis(self, category: str, param: str, baseline: Any, target: Any) -> str:
        hypotheses = {
            "architecture": {
                "DEPTH": lambda b, t: f"Increasing depth from {b} to {t} provides more representational capacity but may require careful init",
                "ASPECT_RATIO": lambda b, t: f"Aspect ratio {t} (vs {b}) balances width and depth for efficient computation",
                "HEAD_DIM": lambda b, t: f"Head dim {t} offers better key-value separation than {b}",
                "BLOCK_TYPE": lambda b, t: f"T(t) ({t}) architecture improves parallel computation vs {b}",
                "MLP_TYPE": lambda b, t: f"{t.upper()} activation potentially improves gradient flow vs {b}",
                "N_KV_HEAD": lambda b, t: f"Using {t} KV heads changes attention expressivity trade-off",
            },
            "optimization": {
                "LEARNING_RATE": lambda b, t: f"LR {t} (vs {b}) expected to [improve/stabilize] convergence",
                "WARMUP_PCT": lambda b, t: f"{int(t*100)}% warmup helps stabilize early training",
                "WARM_DOWN_PCT": lambda b, t: f"Warm down {int(t*100)}% provides smoother convergence end",
                "WEIGHT_DECAY": lambda b, t: f"WD {t} increases regularization to combat overfitting",
                "MOMENTUM": lambda b, t: f"Momentum {t} (higher vs {b}) smooths updates but requires stable LR",
            },
            "training": {
                "BATCH_SIZE": lambda b, t: f"Batch size {t} changes gradient noise and throughput trade-off",
                "GRAD_CLIP": lambda b, t: f"Grad clip {t} prevents gradient explosions during training",
                "SEQ_LEN": lambda b, t: f"Sequence length {t} affects context learning and memory",
            },
            "regularization": {
                "DROPOUT": lambda b, t: f"Dropout {t} combats overfitting but may reduce peak performance",
                "STOCHASTIC_DEPTH": lambda b, t: f"Stochastic depth {t} helps generalization through residual noise",
                "LABEL_SMOOTHING": lambda b, t: f"Label smoothing {t} prevents overconfidence in predictions",
            },
            "novel": {
                "ADAPTIVE_SOFTCAP": lambda b, t: f"Soft cap {t} dynamically stabilizes logits without manual clipping",
                "RESIDUAL_LOGIT_BIAS": lambda b, t: f"Logit bias {t} helps maintain information flow through skip connections",
                "DYNAMIC_WINDOW": lambda b, t: f"Dynamic window {t}x scaling adapts attention range based on context",
            },
        }
        
        fn = hypotheses.get(category, {}).get(param)
        if fn:
            return fn(baseline, target)
        return f"Testing {param}: {target} vs baseline {baseline}"
    
    def _expected_delta(self, category: str, impact: float) -> float:
        """Predict expected metric change (loss delta, lower is better)."""
        return round(-impact * (0.8 + random.random() * 0.5), 3)
    
    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._catalog.values())
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [e for e in self._catalog.values() if e["category"] == category.lower()]
    
    def get_count(self) -> int:
        return len(self._catalog)


# --- Experiment Plan Generator ---
class ExperimentPlanGenerator:
    """Generates different types of experiment combinations and plans."""
    
    def generate_single_factor(
        self,
        catalog: ExperimentCatalog,
        target_category: Optional[str] = None,
        max_count: int = 50,
    ) -> List[Dict[str, Any]]:
        """Pick best single-factor experiments."""
        experiments = catalog.get_all()
        if target_category:
            experiments = [e for e in experiments if e["category"] == target_category.lower()]
        
        # Sort by priority (high to low) and expected_delta (more negative = better)
        sorted_exps = sorted(
            experiments,
            key=lambda x: (x["priority"], x["expected_delta"]),
            reverse=True,
        )
        
        results = []
        for exp in sorted_exps[:max_count]:
            plan = copy.deepcopy(exp)
            plan["design_type"] = "single_factor"
            results.append(plan)
        
        return results
    
    def generate_factorial(
        self, catalog: ExperimentCatalog, target_category: str = None, max_count: int = 20,
    ) -> List[Dict[str, Any]]:
        """Generate 2x2 factorial designs from high-priority parameters."""
        experiments = catalog.get_all()
        if target_category:
            experiments = [e for e in experiments if e["category"] == target_category.lower()]
        
        # Group by parameter
        param_map = defaultdict(list)
        for exp in experiments:
            for param in exp["changes"]:
                param_map[param].append(exp)
        
        fact_results = []
        params_with_many = {p: exps for p, exps in param_map.items() if len(exps) >= 2}
        param_list = list(params_with_many.items())
        
        count = 0
        for (p1, exps1), (p2, exps2) in itertools.islice(
            itertools.combinations(param_list, 2), max_count
        ):
            for e1, e2 in itertools.islice(
                itertools.product(exps1[:3], exps2[:3]), max_count
            ):
                if count >= max_count:
                    break
                eid = f"fact_{count+1:03d}"
                changes = {}
                changes.update(e1["changes"])
                changes.update(e2["changes"])
                impact = max(e1["properties"]["estimated_impact"], e2["properties"]["estimated_impact"]) * 1.2
                risk = min(1.0, (e1["properties"]["risk"] + e2["properties"]["risk"]) * 0.8)
                
                fact_results.append({
                    "experiment_id": eid,
                    "changes": changes,
                    "design_type": "factorial_2x2",
                    "factors": [(p1, e1), (p2, e2)],
                    "hypothesis": f"Interaction between {p1} ({changes.get(p1)}) and {p2} ({changes.get(p2)})",
                    "priority": max(e1["priority"], e2["priority"]),
                    "expected_delta": round(min(e1["expected_delta"], e2["expected_delta"]) * 1.1, 3),
                    "properties": {
                        "estimated_impact": round(impact, 2),
                        "risk": round(risk, 2),
                        "effort": e1["properties"]["effort"] + e2["properties"]["effort"],
                        "category": e1["category"],
                        "dependencies": list(set(e1.get("dependencies", []) + e2.get("dependencies", []))),
                        "compatibility": list(set(e1.get("compatibility", []) + e2.get("compatibility", []))),
                    },
                })
                count += 1
            if count >= max_count:
                break
        
        return fact_results
    
    def generate_known_good_combinations(
        self,
        catalog: ExperimentCatalog,
        good_params: Optional[List[str]] = None,
        max_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Combine parameters likely to have good interaction."""
        good_pairs = [
            ("DEPTH", "ASPECT_RATIO"),
            ("DEPTH", "HEAD_DIM"),
            ("LEARNING_RATE", "WARMUP_PCT"), 
            ("LEARNING_RATE", "WEIGHT_DECAY"), 
            ("BATCH_SIZE", "LEARNING_RATE"),
            ("WINDOW_PATTERN", "DEPTH"),
            ("BLOCK_TYPE", "DEPTH"),
            ("DROPOUT", "STOCHASTIC_DEPTH"),
            ("MLP_TYPE", "HEAD_DIM"),
            ("SEQ_LEN", "BATCH_SIZE"),
        ]
        
        if good_params:
            good_pairs = [p for p in good_pairs if p[0] in good_params or p[1] in good_params]
        
        experiments = catalog.get_all()
        param_map = defaultdict(list)
        for exp in experiments:
            for param in exp["changes"]:
                param_map[param].append(exp)
        
        results = []
        count = 0
        for p1, p2 in good_pairs:
            if p1 not in param_map or p2 not in param_map:
                continue
            for e1, e2 in itertools.islice(
                itertools.product(param_map[p1][:3], param_map[p2][:3]), max_count
            ):
                if count >= max_count:
                    break
                eid = f"combo_{count+1:03d}"
                changes = {}
                changes.update(e1["changes"])
                changes.update(e2["changes"])
                
                results.append({
                    "experiment_id": eid,
                    "changes": changes,
                    "design_type": "known_good_combo",
                    "hypothesis": f"Combining {p1}={changes.get(p1)} and {p2}={changes.get(p2)} based on prior knowledge",
                    "priority": max(e1["priority"], e2["priority"]),
                    "expected_delta": round(min(e1["expected_delta"], e2["expected_delta"]) * 1.15, 3),
                    "properties": {
                        "estimated_impact": round((e1["properties"]["estimated_impact"] + e2["properties"]["estimated_impact"]) / 2, 2),
                        "risk": round((e1["properties"]["risk"] + e2["properties"]["risk"]) * 0.7, 2),
                        "effort": max(e1["properties"]["effort"] + 1, e2["properties"]["effort"] + 1),
                        "category": e1["category"],
                        "dependencies": list(set(e1.get("dependencies", []) + e2.get("dependencies", []))),
                        "compatibility": list(set(e1.get("compatibility", []) + e2.get("compatibility", []))),
                    },
                })
                count += 1
            if count >= max_count:
                break
        
        return results
    
    def generate_ablation(
        self, catalog: ExperimentCatalog, base_config_name: str = "baseline",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate ablation study from a baseline configuration."""
        experiments = catalog.get_all()
        ablation = []
        
        # Group best by category
        best_by_cat = {}
        for exp in experiments:
            cat = exp["category"]
            if cat not in best_by_cat or exp["priority"] > best_by_cat[cat]["priority"]:
                best_by_cat[cat] = exp
        
        count = 0
        for cat, exp in best_by_cat.items():
            count += 1
            eid = f"ablation_{count:03d}"
            changes = {k: exp["changes"][k] for k in exp["changes"]}
            ablation.append({
                "experiment_id": eid,
                "changes": changes,
                "design_type": "ablation",
                "ablation_from": base_config_name,
                "hypothesis": f"Ablating {cat} parameter {exp['changes']} from {base_config_name}",
                "priority": exp["priority"],
                "expected_delta": exp["expected_delta"],
                "properties": exp["properties"],
            })
        
        # Combined ablation - turn everything off from best
        combined_changes = {}
        for exp in best_by_cat.values():
            combined_changes.update(exp["changes"])
        
        ablation.append({
            "experiment_id": f"ablation_full_{0:03d}",
            "changes": combined_changes,
            "design_type": "ablation_full",
            "ablation_from": None,
            "hypothesis": f"Remove all best changes (full ablation) from baseline",
            "priority": 10,
            "expected_delta": -0.1,
            "properties": {
                "estimated_impact": 0.9,
                "risk": 0.8,
                "effort": 5,
                "category": "multi",
                "dependencies": [],
                "compatibility": list(best_by_cat.keys()),
            },
        })
        
        return "baseline", ablation


# --- Results Tracker (Self-improving) ---
class ResultsTracker:
    """Tracks experiment results and updates priorities for self-improvement.
    
    Parses results.tsv to update expected_delta and priority based on observed outcomes.
    """
    
    def update_from_results(
        self,
        experiments: List[Dict[str, Any]],
        results_path: str,
        decay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Update experiment priorities based on observed results.
        
        Args:
            experiments: List of experiment plans
            results_path: Path to results.tsv
            decay: Weight of new results vs old estimates (0.5 = equal)
        """
        results = self._parse_results(results_path)
        updated_experiments = []
        
        for exp in experiments:
            eid = exp["experiment_id"]
            if eid in results:
                result = results[eid]
                observed_delta = result.get("observed_delta", exp["expected_delta"])
                
                # Blend old expectation with new observation
                new_delta = (1 - decay) * exp["expected_delta"] + decay * observed_delta
                
                # Update priority based on actual performance
                # Better results = higher priority
                if observed_delta < exp["expected_delta"]:
                    exp["priority"] = min(10, exp["priority"] + 2)
                elif observed_delta > exp["expected_delta"]:
                    exp["priority"] = max(1, exp["priority"] - 1)
                
                exp["expected_delta"] = round(new_delta, 3)
                exp["observed_delta"] = observed_delta
                exp["result_source"] = result.get("source", "manual")
                exp["priority_reason"] = f"Adjusted from results: observed {observed_delta:.4f}"
                
                updated_experiments.append(exp)
            else:
                updated_experiments.append(exp)
        
        # Sort by updated priority
        sorted_exps = sorted(
            updated_experiments, key=lambda x: x["priority"], reverse=True,
        )
        return sorted_exps
    
    def _parse_results(self, results_path: str) -> Dict[str, Dict[str, Any]]:
        """Parse results.tsv into a dict keyed by experiment_id."""
        results = {}
        
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found, skipping update")
            return results
        
        with open(results_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return results
        
        # Parse header
        header = [h.strip() for h in lines[0].split("\t")]
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            vals = line.split("\t")
            row = dict(zip(header, vals))
            
            eid = row.get("experiment_id", "")
            if not eid:
                continue
            
            results[eid] = {
                "observed_delta": float(row.get("observed_delta", 0)),
                "source": row.get("source", "manual"),
                "status": row.get("status", "completed"),
            }
        
        return results


# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(
        description="Experiment Designer - Systematic experimental design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Generation args
    parser.add_argument("--generate", type=int, default=0, help="Number of experiments to generate")
    parser.add_argument("--phase", type=int, default=1, help="Experiment phase (1, 2, 3...)")
    parser.add_argument("--category", type=str, default=None, help="Limit to category: architecture|optimization|training|regularization|novel")
    parser.add_argument("--output", type=str, default="experiments.json", help="Output file path")
    parser.add_argument("--design-type", type=str, default="single", choices=["single", "factorial", "combo", "ablation"], help="Type of experiment to generate")
    
    # Update args
    parser.add_argument("--load", type=str, default=None, help="Load existing experiments JSON")
    parser.add_argument("--update", action="store_true", help="Update experiments based on results")
    parser.add_argument("--results", type=str, default="results.tsv", help="Path to results TSV")
    parser.add_argument("--decay", type=float, default=0.5, help="Weight for new vs old expectations")
    
    # Suggestion args
    parser.add_argument("--suggest-combinations", action="store_true", help="Generate suggested experiment combinations")
    parser.add_argument("--top", type=int, default=3, help="Top N combinations to suggest")
    
    # List args
    parser.add_argument("--list-all", action="store_true", help="List all experiments, optionally filtered by category")
    
    args = parser.parse_args()
    
    catalog = ExperimentCatalog()
    generator = ExperimentPlanGenerator()
    tracker = ResultsTracker()
    
    # ---- Generate Mode ----
    if args.generate > 0:
        print(f"Generating {args.generate} experiments (Phase {args.phase}, Category: {args.category or 'all'})")
        plans = []
        
        if args.design_type == "single" or args.design_type == "combo":
            single = generator.generate_single_factor(catalog, args.category, args.generate // 2)
            plans.extend(single)
        
        if args.design_type == "factorial":
            fact = generator.generate_factorial(catalog, args.category, args.generate)
            plans.extend(fact)
        
        if args.design_type == "combo":
            combos = generator.generate_known_good_combinations(catalog, max_count=args.generate // 2)
            plans.extend(combos)
        
        # Fill remaining with best if needed
        if len(plans) < args.generate and args.design_type == "single":
            remaining = args.generate - len(plans)
            additional = generator.generate_single_factor(catalog, args.category, remaining * 2)
            existing_ids = {p["experiment_id"] for p in plans}
            for exp in additional:
                if exp["experiment_id"] not in existing_ids:
                    plans.append(exp)
                    if len(plans) >= args.generate:
                        break
        
        # Add metadata
        output = {
            "metadata": {
                "phase": args.phase,
                "generated_at": "now",
                "count": len(plans),
                "categories": list(set(p.get("category", "unknown") for p in plans)),
                "design_types": list(set(p.get("design_type", "unknown") for p in plans)),
            },
            "experiments": plans[:args.generate],
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Wrote {len(output['experiments'])} experiments to {args.output}")
        return
    
    # ---- Update Mode ----
    if args.update and args.load:
        print(f"Loading experiments from {args.load}, applying results from {args.results}")
        with open(args.load, 'r') as f:
            data = json.load(f)
        
        experiments = data.get("experiments", [])
        experiments = tracker.update_from_results(experiments, args.results, args.decay)
        
        data["experiments"] = experiments
        data["metadata"]["updated"] = True
        data["metadata"]["updated_at"] = "now"
        
        with open(args.load, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Updated priorities for experiments in {args.load}")
        return
    
    # ---- Suggest Combinations ----
    if args.suggest_combinations:
        combos = generator.generate_known_good_combinations(catalog, max_count=args.top)
        print(f"Top {args.top} suggested experiment combinations:")
        for i, combo in enumerate(combos[:args.top]):
            print(f"\n{combo['experiment_id']} (Priority: {combo['priority']})")
            print(f"  Changes: {json.dumps(combo['changes'])}")
            print(f"  Expected delta: {combo['expected_delta']}")
            print(f"  Hypothesis: {combo['hypothesis']}")
        return
    
    # ---- List All ----
    if args.list_all:
        experiments = catalog.get_all()
        if args.category:
            experiments = catalog.get_by_category(args.category)
        
        experiments_sorted = sorted(experiments, key=lambda x: x["priority"], reverse=True)
        
        print(f"\n{'ID':<10} {'Priority':<10} {'Impact':<10} {'Risk':<10} {'Effort':<10} {'Category':<15} {'Parameter':<20}")
        print("-" * 85)
        
        for exp in experiments_sorted[:50]:  # Show top 50
            props = exp.get("properties", {})
            print(
                f"{exp['experiment_id']:<10} "
                f"{exp.get('priority', 0):<10} "
                f"{props.get('estimated_impact', 0):<10} "
                f"{props.get('risk', 0):<10} "
                f"{props.get('effort', 0):<10} "
                f"{exp.get('category', 'unknown'):<15} "
                f"{exp.get('param', list(exp.get('changes', {}).keys()))}"
            )
        
        print(f"\nTotal: {len(experiments_sorted)} experiments (showing top 50)")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
