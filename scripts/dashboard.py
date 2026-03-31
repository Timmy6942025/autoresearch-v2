#!/usr/bin/env python3
from __future__ import annotations

"""
VISUALIZATION DASHBOARD FOR AUTORESEARCH v2

Generates progress plots and markdown reports from experiment results.
Run after every ~20 experiments for visual analytics.

Usage:
    python dashboard.py --results results/results.tsv --output-md report.md --plots-dir plots/
    python dashboard.py --results results/results.tsv --results-json results/research_results.json --output-md report.md --plots-dir plots/
    python dashboard.py --results results/results.tsv --baseline 1.45 --output-md report.md --plots-dir plots/

All plots save to the --plots-dir directory.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator


# ─── Category mapping ─────────────────────────────────────────────────────────

# Attempt to derive a visual category from change names / descriptions.
_KNOWN_CATEGORIES = {
    "hyperparameter": "HyperParams",
    "architectural": "Architecture",
    "training_loop": "Training",
    "data_pipeline": "Data",
    "regularization": "Regularization",
    # aliases
    "architecture": "Architecture",
    "hyperparams": "HyperParams",
}

_CATEGORY_COLORS = {
    "HyperParams":    "#2196F3",  # blue
    "Architecture":   "#F44336",  # red
    "Training":       "#4CAF50",  # green
    "Data":           "#FF9800",  # orange
    "Regularization": "#9C27B0",  # purple
    "Baseline":       "#607D8B",  # blue-grey
    "Other":          "#78909C",  # grey
}


def _parse_category(raw: str) -> str:
    """Derive a display category from change-name text."""
    if not raw:
        return "Other"
    lower = raw.lower()
    for key, label in _KNOWN_CATEGORIES.items():
        if key in lower:
            return label
    # Check by keyword heuristics
    if any(kw in lower for kw in ["depth", "width", "layer", "head", "hidden"]):
        return "Architecture"
    if any(kw in lower for kw in ["lr", "batch", "epoch", "warmup", "scheduler", "optimizer"]):
        return "HyperParams"
    if any(kw in lower for kw in ["dropout", "wd", "weight_decay", "clip", "grad_clip", "early"]):
        return "Training"
    if any(kw in lower for kw in ["regular", "augment", "noise"]):
        return "Regularization"
    if any(kw in lower for kw in ["data", "sample", "tokenize", "vocab", "sequence"]):
        return "Data"
    return "Other"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tsv(path: str) -> list[dict]:
    """Load results from results.tsv."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        headers = header_line.split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(headers, parts))
            # Parse numeric fields
            for key in ("val_bpb", "delta", "duration"):
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = None
            # Ensure experiment_id / phase are parsed
            if "phase" in row:
                try:
                    row["phase"] = str(row["phase"])
                except Exception:
                    pass
            rows.append(row)
    return rows


def load_json(path: str) -> list[dict]:
    """Load rich results from research_results.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def prepare_dataset(rows: list[dict]) -> list[dict]:
    """Normalise TSV/JSON rows into a unified record list."""
    records = []
    for i, r in enumerate(rows):
        rec = {
            "experiment_id": r.get("experiment_id", r.get("name", f"exp_{i}")),
            "experiment_name": r.get("experiment_name", r.get("experiment_id", r.get("name", f"exp_{i}"))),
            "val_bpb": r.get("val_bpb") or 0.0,
            "delta": r.get("delta") or 0.0,
            "phase": str(r.get("phase", "0")),
            "status": r.get("status", "unknown"),
            "timestamp": r.get("timestamp", ""),
            "changes": r.get("changes", r.get("changes_applied", [])),
            "description": r.get("description", ""),
            "configuration": r.get("configuration", {}),
            "baseline_bpb": r.get("baseline_bpb"),
        }
        # Derive category from changes
        changes = rec["changes"]
        if isinstance(changes, str):
            changes = changes.split("|")
        cats = [_parse_category(c) for c in changes]
        rec["category"] = cats[0] if cats else "Other"
        # Treat baseline experiments specially
        if rec["status"] == "baseline":
            rec["category"] = "Baseline"
        records.append(rec)
    return records


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str, plots_dir: Path, dpi: int = 150):
    _ensure_dir(plots_dir)
    out = plots_dir / name
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 1. VAL_BPB PROGRESSION PLOT ─────────────────────────────────────────────

def plot_val_bpb_progression(records: list[dict], plots_dir: Path,
                              baseline: float | None = None) -> Path:
    """X=experiment number, Y=val_bpb. Points colored by category, with
    baseline and best-so-far horizontal lines."""

    # Sort by timestamp or experiment number
    records = sorted(records, key=lambda r: r.get("experiment_id", ""))

    fig, ax = plt.subplots(figsize=(14, 7))

    xs = []
    ys = []
    categories = []
    descriptions = []
    statuses = []

    for idx, r in enumerate(records):
        if r["val_bpb"] is None or r["val_bpb"] <= 0:
            continue
        xs.append(idx + 1)
        ys.append(r["val_bpb"])
        categories.append(r["category"])
        descriptions.append(r["experiment_name"] or r["experiment_id"])
        statuses.append(r["status"])

    if not xs:
        # Draw empty plot with a label
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, "val_bpb_progress.png", plots_dir)
        return plots_dir / "val_bpb_progress.png"

    # Scatter by category
    unique_cats = list(dict.fromkeys(categories))  # preserve order
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cx = [x for x, m in zip(xs, mask) if m]
        cy = [y for y, m in zip(ys, mask) if m]
        color = _CATEGORY_COLORS.get(cat, "#78909C")
        ax.scatter(cx, cy, c=color, s=70, label=cat, zorder=3, edgecolors="white", linewidth=0.5)

    # Annotate top improvements (best 7 lowest val_bpb)
    best_pts = sorted(zip(xs, ys, descriptions), key=lambda t: t[1])[:7]
    for x_val, y_val, label in best_pts:
        ax.annotate(label, (x_val, y_val), textcoords="offset points",
                    xytext=(5, 8), ha="left", fontsize=6.5, color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaa", alpha=0.9))

    # Baseline line
    if baseline is not None:
        ax.axhline(y=baseline, color="red", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.4f})", zorder=2)

    # Best-so-far line (running minimum)
    if ys:
        running_min = []
        ymin = float("inf")
        for y in ys:
            if y < ymin:
                ymin = y
            running_min.append(ymin)
        ax.plot(xs, running_min, color="green", linestyle="-", linewidth=1.5,
                alpha=0.5, label="Best-so-far", zorder=1)

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title("autoresearch v2 — Validation BPB Progression")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "val_bpb_progress.png", plots_dir)
    return plots_dir / "val_bpb_progress.png"


# ─── 2. PARAMETER IMPACT PLOTS ───────────────────────────────────────────────

def _extract_param_values(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Extract parameter→{value_str: [val_bpb_list]} from experiment configurations."""
    param_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["val_bpb"] is None or r["val_bpb"] <= 0:
            continue
        if r["status"] in ("failed", "crashed", "skipped"):
            continue
        cfg = r.get("configuration") or {}
        if not cfg:
            # Fallback: try to pull from experiment_name or description
            continue
        for k, v in cfg.items():
            key = str(k).upper()
            val_key = f"{v}" if not isinstance(v, float) else f"{v:.6g}"
            param_data[key][val_key].append(r["val_bpb"])
    return dict(param_data)


def plot_parameter_impacts(records: list[dict], plots_dir: Path) -> list[Path]:
    """For each parameter that has ≥2 distinct values, plot mean val_bpb per value with std error bars."""
    param_data = _extract_param_values(records)
    saved = []

    if not param_data:
        # Create a placeholder telling the user
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No parameter-level configuration data available\n"
                " (records lack 'configuration' field with varied params)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Parameter Impact — No Data")
        saved.append(_save(fig, "param_impact_NO_DATA.png", plots_dir))
        return saved

    for param_name, val_groups in param_data.items():
        if len(val_groups) < 2:
            continue  # skip params with only one value

        labels = list(val_groups.keys())
        means = []
        stds = []
        counts = []

        for v in labels:
            data = val_groups[v]
            data = [x for x in data if x > 0]
            if data:
                means.append(sum(data) / len(data))
                stds.append((sum((x - means[-1])**2 for x in data) / len(data))**0.5 if len(data)>1 else 0)
                counts.append(len(data))
            else:
                means.append(None)
                stds.append(0)
                counts.append(0)

        valid = [(l, m, s, c) for l, m, s, c in zip(labels, means, stds, counts) if m is not None]
        if not valid:
            continue

        labels, plot_means, plot_stds, plot_counts = zip(*valid)

        fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.2), 5))
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, plot_means, yerr=plot_stds, capsize=4,
                       color="#2196F3", alpha=0.7, edgecolor="#1565C0")

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel(f"{param_name} Value")
        ax.set_ylabel("Mean Validation BPB (lower = better)")
        ax.set_title(f"Parameter Impact: {param_name}")

        # Annotate bars with mean values
        for i, (m, s) in enumerate(zip(plot_means, plot_stds)):
            ax.text(i, m + s + 0.003, f"{m:.4f} (n={plot_counts[i]})",
                    ha="center", va="bottom", fontsize=7)

        # Highlight optimal
        best_idx = plot_means.index(min(plot_means))
        bars[best_idx].set_color("#4CAF50")
        bars[best_idx].set_alpha(0.9)
        ax.annotate("Optimal", xy=(best_idx, plot_means[best_idx]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#2E7D32", fontweight="bold")

        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        safe_name = param_name.replace(" ", "_").replace("/", "_")
        saved.append(_save(fig, f"param_impact_{safe_name}.png", plots_dir))

    if not saved:
        # No parameters with ≥2 values — create placeholder
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "All configuration parameters have only one value — no impact comparison possible.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Parameter Impact — No Variation")
        saved.append(_save(fig, "param_impact_NO_VARIATION.png", plots_dir))

    return saved


# ─── 3. SUCCESS RATE CHART ───────────────────────────────────────────────────

def plot_success_rate(records: list[dict], plots_dir: Path) -> Path:
    """Bar chart of success rate by experiment category."""
    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    for r in records:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        if r["status"] in ("success",):
            cat_stats[cat]["success"] += 1

    categories = sorted(cat_stats.keys())
    rates = []
    counts = []
    for cat in categories:
        s = cat_stats[cat]["success"]
        t = cat_stats[cat]["total"]
        rate = (s / t * 100) if t > 0 else 0
        rates.append(rate)
        counts.append(f"{s}/{t}")

    fig, ax = plt.subplots(figsize=(max(8, len(categories)*1.1), 5))
    colors = [_CATEGORY_COLORS.get(c, "#78909C") for c in categories]
    bars = ax.bar(categories, rates, color=colors, alpha=0.8, edgecolor="none")

    for bar, rate_str, rate_val in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate_str}\n({rate_val:.0f}%)", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xlabel("Experiment Category")
    ax.set_ylabel("Improvement Success Rate (%)")
    ax.set_title("Success Rate by Research Direction")
    ax.set_ylim(0, max(100, max(rates) * 1.3) if rates else 100)
    ax.axhline(20, color="red", linestyle="--", alpha=0.4, label="20% threshold")
    ax.axhline(50, color="green", linestyle="--", alpha=0.4, label="50% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "success_rate.png", plots_dir)
    return plots_dir / "success_rate.png"


# ─── 4. IMPROVEMENT TREND (ROLLING) ──────────────────────────────────────────

def plot_improvement_trend(records: list[dict], plots_dir: Path, window: int = 10) -> Path:
    """Rolling average of improvement magnitude over time — shows diminishing returns."""
    records = sorted(records, key=lambda r: r.get("experiment_id", ""))

    xs = []
    deltas = []
    for idx, r in enumerate(records):
        if r["val_bpb"] is None or r["val_bpb"] <= 0:
            continue
        xs.append(idx + 1)
        deltas.append(r["delta"] if r["delta"] else 0.0)

    if len(deltas) < 3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Not enough data for trend (need ≥3 experiments)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Improvement Trend — Insufficient Data")
        _save(fig, "improvement_trend.png", plots_dir)
        return plots_dir / "improvement_trend.png"

    # Rolling mean and std
    n = len(deltas)
    roll_mean = []
    roll_std = []
    roll_xs = []
    for i in range(n):
        start = max(0, i - window + 1)
        window_data = deltas[start: i + 1]
        m = sum(window_data) / len(window_data)
        s = (sum((x - m)**2 for x in window_data) / max(len(window_data)-1, 1))**0.5
        roll_mean.append(m)
        roll_std.append(s)
        roll_xs.append(i + 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Individual deltas as dots
    ax.scatter(xs, deltas, c="#B0BEC5", s=30, alpha=0.6, label="Individual experiment delta")

    # Rolling mean line
    ax.plot(roll_xs, roll_mean, color="#1565C0", linewidth=2.5, label=f"Rolling mean (window={window})")

    # Rolling std band
    ax.fill_between(roll_xs,
                    [m - s for m, s in zip(roll_mean, roll_std)],
                    [m + s for m, s in zip(roll_mean, roll_std)],
                    color="#1565C0", alpha=0.15, label="Rolling ±1σ")

    # Zero line
    ax.axhline(0, color="black", linewidth=1, alpha=0.3)

    # Trend line (linear fit) — shows overall direction
    if len(roll_xs) >= 2:
        coeffs = _linear_fit(roll_xs, roll_mean)
        trend_y = [coeffs[0] * x + coeffs[1] for x in roll_xs]
        ax.plot(roll_xs, trend_y, color="#FF5722", linestyle="--", linewidth=1.5,
                label=f"Trend (slope={coeffs[0]:.6f})")

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Delta (Baseline - val_bpb, positive = improvement)")
    ax.set_title("Improvement Magnitude Trend — Diminishing Returns Detection")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "improvement_trend.png", plots_dir)
    return plots_dir / "improvement_trend.png"


def _linear_fit(xs: list, ys: list) -> tuple:
    """Simple least-squares linear fit, returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return (0, 0)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return (0, y_mean)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return (slope, intercept)


# ─── 5. MARKDOWN REPORT ─────────────────────────────────────────────────────

def generate_markdown_report(records: list[dict], baseline: float | None = None,
                              plots_dir: Path | None = None) -> str:
    """Generate a concise markdown progress report."""
    if baseline is None:
        # Infer baseline from baseline experiments
        baseline_rows = [r for r in records if r["status"] == "baseline" and r["val_bpb"]]
        if baseline_rows:
            baseline = baseline_rows[-1]["val_bpb"]
        else:
            baseline = 1.45  # default fallback

    total = len(records)
    successful = [r for r in records if r["status"] == "success" and r.get("delta") and r["delta"] > 0]
    failed = [r for r in records if r["status"] in ("failed", "crashed")]
    success_pct = (len(successful) / total * 100) if total > 0 else 0

    # Best val_bpb
    valid = [r for r in records if r["val_bpb"] and r["val_bpb"] > 0]
    best_record = min(valid, key=lambda r: r["val_bpb"]) if valid else None
    best_bpb = best_record["val_bpb"] if best_record else None
    best_id = best_record["experiment_id"] if best_record else "N/A"
    best_name = best_record.get("experiment_name", "") if best_record else ""

    # Best configuration
    best_cfg = {}
    if best_record:
        best_cfg = best_record.get("configuration", {})

    # Top improvements
    improvements = sorted(successful, key=lambda r: r.get("delta", 0) or 0, reverse=True)[:10]

    # Category breakdown
    cat_counts: dict[str, int] = defaultdict(int)
    cat_success: dict[str, int] = defaultdict(int)
    for r in records:
        cat_counts[r["category"]] += 1
        if r["status"] == "success" and r.get("delta") and r["delta"] > 0:
            cat_success[r["category"]] += 1

    lines = []
    lines.append("# Autoresearch Progress Report\n")
    lines.append(f"## Summary\n")
    lines.append(f"- **Total experiments:** {total}")
    lines.append(f"- **Successful improvements:** {len(successful)} ({success_pct:.1f}%)")
    lines.append(f"- **Failed/crashed:** {len(failed)}")
    lines.append(f"- **Best val_bpb:** {best_bpb:.4f}" if best_bpb else "- **Best val_bpb:** N/A")
    lines.append(f"  - Experiment: **{best_id}** ({best_name})" if best_bpb else "")
    lines.append(f"- **Baseline:** {baseline:.4f}")
    lines.append(f"- **Current best config:** ```{json.dumps(best_cfg, indent=2)}```" if best_cfg else "- **Current best config:** N/A")
    lines.append("")

    # Top improvements table
    lines.append("## Top Improvements\n")
    if improvements:
        lines.append("| Rank | Experiment | val_bpb | Delta | Category |")
        lines.append("|------|------------|---------|-------|----------|")
        for rank, r in enumerate(improvements, 1):
            lines.append(f"| {rank} | {r.get('experiment_name', r.get('experiment_id', '???'))} | {r['val_bpb']:.4f} | {r.get('delta', 0):+.6f} | {r['category']} |")
    else:
        lines.append("_No successful improvements yet._")
    lines.append("")

    # Category breakdown
    lines.append("## Success Rate by Category\n")
    for cat in sorted(cat_counts.keys()):
        s = cat_success.get(cat, 0)
        t = cat_counts[cat]
        pct = (s / t * 100) if t > 0 else 0
        bar = "█" * int(pct / 5)
        lines.append(f"- **{cat}**: {s}/{t} ({pct:.0f}%) {bar}")
    lines.append("")

    # Plots generated
    if plots_dir:
        lines.append("## Generated Plots\n")
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            for pf in plot_files:
                lines.append(f"- `{pf.name}`")
        else:
            lines.append("_No plots generated yet._")
        lines.append("")

    # Trendline analysis
    if len(valid) >= 5:
        xs = list(range(1, len(valid) + 1))
        deltas = [r.get("delta", 0) or 0 for r in sorted(valid, key=lambda r: r.get("experiment_id", ""))]
        slope, _ = _linear_fit(xs, deltas)
        if slope > 0.0001:
            trend_str = f"**Improving** (average delta increasing, slope={slope:.6f})"
        elif slope < -0.0001:
            trend_str = f"**Diminishing returns** (average delta decreasing, slope={slope:.6f})"
        else:
            trend_str = f"**Flat** (no clear trend, slope={slope:.6f})"
        lines.append(f"## Trend\n\n- {trend_str}\n")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch v2 Visualization Dashboard"
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to results.tsv file"
    )
    parser.add_argument(
        "--results-json", default=None,
        help="Optional: Path to research_results.json for richer data (configs, descriptions)"
    )
    parser.add_argument(
        "--output-md", default="report.md",
        help="Path for markdown report output (default: report.md)"
    )
    parser.add_argument(
        "--plots-dir", default="plots",
        help="Directory to save plots (default: plots/)"
    )
    parser.add_argument(
        "--baseline", type=float, default=None,
        help="Override baseline value (default: auto-detect or 1.45)"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=10,
        help="Window size for rolling improvement trend (default: 10)"
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    _ensure_dir(plots_dir)

    # Load data
    print(f"[dashboard] Loading results from {args.results}")
    if not Path(args.results).exists():
        print(f"[dashboard] ERROR: Results file not found: {args.results}")
        sys.exit(1)

    records = load_tsv(args.results)
    print(f"[dashboard] Loaded {len(records)} records from TSV")

    # Optionally enrich with JSON data
    if args.results_json and Path(args.results_json).exists():
        json_records = load_json(args.results_json)
        print(f"[dashboard] Enriching with {len(json_records)} JSON records")
        # Build lookup by experiment_id
        json_lookup = {r.get("experiment_id", r.get("name", "")): r for r in json_records}
        for rec in records:
            eid = rec.get("experiment_id", rec.get("name", ""))
            if eid in json_lookup:
                jr = json_lookup[eid]
                rec["description"] = jr.get("description", rec.get("description", ""))
                rec["configuration"] = jr.get("configuration", rec.get("configuration", {}))
                rec["changes_applied"] = jr.get("changes_applied", [])
                if jr.get("baseline_bpb"):
                    rec["baseline_bpb"] = jr["baseline_bpb"]

    # Prepare dataset
    records = prepare_dataset(records)

    # Infer or use baseline
    baseline = args.baseline
    if baseline is None:
        baseline_rows = [r for r in records if r["status"] == "baseline" and r.get("val_bpb")]
        if baseline_rows:
            baseline = baseline_rows[-1]["val_bpb"]
        else:
            baseline = 1.45

    print(f"[dashboard] Using baseline: {baseline}")
    print(f"[dashboard] Non-baseline records: {len([r for r in records if r['status'] != 'baseline'])}")

    # Generate plots
    print(f"[dashboard] Generating val_bpb_progression plot...")
    plot_val_bpb_progression(records, plots_dir, baseline=baseline)

    print(f"[dashboard] Generating parameter impact plots...")
    param_paths = plot_parameter_impacts(records, plots_dir)
    print(f"[dashboard]   → {len(param_paths)} parameter impact plots")

    print(f"[dashboard] Generating success rate chart...")
    plot_success_rate(records, plots_dir)

    print(f"[dashboard] Generating improvement trend plot...")
    plot_improvement_trend(records, plots_dir, window=args.rolling_window)

    # Generate markdown report
    print(f"[dashboard] Generating markdown report...")
    md = generate_markdown_report(records, baseline=baseline, plots_dir=plots_dir)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[dashboard] Report written to {args.output_md}")

    print("[dashboard] Done.")


if __name__ == "__main__":
    main()
