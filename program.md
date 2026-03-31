# Autoresearch V2 — Agent Operating Manual

## 1. SYSTEM OVERVIEW

Autoresearch V2 replaces v1's simple hill-climbing with a multi-component autonomous research system.
The agent NEVER asks permission to continue — it runs indefinitely until manually stopped.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI RESEARCH AGENT                            │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────────┐      │
│  │ orchestrator │─▶│ knowledge  │◀─│ experiment_designer │      │
│  │  (loop ctrl) │  │  base      │  │ (plan generator)    │      │
│  └──────┬───────┘  └──────┬─────┘  └──────────┬──────────┘      │
│         │                 │                    │                  │
│         ▼                 ▼                    ▼                  │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────┐      │
│  │  train.py    │  │meta_analyzer│  │    dashboard.py     │      │
│  │ (MODIFY)     │  │ (insights)  │  │  (visual analytics) │      │
│  └──────┬───────┘  └─────────────┘  └─────────────────────┘      │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐                                                  │
│  │ prepare.py   │ ← READ-ONLY (data, tokenizer, eval harness)     │
│  └──────────────┘                                                  │
│  Persistent State: knowledge.json, research_results.json,           │
│  results.tsv, logs/*, plots/*                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Component Roles

- **research_orchestrator.py** — Master loop with 4-phase strategy, Bayesian+Thompson-Sampling
  prioritization, crash detection, results database. Run:
  `python scripts/research_orchestrator.py --phase 1 --max-experiments 100`
- **knowledge_base.py (1016 lines)** — Persistent structured memory. Tracks improvements, dead
  ends, per-category success rates, interaction records. Key methods: .record_result(),
  .suggest_next_experiment(), .get_best_config(), .get_combination_suggestions(),
  .export_markdown(). The knowledge.json file survives across sessions.
- **meta_analyzer.py (1608 lines)** — Deep statistical analysis. Computes parameter correlations,
  interaction matrices, diminishing-returns detection, ranked hypotheses. Run:
  `python scripts/meta_analyzer.py --results results/results.tsv --output analysis.md
  --next-experiments 5`
- **experiment_designer.py (802 lines)** — Systematic plan generator across 5 categories:
  architecture, optimization, training, regularization, novelty. Supports single-factor,
  factorial 2×2, known-good-combo, and ablation designs with impact/risk priority scoring.
- **dashboard.py (673 lines)** — Visual analytics: val_bpb progress, parameter impact plots,
  success-rate charts, improvement trends. Generates PNG plots + markdown report. Run every ~20
  experiments: `python scripts/dashboard.py --results results/results.tsv --output-md report.md
  --plots-dir plots/`
- **train.py** — The ONLY file you modify. Enhanced v2 version with many tunable parameters.
- **prepare.py** — READ-ONLY. Data prep, tokenizer, evaluator. Never modify.

---

## 2. INITIALIZATION

Complete these steps before the first experiment:

### 2.1 Git + Environment Setup
```bash
cd /tmp/autoresearch-v2
git checkout -b autoresearch/v2  # or increment night counter
ls ~/.cache/autoresearch/        # verify data exists
uv run python -c "import torch; print(torch.__version__)"
```

### 2.2 Knowledge Base Init
```python
from scripts.knowledge_base import KnowledgeBase
kb = KnowledgeBase(path="knowledge.json")  # auto-seeds if fresh
kb.set_current_branch("autoresearch/v2")
```

### 2.3 Baseline Run (MUST be first)
```bash
uv run train.py > run.log 2>&1
grep "^val_bpb:\\|^peak_vram_mb:" run.log
# Record baseline with the val_bpb you get:
python -c "
from scripts.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
kb.record_result('exp_000_baseline', {}, YOUR_VAL_BPB, 'baseline', 'Initial baseline')
"
```

### 2.4 Experiment Plan Generation
```bash
python scripts/experiment_designer.py --output scripts/experiments.json --plan single_factor --max-count 30
```

---

## 3. PHASED RESEARCH STRATEGY

Four sequential phases with distinct goals. Transition on quality criteria, not just count.

### Phase 1 (Experiments 1-30): Single-Factor Systematic Exploration
Map individual parameter impact in isolation. Cycle through categories: architecture → optimization
→ training → regularization → novelty. Test parameters at 2-4 values to establish response curves.

**Architecture params:** DEPTH (4-16), ASPECT_RATIO (32-128), HEAD_DIM (64-256), BLOCK_TYPE
(standard/parallel/sandwich), MLP_TYPE (standard/moe/swiglu/geglu), N_KV_HEAD (1-16),
WINDOW_PATTERN (short/long alternation).

**Optimization params:** LEARNING_RATE (0.001-0.1), WARMUP_PCT (0-10%), WARM_DOWN_PCT (5-50%),
WEIGHT_DECAY (0.01-0.5), MOMENTUM (0.8-0.99). LR is highest-priority — test wide range first.

**Training params:** BATCH_SIZE (16-256), GRAD_CLIP (0.5-5.0), SEQ_LEN (512-8192). Larger batches
give smoother gradients; longer sequences improve context but reduce throughput.

**Regularization params:** DROPOUT (0.02-0.2), STOCHASTIC_DEPTH (0.1-0.3), LABEL_SMOOTHING (0-0.3).
These combat overfitting on the evaluation set but may hurt peak performance.

**Novelty params:** ADAPTIVE_SOFTCAP (0.01-0.2), RESIDUAL_LOGIT_BIAS (0.001-0.1), DYNAMIC_WINDOW
(0.5-2.0x). High-risk experiments — expect crashes and dead ends.

**Exit criteria:** ≥15 experiments, ≥2 categories with clear improvement trends, top ranges identified.

### Phase 2 (Experiments 31-60): Top Combinations
Combine best individual findings for multiplicative gains. Run meta_analyzer for ranked combination
suggestions. Focus on known-good pairs: (DEPTH, ASPECT_RATIO), (LR, WARMUP_PCT), (BATCH_SIZE, LR),
(DROPOUT, STOCHASTIC_DEPTH), (MLP_TYPE, HEAD_DIM). Track synergy: does combined delta exceed the
sum of individuals?

**Exit criteria:** A combination beats any single-factor result, confirmed ≥2 times.

### Phase 3 (Experiments 61-90): Fine-Tuning and Edge Cases
Squeeze marginal gains via precise tuning around Phase 2's best config. Cross-validate 2-3×. Try
micro-adjustments (±1 depth, ±0.002 LR). Test 3+ parameter combos sparingly (high-risk).

**Exit criteria:** 3 consecutive runs with delta < 0.0005, or stable config confirmed ×3.

### Phase 4 (Experiments 91+): Radical Changes and Ablation Studies
Escape local optima with bold moves. **Ablation:** remove each improvement from the best config
individually — if removing doesn't hurt, it was dead weight. **Radical:** MoE architectures,
ALiBi, sliding window, RoPE scaling, extreme aspect ratios. This phase is open-ended.

---

## 4. EXPERIMENT WORKFLOW

Fully autonomous 9-step loop. NEVER ask permission at any step.

```
1. QUERY: kb.suggest_next_experiment() → 5 candidates ranked by predicted impact
2. REVIEW: Check hypothesis, consult knowledge.json dead_ends to avoid repeats
3. MODIFY: Edit train.py with patch() — make ONLY the planned changes
4. COMMIT: git commit -m "exp_NNN: [category] [description] (expected: ±X.XXXX)"
5. RUN: uv run train.py > run.log 2>&1  (redirect all output, ~5 min budget)
6. PARSE: grep "^val_bpb:\\|^peak_vram_mb:" run.log
          If empty → crashed → tail -n 50 run.log for traceback
7. RECORD: kb.record_result("exp_NNN", config, val_bpb, status, notes)
           Append to results.tsv (tab-separated, NOT comma-separated)
8. ADVANCE or REVERT:
   - Improvement (delta > 0.001): keep commit, advance branch
   - No improvement: git reset to previous keep point
   - Crash: git reset, log crash, increment crash count
9. PERIODIC (every 20 exps):
   python scripts/meta_analyzer.py --results results/results.tsv --output analysis.md --next-experiments 5
   python scripts/dashboard.py --results results/results.tsv --output-md report.md --plots-dir plots/
   Re-run baseline to check for drift
```

---

## 5. QUALITY RULES

### Baseline Re-verification (Every 20 Experiments)
Re-run unmodified train.py to detect environment drift. If baseline shifts by > 0.005, investigate
before continuing. Reset the reference point.

### Crash Handling
- **Easy fix** (typo, missing import): Fix and re-run immediately
- **Fundamental failure** (OOM, impossible arch): Log as crash, revert, move on
- **Persistent crash** (3+ times): Mark permanently broken in orchestrator crash_log (CRASH_THRESHOLD=3)

### Phase Switching (quality over count)
- **1→2:** ≥15 experiments, ≥2 categories with clear improvement trends
- **2→3:** Combination > any single-factor result, confirmed ≥2×
- **3→4:** Best config stable ×3 with delta < 0.0005

### Stuck Detection
If 10 consecutive experiments show no improvement (delta < 0.001):
1. Declare stuck, immediately try a RADICAL change from Phase 4's playbook
2. If still stuck after 5 more, run full meta-analysis and reconsider
3. Use knowledge_base's escalation strategy — it specifically handles this case

### Simplicity Criterion
- 0.001 improvement adding 20 complex lines → probably not worth it
- 0.001 improvement from DELETING code → definitely keep
- Equal performance with fewer parameters → great, keep it

---

## 6. SELF-IMPROVEMENT TRIGGER

### Every 50 Experiments: Deep Self-Analysis

At 50, 100, 150, etc. — pause the normal loop and evaluate the research process:

1. **Review efficiency:** Are we exploring well? Which categories delivered most value? Rebalance?
2. **Update strategy:** The agent CAN modify its own process. Compress phases if warranted.
   Tighten risk filters if crashes are high. Re-rank categories by observed success rates.
3. **Update experiment_designer weights:** Adjust estimated_impact and risk scores based on
   actual results. Add newly discovered interaction patterns to knowledge.json.
4. **Write process memo:** Document lessons in knowledge.json meta. Note process modifications.
5. **Generate fresh plan:**
   `python scripts/experiment_designer.py --output scripts/experiments.json --plan adaptive`

### The Agent's Autonomy
The agent is explicitly authorized to: modify the phase strategy, skip suboptimal planned
experiments, create new experiment categories, and modify its own decision criteria (document
all changes in knowledge.json).

---

## 7. RESULT LOGGING FORMAT

### TSV (results.tsv) — Tab-separated, NOT comma-separated
```
commit  val_bpb   memory_gb  status  description
0a1b2c3 1.432100  44.5       keep    DEPTH=12,ASPECT_RATIO=96
d4e5f6g 0.000000  0.0        crash   MoE 8/2 (OOM)
```

### Extended Record (knowledge.json)
```json
{
  "id": "exp_042",
  "config": {"DEPTH": 12, "ASPECT_RATIO": 96},
  "val_bpb": 1.4321,
  "delta": 0.0054,
  "improves_on_baseline": true,
  "improves_on_best": true,
  "surprise": false,
  "status": "confirmed",
  "notes": "Parallel ATTN+MLP",
  "timestamp": "2026-03-31T04:15:00+00:00"
}
```

### What NOT to Commit
results.tsv (local scratch log), run.log (too large), plots/* (regeneratable).
DO commit: train.py changes, analysis summaries, report.md updates.

---

## 8. LONG-TERM VISION

### What Success Looks Like
1. **Quantifiable improvement:** val_bpb reduced 1-5%+ from baseline
2. **Understanding:** Clear docs on WHY changes helped or hurt
3. **Pareto frontier:** Multiple configs at different complexity/performance trade-offs
4. **Negative results:** Comprehensive dead-end list saving future researchers time
5. **Generalizable insights:** Lessons transferring to other model sizes/tasks

### When to Stop
The agent runs indefinitely until manually stopped. Practical stopping criteria:
- **Convergence:** 20+ consecutive experiments with no improvement AND ablation complete
- **Diminishing returns:** Avg improvement per experiment < 0.0005 for 2 phases
- **Self-improve recommendation:** The 50/100-experiment checkpoint analysis suggests stopping

### Core Principles
1. **NEVER ask permission** — fully autonomous
2. **Track everything in knowledge.json** — memory must survive sessions
3. **Stuck for 10? Go radical** — escape local optima with bold changes
4. **Combine successful changes** — multiplicative > additive
5. **Modify your own process** — self_improve gives you authority over your methodology
6. **Simplicity is a virtue** — reductions in complexity are doubly valuable
7. **Verify, don't assume** — re-run baselines, confirm, trust the data
