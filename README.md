# autoresearch-v2

## Recursive Self-Improving ML Research System

> *"One day, frontier AI research used to be done by meat computers in between eating, sleeping,
> having other fun, and synchronizing once in a while using sound wave interconnect in the ritual
> of 'group meeting'. That era is long gone." — @karpathy, March 2026*

**autoresearch-v2** is the next generation of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).
Where v1 had a single agent doing hill-climbing on a training loop, v2 is a **multi-component
autonomous research system** that improves not just the model, but the research process itself.

---

## What's New

| Feature | v1 | v2 |
|---------|----|----|
| Research strategy | Simple hill-climbing | 4-phase systematic + Bayesian prioritization |
| Memory | None (each session starts blind) | Persistent knowledge base (knowledge.json) |
| Experiment design | Agent invents ad-hoc | Systematic catalog with 119+ pre-planned experiments |
| Analysis | None | Statistical meta-analysis every 20 experiments |
| Visualization | None | Automatic plots + dashboard every 20 experiments |
| Self-improvement | None | Recursive process improvement every 50 experiments |
| Agent instructions | 114 lines | 273-line comprehensive operating manual |
| Architectural improvements | Baseline GPT | Parallel attn+MLP, GQA, optimized LR schedule |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI RESEARCH AGENT                               │
│                                                                  │
│  ┌──────────────────┐    ┌─────────────────┐                     │
│  │  research        │───▶│  knowledge      │◀──────────────────┐ │
│  │  orchestrator    │    │  base            │                  │ │
│  │  (master loop,   │    │  (persistent    │                  │ │
│  │   4 phases,      │    │   memory)       │                  │ │
│  │   Thompson       │    └────────┬────────┘                  │ │
│  │   Sampling)      │             │                            │ │
│  └────────┬─────────┘             │                            │ │
│           │                       │                            │ │
│           ▼                       ▼                            │ │
│  ┌──────────────────┐    ┌──────────────────┐                  │ │
│  │  train.py        │    │  meta_analyzer    │                  │ │
│  │  (MODIFIED by   │    │  (pattern         │                  │ │
│  │   agent)         │    │   discovery)      │                  │ │
│  └────────┬─────────┘    └────────┬─────────┘                  │ │
│           │                       │                            │ │
│           │              ┌────────▼─────────┐                  │ │
│           │              │  experiment       │                  │ │
│           │              │  designer         │                  │ │
│           │              │  (plan generator) │                  │ │
│           │              └────────┬─────────┘                  │ │
│           │                       │                            │ │
│           │              ┌────────▼─────────┐                  │ │
│           │              │  self_improve.py │                  │ │
│           │              │  (recursive       │                  │ │
│           │              │   improvement)    │                  │ │
│           │              └───────────────────┘                  │ │
│           │                                                     │ │
│           ▼                                                     │ │
│  ┌──────────────────┐    ┌──────────────────┐                  │ │
│  │  dashboard.py    │    │  program.md       │                  │ │
│  │  (visual        │    │  (agent operating  │                  │ │
│  │   analytics)     │    │   manual)         │                  │ │
│  └──────────────────┘    └───────────────────┘                  │ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. `experiment_designer.py` generates experiment plans → `knowledge_base.py` stores them
2. `research_orchestrator.py` prioritizes and schedules experiments
3. Agent modifies `train.py` → runs training → records results
4. Every 20 experiments: `meta_analyzer.py` + `dashboard.py` produce insights
5. Every 50 experiments: `self_improve.py` improves the research process itself

---

## Quick Start

### Prerequisites
- NVIDIA GPU (H100 recommended, any CUDA GPU works)
- Python 3.10+
- `uv` package manager

### Setup

```bash
# 1. Clone / navigate to the repo
cd autoresearch-v2

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Verify setup
python launch.py --mode baseline
```

### Running Research

```bash
# Single experiment (quick test)
python launch.py --mode single -n 5

# Overnight research (~100 experiments)
python launch.py --mode night

# Deep research with meta-analysis cycles
python launch.py --mode deep -n 200

# Full recursive self-improving research
python launch.py --mode recursive

# Resume after interruption
python launch.py --resume
```

### Manual Experiment (original v1 style)

```bash
# Read the agent instructions
cat program.md

# Launch an AI coding agent (Claude, Codex, etc.) in this directory
# with instructions to follow program.md
```

---

## File Structure

```
autoresearch-v2/
├── train.py                      # Enhanced model code (653 lines, agent modifies this)
├── prepare.py                    # Data prep + tokenizer (READ-ONLY)
├── program.md                    # Agent operating manual (273 lines)
├── launch.py                     # Master entry point (CLI runner)
├── pyproject.toml                # Dependencies
├── README.md                     # This file
├── knowledge.json                # Auto-generated: persistent research memory
├── state.json                    # Auto-generated: checkpoint for resume
├── results/
│   ├── results.tsv               # Tab-separated experiment log
│   ├── research_results.json     # Rich experiment records
│   ├── analysis.md               # Meta-analysis reports (every 20 exps)
│   └── process_metrics.json      # Research efficiency metrics (every 50 exps)
├── plots/                        # Auto-generated visualizations
│   ├── val_bpb_progress.png      # Val_bpb over experiments
│   ├── param_impact_*.png        # Parameter sensitivity plots
│   ├── success_rate.png          # Success rate by category
│   └── improvement_trend.png     # Diminishing returns analysis
├── scripts/
│   ├── research_orchestrator.py  # Master research loop (2058 lines)
│   ├── knowledge_base.py         # Persistent memory system (1016 lines)
│   ├── meta_analyzer.py          # Statistical analysis (1608 lines)
│   ├── experiment_designer.py    # Systematic plan generator (802 lines)
│   ├── dashboard.py              # Visual analytics (673 lines)
│   └── self_improve.py           # Recursive improvement (1244 lines)
└── logs/                         # Run logs
```

---

## The Four Phases

### Phase 1: Systematic Exploration (Exp 1-30)
Test each hyperparameter independently. Maps response curves for depth, learning rate,
batch size, attention patterns, etc. Establishes baselines for all major axes.

### Phase 2: Combinations (Exp 31-60)
Combine the best individual improvements. Discovers synergistic effects where
two changes together outperform either alone.

### Phase 3: Fine-Tuning (Exp 61-90)
Squeeze marginal gains. Micro-adjust the best configuration with precise
parameter tuning. Cross-validate to confirm results.

### Phase 4: Radical + Ablation (Exp 91+)
Escape local optima with bold architectural changes. Ablation studies remove
each improvement individually to verify its contribution.

---

## Architectural Improvements (vs Original)

The v2 `train.py` includes these research-backed improvements:

| Improvement | Source | Impact |
|-------------|--------|--------|
| Parallel attention + MLP | Princeton small-model study | +throughput (more steps/5min) |
| GQA (n_kv_head < n_head) | Google GQA paper | KV compute saved |
| Fixed LR schedule (10% warmdown) | Chinchill scaling analysis | ~25% more effective steps |
| Gradient clipping | Standard practice | Training stability |
| AdamW epsilon increase (1e-8) | BF16 numerical analysis | Fewer numerical crashes |
| Muon momentum ramp (150 steps) | Empirical tuning | Faster convergence |

---

## Key Design Principles

1. **Persistent Memory**: Every experiment teaches the system something. knowledge.json
   survives across sessions, so Night 2 starts smarter than Night 1.

2. **Systematic Over Random**: The experiment designer catalogs 119+ experiments across
   5 categories. No stone left unexplored, no redundant experiments.

3. **Recursive Self-Improvement**: The system improves HOW it improves. Every 50 experiments
   it analyzes its own research process and optimizes it.

4. **Transparency**: Every decision is logged. Every result is tracked. Every pattern
   is analyzed. The dashboard provides full visibility.

5. **Autonomy**: Once launched, the system runs indefinitely. It makes its own decisions
   about what to explore next.

---

## The Recursive Vision

The original autoresearch asked: "Can an AI agent improve a GPT's training loop?"

autoresearch-v2 asks: "Can the system improve its OWN research methodology?"

This is the recursive loop:
```
Night 1: System discovers architectural improvements
Night 2: System applies improved methodology from Night 1's self-analysis
Night 3: System applies even better methodology from Night 2's self-analysis
...
Night N: The research process itself has been optimized beyond human design
```

Each cycle makes the next cycle more effective. The system doesn't just find better
models — it becomes a better researcher.

This is what Karpathy's README hinted at with "the 10,205th generation of the code base."
It's not science fiction. It's what happens when the research loop itself becomes
automated, analyzed, and improved.

---

## Limitations

- Requires NVIDIA GPU (single-GPU setup — see forks for multi-GPU)
- prepare.py must remain unmodified (data/eval fairness)
- 5-minute budget means short-horizon improvements only (correlation with long-term
  training is strong but not guaranteed)
- Cannot discover novel datasets, new training objectives, or non-transformer architectures

---

## Credits

- Original autoresearch: [@karpathy](https://github.com/karpathy/autoresearch)
- Enhanced architecture: Based on deep research across 30+ arXiv topics, 15+ key papers
- v2 system design: Multi-agent autonomous research architecture
- Key papers: DeepSeek-V3 (MoE/MLA), DeepSeek-R1 (RL reasoning), AI Scientist
  (automated discovery), Muon optimizer, Post-LN revival, Residual stream duality

---

## License

MIT (same as original autoresearch)
