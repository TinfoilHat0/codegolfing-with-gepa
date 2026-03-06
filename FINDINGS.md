# Code Golf Agent — Findings

## Overview

We built an agent targeting [code.golf](https://code.golf) Python holes, using GEPA's
`optimize_anything` library as the optimization backbone. The key experiment was comparing
two fundamentally different strategies for producing golfed code.

---

## Two Optimization Strategies

### Strategy A: Prompt Optimization (Generalization Mode)

Optimize a *system prompt* (the "teacher") that a model uses to generate code across all
problems simultaneously. GEPA's Pareto frontier tracks correctness vs. compression.

```
Candidate = system_prompt
Evaluator: system_prompt → Claude generates code → code runs → score
GEPA optimizes: a single prompt that transfers across unseen problems
```

**Results (sonnet gen, sonnet reflection, budget=50, 15 train / 10 val):**

| | Score |
|---|---|
| Seed val score | -0.233 |
| Final val score | -0.161 |
| Improvement | +0.072 |
| Val problems correct | 6/10 |

The optimized prompt grows into a rich golf cheat sheet (walrus, comprehensions, sieve
patterns, etc.) that GEPA discovers through reflection. Helpful but limited:
**aggressive golf hints cause correctness regressions** — the model tries to compress
before it fully understands the problem.

### Strategy B: Direct Code Optimization (Single-Task Mode)  ← winner

Optimize the *actual code* for each problem. GEPA's reflection LM directly edits code
bytes, not a meta-prompt. Each problem gets its own optimization budget.

```
Candidate = python_code  (the artifact itself)
Evaluator: run code → score + diagnostics (bytes, expected vs actual output, errors)
GEPA optimizes: one code solution per problem
```

**Results (opus seed, sonnet reflection, budget=30/problem, 10 problems):**

| Problem | Seed | Final | Human | Ratio | Δ from seed |
|---|---|---|---|---|---|
| evil-numbers | 83B | **45B** | 40B | **0.889** | +46% |
| fractions | 276B | **128B** | 73B | **0.570** | +54% |
| arabic-to-roman | 550B | **218B** | 124B | **0.569** | +60% |
| n-queens | 1278B | **200B** | 103B | **0.515** | +84% |
| card-number-validation | 713B | **167B** | 83B | **0.497** | +77% |
| repeating-decimals | 1613B | **286B** | 110B | **0.385** | +82% |
| binary-lambda-calculus | 2941B | **760B** | 276B | **0.363** | +74% |
| mandelbrot *(wrong seed)* | 429B | — | 103B | — | — |
| mahjong *(wrong seed)* | 3287B | — | 174B | — | — |
| sierpiński-triangle *(wrong seed)* | 1171B | — | 57B | — | — |

**Avg compression ratio (correct only): 0.190 → 0.541 (+0.351)**

This is dramatically better. GEPA can see the actual code and make surgical edits —
removing `abs()`, switching `if/else` to `"-"*bool`, collapsing nested loops into
comprehensions, etc.

---

## Generation Model Comparison

Tested sonnet vs opus as the *seed generator* (before any optimization):

| Model | Seed val score | Val correct | Note |
|---|---|---|---|
| claude-sonnet-4-6 | -0.233 | ~7/10 | Fails harder problems |
| claude-opus-4-5 | **+0.026** | **8–10/10** | Correct but verbose |

**Opus is better at correctness; sonnet benefits more from GEPA prompt optimization.**
With direct code compression (Strategy B), opus is the better seed generator because
it reliably produces *correct* seeds, and GEPA then handles the compression.

---

## Key Insight: Data Generation vs. Teacher Distillation

**Prompt optimization** (Strategy A) trains a "teacher prompt" for distillation:
- Useful for few-shot inference at scale
- Limited by the model's willingness to sacrifice brevity for correctness
- The prompt generalizes, but compression quality is modest

**Direct code optimization** (Strategy B) generates high-quality *training data*:
- Each GEPA run produces a (problem, golfed_code) pair
- No inference-time cost — just store the best code per problem
- Can fine-tune a small model on these pairs: `SFT(problem → golfed_code)`
- Far higher compression ratios than prompt optimization achieves

This reframes the task: instead of building a better code-golf *inference agent*,
we're using GEPA as a *data synthesis engine* that produces compressed code
examples which can then train any model.

---

## Eval Robustness Fix

The code.golf judge strips trailing whitespace from **every output line** before
comparing. Our initial eval only did `rstrip("\n")` on the full output, causing false
negatives when code emitted trailing spaces on intermediate lines.

Fix: `"\n".join(line.rstrip() for line in output.splitlines())`

Impact: +1 problem recovered (58 → 59 correct in baseline).

---

## Baseline Correctness (no golfing, claude-sonnet-4-6)

| Category | Correct | Total | % |
|---|---|---|---|
| Sequence | 36 | 42 | 86% |
| Computing | 5 | 8 | 63% |
| Gaming | 3 | 6 | 50% |
| Art | 7 | 18 | 39% |
| Mathematics | 7 | 22 | 32% |
| Transform | 1 | 25 | 4% |
| **Total** | **59** | **121** | **49%** |

Transform failures: the judge expects Unicode block characters (`▄`) for morse code,
not dots/dashes. Math failures: edge cases in number theory (Jacobi symbol, continued
fractions). Art failures: precise grid dimensions in ASCII art.

---

## Optimization Stack

```
optimize_anything (GEPA)
├── Generalization mode   → prompt optimization (Strategy A)
├── Single-task mode      → direct code optimization (Strategy B) ← best
├── SideInfo / ASI        → richer diagnostics = better reflection LM proposals
│   ├── "scores": {"correctness": ..., "compression": ...}  ← Pareto tracking
│   ├── "Expected" / "Output" / "Error"                     ← diagnostic context
│   └── background param  → domain constraints for reflection LM
└── Pareto frontier       → preserves both correct+compressed candidates
```

---

## Next Steps

1. **Scale Strategy B across all 59 correct holes** — run `--compress-direct` on the
   full correct subset with budget=50/problem. This generates a dataset of high-quality
   golfed solutions.

2. **Fix seed failures** — mandelbrot, mahjong, sierpiński-triangle fail with Opus.
   These need problem-specific prompting (e.g., mandelbrot requires specific Unicode
   character `░▒▓█`; sierpiński needs bitwise AND).

3. **SFT on generated data** — fine-tune a small model (haiku or a distilled model)
   on the (problem, golfed_code) pairs from Strategy B. This is the most promising
   path to a capable inference-time code golf agent.

4. **Increase budget** — GEPA improved evil-numbers from 83B → 45B in 30 rollouts
   (human best: 40B). With budget=100 it likely closes that gap further.
