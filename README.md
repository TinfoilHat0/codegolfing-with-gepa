# Code Golf Agent — GEPA × code.golf

An experiment applying [GEPA](https://gepa-ai.github.io/gepa/) (`optimize_anything`) to automatically compress Python solutions on [code.golf](https://code.golf).

## How It Works

GEPA's `optimize_anything` is used in **single-task mode**: the *candidate* is the actual Python code, and the evaluator runs it against the judge's test cases and returns `human_best_bytes / code_bytes` as the score (-1.0 if wrong). GEPA's reflection LM then iteratively edits the code — shrinking it byte by byte over 100 rollouts.

```
for each problem:
    seed   = model.generate_correct_solution(problem)   # verbose but correct
    final  = GEPA.optimize(seed, evaluator=run_and_score)  # 100 rollouts of compression
```

The key insight: by making the **code itself** the candidate (rather than a meta-prompt), GEPA can make surgical edits — replacing `for` loops with comprehensions, collapsing `if/else` to ternaries, using `exec` tricks, exploiting Python's short-circuit evaluation, and so on.

## Problem Selection

We first ran `claude-sonnet-4-6` as a baseline across all **121 code.golf Python holes**, getting **59/121 correct**. From those 59 we selected the **10 where we were already closest to the human record** (`human_best / our_bytes` highest) — problems where GEPA has the least ground to cover and correctness is well-established.

The 10 problems span five categories: Computing, Gaming, Mathematics, Sequence, Transform.

## Results

All runs: **budget = 100 GEPA rollouts**, model used as both seed generator and reflection LM.

| Problem | Category | Human | Opus 4.5 | Sonnet 4.6 | Opus 4.6 |
|---|---|---|---|---|---|
| fibonacci | Sequence | **36B** | 38B (0.947) | 38B (0.947) | 64B (0.562) |
| evil-numbers | Sequence | **40B** | 51B (0.784) | **45B (0.889)** | 71B (0.563) |
| pernicious-numbers | Sequence | **40B** | 59B (0.678) | **50B (0.800)** | **50B (0.800)** |
| prime-numbers | Sequence | **40B** | 69B (0.580) | 62B (0.645) | **44B (0.909)** |
| arabic-to-roman | Transform | **124B** | 175B (0.709) | 184B (0.674) | 186B (0.667) |
| fractions | Mathematics | **73B** | 95B (0.768) | 165B (0.442) | 294B (0.248) |
| card-number-validation | Computing | **83B** | 156B (0.532) | 171B (0.485) | 187B (0.444) |
| n-queens | Gaming | **103B** | 229B (0.450) | 231B (0.446) | 218B (0.472) |
| repeating-decimals | Mathematics | **110B** | 272B (0.404) | 468B (0.235) | 355B (0.310) |
| binary-lambda-calculus | Computing | **276B** | 905B (0.305) | 745B (0.370) | ✗ |
| **Correct** | | | **10/10** | **10/10** | **9/10** |
| **Avg ratio** | | | **0.616** | **0.593** | **0.553** |

*Ratio = human_best / our_bytes. Higher = better. 1.0 = matches human record. ✗ = model couldn't generate a correct seed.*

### Highlights

- **fibonacci**: Both Opus 4.5 and Sonnet 4.6 reached **38B — just 2 bytes from the world record**
- **prime-numbers**: Opus 4.6 reached **44B (0.909)**, 4 bytes from the record
- **evil-numbers**: Sonnet 4.6 reached **45B (0.889)** — found the `bit_count()` trick
- **binary-lambda-calculus**: Sonnet solved it (745B) while Opus 4.6 couldn't even generate a correct seed — model capability differences matter on hard problems
- **Sonnet 4.6 vs Opus**: Sonnet is ~5× cheaper and competitive overall — better choice for scale

---

## Before & After Examples

### fibonacci — 64B → 38B (ratio: 0.947)
Human record: 36B. Just 2 bytes away.

```python
# Seed (64B) — correct, readable
a, b = 0, 1
for _ in range(31):
    print(a)
    a, b = b, a + b

# GEPA final (38B) — exec loop trick
a=0;b=1;exec("print(a);a,b=b,a+b;"*31)
```

---

### pernicious-numbers — 257B → 50B (ratio: 0.800)
A pernicious number has a prime popcount. GEPA realised the valid popcounts in range(51) are just {2,3,5} and encoded them as a bitmask.

```python
# Seed (257B) — full is_prime() + popcount functions
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

for num in range(51):
    if is_prime(bin(num).count('1')):
        print(num)

# GEPA final (50B) — bitmask encodes which popcounts are prime
[print(n)for n in range(51)if 44>>n.bit_count()&1]
```

*`44` in binary is `101100` — bits 2, 3, 5 are set, exactly the primes ≤ 6.*

---

### evil-numbers — 71B → 45B (ratio: 0.889)
An evil number has even popcount. GEPA found Python's `or` short-circuit as a compact conditional print.

```python
# Seed (71B)
for n in range(51):
    if bin(n).count('1') % 2 == 0:
        print(n)

# GEPA final (45B) — short-circuit or, bit_count() builtin
for n in range(51):n.bit_count()%2or print(n)
```

---

### binary-lambda-calculus — 3369B → 745B (ratio: 0.370)
A full binary lambda calculus normaliser. GEPA can't match the human record here (276B — a tour de force), but still compressed a 3369B class-based implementation by 78%.

```python
# Seed (3369B) — full OOP implementation with Lam/App/Var classes,
#                shift/subst/normalize/encode functions

# GEPA final (745B) — same algorithm, golfed with tuples instead of classes,
#                     single-letter names, collapsed control flow
import sys
def P(s,i):
 if s[i:i+2]<'01':b,i=P(s,i+2);return(0,b),i
 if s[i]<'1':f,i=P(s,i+2);a,i=P(s,i);return(1,f,a),i
 c=0
 while s[i]>'0':c+=1;i+=1
 return(2,c),i+1
def S(t,d,c=0):
 if t[0]==2:return(2,t[1]+d)if t[1]>c else t
 if t[0]==0:return(0,S(t[1],d,c+1))
 return(1,S(t[1],d,c),S(t[2],d,c))
def U(t,i,v):
 if t[0]==2:return v if t[1]==i else(2,t[1]-1)if t[1]>i else t
 if t[0]==0:return(0,U(t[1],i+1,S(v,1)))
 return(1,U(t[1],i,v),U(t[2],i,v))
def N(t):
 while t[0]==1:
  f=N(t[1])
  if f[0]!=0:return(1,f,N(t[2]))
  t=U(f[1],1,t[2])
 if t[0]==0:return(0,N(t[1]))
 return t
def E(t):
 if t[0]==0:return'00'+E(t[1])
 if t[0]==1:return'01'+E(t[1])+E(t[2])
 return'1'*t[1]+'0'
for l in sys.stdin:
 l=l.strip()
 if l:print(E(N(P(l,0)[0])))
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# 1. Fetch all problems from code.golf (run once)
python main.py --fetch

# 2. Find which problems the model solves correctly
python main.py --baseline --val-n 121 --gen-model sonnet

# 3. Run GEPA compression on target problems
python main.py --compress-direct \
    --correct-subset logs/correct_baseline_holes.json \
    --val-n 10 --budget 100 \
    --gen-model sonnet --reflection-model sonnet
```

**`--gen-model` / `--reflection-model` options:** `haiku`, `sonnet`, `opus45`, `opus46`

## Key Files

| File | Description |
|---|---|
| `main.py` | Fetch problems, run baseline, run GEPA compression |
| `problems.py` | Loads and caches code.golf problem specs |
| `logs/compress_direct_20260305_055718.json` | Opus 4.5 results |
| `logs/compress_direct_20260305_062752.json` | Opus 4.6 results |
| `logs/compress_direct_20260306_015419.json` | Sonnet 4.6 results |
| `logs/correct_baseline_holes.json` | 59 problems Sonnet solves correctly |
| `logs/top10_targets.json` | The 10 selected target problems |
