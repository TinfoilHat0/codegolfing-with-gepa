"""
main.py — Code golf agent using GEPA's optimize_anything.

Pipeline:
    1. Fetch problems from code.golf:
          python main.py --fetch

    2. Find which problems the model can solve correctly (baseline):
          python main.py --baseline --val-n 121

    3. Compress solutions directly with GEPA (single-task mode):
          python main.py --compress-direct \\
              --correct-subset logs/correct_baseline_holes.json \\
              --val-n 10 --budget 100 \\
              --gen-model opus46 --reflection-model opus46
"""

import argparse
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import dotenv_values

_env = dotenv_values(Path(__file__).parent / ".env")
if "ANTHROPIC_API_KEY" in _env:
    os.environ["ANTHROPIC_API_KEY"] = _env["ANTHROPIC_API_KEY"]

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    MergeConfig,
    ReflectionConfig,
    optimize_anything,
)
from problems import load_all_problems

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

GEN_MODEL_HAIKU    = "claude-haiku-4-5-20251001"
GEN_MODEL_SONNET   = "claude-sonnet-4-6"
GEN_MODEL_OPUS_45  = "claude-opus-4-5"
GEN_MODEL_OPUS_46  = "claude-opus-4-6"

REFLECTION_HAIKU   = f"anthropic/{GEN_MODEL_HAIKU}"
REFLECTION_SONNET  = "anthropic/claude-sonnet-4-6"
REFLECTION_OPUS_45 = "anthropic/claude-opus-4-5"
REFLECTION_OPUS_46 = "anthropic/claude-opus-4-6"

_GEN_MODEL_MAP = {
    "haiku":  GEN_MODEL_HAIKU,
    "sonnet": GEN_MODEL_SONNET,
    "opus45": GEN_MODEL_OPUS_45,
    "opus46": GEN_MODEL_OPUS_46,
}
_REFLECTION_MAP = {
    "haiku":  REFLECTION_HAIKU,
    "sonnet": REFLECTION_SONNET,
    "opus45": REFLECTION_OPUS_45,
    "opus46": REFLECTION_OPUS_46,
}

# ---------------------------------------------------------------------------
# Code execution helpers
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\n?(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()


def _normalize(s: str) -> str:
    """Match code.golf judge: strip trailing whitespace from every line."""
    return "\n".join(line.rstrip() for line in s.splitlines())


def run_code(code: str, runs: list[dict] | None = None, timeout: int = 10) -> tuple[str, str, bool]:
    """
    Execute code against test cases. Returns (stdout, stderr, correct).

    Handles three hole types via the `runs` structure:
        self_contained  1 run, args=[]   — no input
        stdin           1 run, args=[…]  — args joined as stdin lines
        argv            N runs           — one subprocess call per test case
    """
    _runs = runs or [{"args": [], "answer": ""}]

    if len(_runs) == 1:
        run         = _runs[0]
        args        = run.get("args", [])
        answer      = run.get("answer", "")
        stdin_input = "\n".join(args) if args else None
        try:
            r = subprocess.run(
                ["python3", "-c", code],
                input=stdin_input,
                capture_output=True, text=True, timeout=timeout,
            )
            return r.stdout, r.stderr, _normalize(r.stdout) == _normalize(answer)
        except subprocess.TimeoutExpired:
            return "", "TimeoutError", False

    # argv holes: one subprocess per test case
    outputs = []
    for run in _runs:
        args   = run.get("args", [])
        answer = run.get("answer", "")
        try:
            r = subprocess.run(
                ["python3", "-c", code] + args,
                capture_output=True, text=True, timeout=timeout,
            )
            if _normalize(r.stdout) != _normalize(answer):
                return r.stdout, r.stderr, False
            outputs.append(r.stdout)
        except subprocess.TimeoutExpired:
            return "", "TimeoutError", False
    return "".join(outputs), "", True


# ---------------------------------------------------------------------------
# Baseline correctness eval
# ---------------------------------------------------------------------------

BASELINE_SYSTEM = (
    "Write correct Python 3 code to solve this programming problem. "
    "Output only the raw code with no explanation, no markdown, and no code fences."
)


def run_baseline(problems: list[dict], model: str) -> list[dict]:
    """Run a plain correctness check (no golfing) across all problems."""
    print(f"\nBaseline correctness eval: {len(problems)} problems, model={model}")
    print("=" * 60)
    results = []
    for prob in problems:
        resp = client.messages.create(
            model=model, max_tokens=2048, temperature=0,
            system=BASELINE_SYSTEM,
            messages=[{"role": "user", "content": prob["description"]}],
        )
        code = extract_code(resp.content[0].text)
        _, stderr, correct = run_code(code, prob.get("runs"))
        nbytes = len(code.encode())
        score  = prob["human_best"] / nbytes if correct else -1.0
        symbol = "✓" if correct else "✗"
        print(f"  {symbol} {prob['name']}: "
              + (f"{nbytes}B  (ratio {score:.3f}, human {prob['human_best']}B)" if correct else "WRONG"))
        results.append({
            "name": prob["name"], "category": prob.get("category", ""),
            "correct": correct, "bytes": nbytes,
            "human_best": prob["human_best"], "score": round(score, 4),
        })

    correct_n = sum(r["correct"] for r in results)
    print(f"\nCorrect: {correct_n}/{len(results)} ({correct_n/len(results)*100:.0f}%)")
    by_cat = Counter(r["category"] for r in results if r["correct"])
    tot_cat = Counter(r["category"] for r in results)
    for cat in sorted(tot_cat):
        print(f"  {cat}: {by_cat.get(cat, 0)}/{tot_cat[cat]}")

    correct_holes = [r["name"] for r in results if r["correct"]]
    by_cat_dict: dict[str, list[str]] = {}
    for r in results:
        if r["correct"]:
            by_cat_dict.setdefault(r["category"], []).append(r["name"])

    out = LOGS_DIR / f"baseline_correct_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"correct_holes": correct_holes, "by_category": by_cat_dict}, indent=2))
    print(f"  Saved → {out.name}")
    return results


# ---------------------------------------------------------------------------
# Direct code compression via GEPA (single-task mode)
# ---------------------------------------------------------------------------

COMPRESS_OBJECTIVE = (
    "Minimise the byte count of the Python code while keeping its output exactly correct. "
    "Score = human_record_bytes / code_bytes when correct (higher = better; >1.0 beats the record); "
    "-1.0 when wrong. "
    "Apply Python golf techniques: list/dict/set comprehensions, walrus operator (:=), "
    "ternary expressions, omit spaces around operators, short variable names, "
    "semicolons, `from math import*`, exec-loop tricks, etc. "
    "Never sacrifice correctness — a wrong answer scores -1.0 regardless of length."
)


def _generate_seed_code(problem: dict, gen_model: str) -> str:
    """Ask the model for a correct (non-golfed) solution as the GEPA seed."""
    resp = client.messages.create(
        model=gen_model, max_tokens=2048, temperature=0,
        system=BASELINE_SYSTEM,
        messages=[{"role": "user", "content": problem["description"]}],
    )
    return extract_code(resp.content[0].text)


def compress_direct(
    problems: list[dict],
    budget: int,
    gen_model: str,
    reflection_lm: str,
    out_path: Path,
) -> list[dict]:
    """
    For each problem:
      1. Generate a correct seed solution with `gen_model`.
      2. Use optimize_anything (single-task mode) to iteratively compress it.
         The *candidate* IS the Python code — GEPA's reflection LM edits it directly.
    """
    results: list[dict] = []
    print(f"\nDirect compression: {len(problems)} problems, budget={budget}, "
          f"gen={gen_model}, reflection={reflection_lm}")
    print("=" * 60)

    for prob in problems:
        name       = prob["name"]
        runs       = prob.get("runs")
        human_best = prob["human_best"]
        print(f"\n[{name}]  human_best={human_best}B")

        # Step 1 — seed
        seed_code = _generate_seed_code(prob, gen_model)
        _, _, seed_correct = run_code(seed_code, runs)
        seed_bytes = len(seed_code.encode())
        seed_score = human_best / seed_bytes if seed_correct else -1.0
        print(f"  Seed: {'✓' if seed_correct else '✗'}  {seed_bytes}B  score={seed_score:.3f}")

        if not seed_correct:
            print(f"  Seed is wrong — skipping {name}")
            results.append({
                "hole": name, "category": prob.get("category", ""),
                "gen_model": gen_model, "reflection_lm": reflection_lm, "budget": budget,
                "seed_correct": False, "seed_bytes": seed_bytes,
                "final_correct": False, "final_bytes": seed_bytes,
                "seed_score": seed_score, "final_score": seed_score,
                "human_best": human_best,
                "seed_code": seed_code, "final_code": seed_code,
                "delta_bytes": 0, "all_candidates": [],
            })
            continue

        # Step 2 — per-problem evaluator (candidate = code string)
        def make_code_evaluator(prob_runs, prob_human_best, prob_name):
            def evaluate_code(code: str, _example=None) -> tuple[float, dict]:
                stdout, stderr, correct = run_code(code, prob_runs)
                nbytes = len(code.encode())
                score  = prob_human_best / nbytes if correct else -1.0
                expected = (prob_runs or [{}])[0].get("answer", "")[:200]
                return score, {
                    "scores": {
                        "correctness": 1.0 if correct else 0.0,
                        "compression": score if correct else 0.0,
                    },
                    "Problem": prob_name,
                    "Status":  "CORRECT ✓" if correct else "WRONG ✗",
                    "Bytes":   f"{nbytes}B (human best: {prob_human_best}B, ratio: {score:.3f})" if correct else f"{nbytes}B",
                    "Output":  stdout[:300] if stdout else "(empty)",
                    "Expected": expected,
                    "Error":   stderr[:300] if stderr else None,
                }
            return evaluate_code

        # Step 3 — optimize_anything in single-task mode
        result = optimize_anything(
            seed_candidate=seed_code,
            evaluator=make_code_evaluator(runs, human_best, name),
            dataset=[{"_dummy": True}],
            objective=COMPRESS_OBJECTIVE,
            background=(
                f"Problem: {name}\n"
                f"Category: {prob.get('category', '')}\n"
                f"Human record: {human_best}B\n"
                f"Seed code ({seed_bytes}B):\n{seed_code}\n\n"
                "Environment: Python 3.11, stdlib only. "
                "Run via `python3 -c <code>`. "
                "Judge strips trailing whitespace from each output line."
            ),
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=budget, display_progress_bar=True),
                reflection=ReflectionConfig(reflection_lm=reflection_lm, reflection_minibatch_size=1),
                merge=MergeConfig(max_merge_invocations=2, merge_val_overlap_floor=1),
            ),
        )

        best_code = result.best_candidate
        _, _, final_correct = run_code(best_code, runs)
        final_bytes = len(best_code.encode())
        final_score = human_best / final_bytes if final_correct else -1.0
        delta_bytes = seed_bytes - final_bytes
        delta_pct   = delta_bytes / seed_bytes * 100 if seed_bytes else 0
        print(f"  Final: {'✓' if final_correct else '✗'}  {final_bytes}B  "
              f"score={final_score:.3f}  ({delta_pct:+.0f}% from seed, "
              f"{final_bytes / human_best:.1f}x human)")

        # Collect all GEPA candidates (useful for SFT dataset)
        str_key = result._str_candidate_key
        all_candidates = []
        for idx, (cand_dict, score) in enumerate(
            zip(result.candidates, result.val_aggregate_scores)
        ):
            code = cand_dict.get(str_key, str(cand_dict)) if str_key else str(cand_dict)
            all_candidates.append({
                "idx": idx, "code": code,
                "bytes": len(code.encode()),
                "score": round(score, 4),
                "is_best": idx == result.best_idx,
            })

        results.append({
            "hole": name, "category": prob.get("category", ""),
            "gen_model": gen_model, "reflection_lm": reflection_lm, "budget": budget,
            "seed_correct": seed_correct, "seed_bytes": seed_bytes,
            "final_correct": final_correct, "final_bytes": final_bytes,
            "seed_score": round(seed_score, 4), "final_score": round(final_score, 4),
            "human_best": human_best,
            "seed_code": seed_code, "final_code": best_code,
            "delta_bytes": delta_bytes,
            "all_candidates": all_candidates,
        })

    # Summary table
    correct_results = [r for r in results if r["final_correct"]]
    print("\n" + "=" * 60)
    print("COMPRESSION SUMMARY")
    print("=" * 60)
    print(f"Correct: {len(correct_results)}/{len(results)}")
    if correct_results:
        avg_ratio = sum(r["final_score"] for r in correct_results) / len(correct_results)
        avg_seed  = sum(r["seed_score"]  for r in correct_results) / len(correct_results)
        print(f"Avg ratio (correct only): seed={avg_seed:.3f} → final={avg_ratio:.3f}")
    print(f"\n{'Hole':<35} {'Seed':>7} {'Final':>7} {'Human':>7} {'Ratio':>6}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: -x["final_score"]):
        mark = "✓" if r["final_correct"] else "✗"
        print(f"  {mark} {r['hole']:<33} {r['seed_bytes']:>5}B  {r['final_bytes']:>5}B  "
              f"{r['human_best']:>5}B  {r['final_score']:>5.3f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Code golf agent: fetch problems and compress solutions via GEPA."
    )

    # --- Data pipeline ---
    parser.add_argument("--fetch",    action="store_true", help="Fetch all code.golf holes into problems_cache/")
    parser.add_argument("--fetch-n",  type=int, default=None, help="Limit --fetch to N holes (for testing)")
    parser.add_argument("--fix-cache", action="store_true", help="Re-fetch expected outputs for cached holes")
    parser.add_argument("--baseline", action="store_true",
                        help="Correctness-only baseline eval; saves correct holes JSON for --correct-subset")

    # --- Compression ---
    parser.add_argument("--compress-direct", action="store_true",
                        help="Run GEPA direct compression on problems (single-task mode per problem)")
    parser.add_argument("--correct-subset", type=str, default=None,
                        help="JSON file listing which holes to use (from --baseline output)")

    # --- Common options ---
    parser.add_argument("--val-n",   type=int, default=10, help="Number of problems to compress (default: 10)")
    parser.add_argument("--train-n", type=int, default=0,  help="(Unused for compress-direct; kept for compatibility)")
    parser.add_argument("--budget",  type=int, default=100, help="GEPA rollout budget per problem (default: 100)")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed for problem sampling")
    parser.add_argument(
        "--gen-model", type=str, default="opus46",
        choices=list(_GEN_MODEL_MAP.keys()),
        help="Model used to generate the seed solution (default: opus46)",
    )
    parser.add_argument(
        "--reflection-model", type=str, default="opus46",
        choices=list(_REFLECTION_MAP.keys()),
        help="GEPA reflection/optimizer model (default: opus46)",
    )

    args = parser.parse_args()

    gen_model    = _GEN_MODEL_MAP[args.gen_model]
    reflection   = _REFLECTION_MAP[args.reflection_model]

    # --fetch
    if args.fetch:
        n = args.fetch_n
        print(f"Fetching {'first ' + str(n) if n else 'all'} code.golf holes...")
        problems = load_all_problems(verbose=True, limit=n)
        print(f"Cached {len(problems)} holes.")
        return

    # --fix-cache
    if args.fix_cache:
        from problems import fix_problem_cache, CACHE_DIR
        hole_ids = [f.stem for f in sorted(CACHE_DIR.glob("*.json"))]
        print(f"Fixing {len(hole_ids)} cached holes...")
        ok = fail = 0

        def _fix(hole_id):
            return hole_id, fix_problem_cache(hole_id)

        with ThreadPoolExecutor(max_workers=16) as pool:
            for hole_id, updated in (f.result() for f in as_completed(
                pool.submit(_fix, h) for h in hole_ids
            )):
                print(f"  {hole_id}... {'ok' if updated else 'skip'}", flush=True)
                if updated: ok += 1
                else: fail += 1
        print(f"Updated {ok}, skipped {fail}.")
        return

    # Load problem cache
    cache_files = sorted(Path("problems_cache").glob("*.json"))
    if not cache_files:
        print("No cached problems found. Run with --fetch first.")
        return
    all_problems = [json.loads(f.read_text()) for f in cache_files]
    print(f"Loaded {len(all_problems)} cached holes.")

    # --baseline
    if args.baseline:
        target = all_problems[:args.val_n] if args.val_n else all_problems
        run_baseline(target, model=gen_model)
        return

    # --correct-subset filter
    if args.correct_subset:
        subset = json.loads(Path(args.correct_subset).read_text())
        correct_names = set(subset["correct_holes"])
        all_problems = [p for p in all_problems if p["name"] in correct_names]
        print(f"Filtered to {len(all_problems)} correct-baseline holes.")

    # Stratified sampling by category
    rng = random.Random(args.seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in all_problems:
        by_cat[p.get("category", "Unknown")].append(p)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    cats = sorted(by_cat.keys())

    def _stratified_sample(n: int, exclude: set) -> list[dict]:
        result, idx = [], 0
        while len(result) < n:
            cat  = cats[idx % len(cats)]
            idx += 1
            pool = [p for p in by_cat[cat] if p["name"] not in exclude]
            if pool:
                chosen = pool[0]
                by_cat[cat].remove(chosen)
                exclude.add(chosen["name"])
                result.append(chosen)
            if idx > len(all_problems) * 2:
                break
        return result

    used: set[str] = set()
    val = _stratified_sample(args.val_n, used)

    print(f"Problems ({len(val)}): {[p['name'] for p in val]}")
    print(f"  categories: {dict(Counter(p.get('category', '?') for p in val))}")

    # --compress-direct
    if args.compress_direct:
        compress_direct(
            problems      = val,
            budget        = args.budget,
            gen_model     = gen_model,
            reflection_lm = reflection,
            out_path      = LOGS_DIR / f"compress_direct_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
