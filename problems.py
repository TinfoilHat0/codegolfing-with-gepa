"""
problems.py — Auto-fetch and cache code.golf holes.

Flow per hole:
  1. GET /api/holes  →  all holes with HTML preambles
  2. Skip holes in ALWAYS_SKIP
  3. POST dummy code to code.golf judge → get real test cases (args + expected answers)
  4. Fetch human-best byte count from the rankings page
  5. Cache everything to problems_cache/{hole_id}.json

Cached entries are never re-fetched unless you delete the cache file.
Use --fix-cache to update existing cache files with real expected outputs.
"""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "problems_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Holes we always skip — judge API can't evaluate these meaningfully
ALWAYS_SKIP = {
    "quine",                            # expected output == source code (changes every time)
    "brainfuck",                        # requires a Brainfuck interpreter
    "qr-decoder",                       # requires image input
    "star-wars-gpt",                    # non-deterministic output
}

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True).strip()

# ---------------------------------------------------------------------------
# code.golf API calls
# ---------------------------------------------------------------------------

def fetch_all_hole_stubs() -> list[dict]:
    """GET /api/holes — returns all holes with id, name, preamble, category."""
    r = requests.get("https://code.golf/api/holes", timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_human_best(hole_id: str) -> int | None:
    """Scrape the minimum Python byte count from the rankings page."""
    try:
        url = f"https://code.golf/rankings/holes/{hole_id}/python/bytes"
        r = requests.get(url, timeout=10)
        m = re.search(
            r'<script[^>]+id=["\']?chart-data["\']?[^>]*>(.*?)</script>',
            r.text, re.DOTALL,
        )
        if m:
            data = json.loads(m.group(1))
            if isinstance(data, list) and data and "strokes" in data[0]:
                return min(d["strokes"] for d in data)
    except Exception:
        pass
    return None


def fetch_from_judge(hole_id: str) -> list[dict] | None:
    """
    Submit dummy code to the code.golf judge and capture one sample of test cases.

    Returns a list of run dicts, each with:
        args    list[str]  — argv items OR stdin lines for this test case
        answer  str        — expected stdout for this test case

    Three hole types (implicit in the returned structure):
        self_contained  — 1 run, args=[]         (answer is always stable)
        stdin           — 1 run, args=[l1, l2…]  (pass args joined as stdin)
        argv            — N runs, each with args  (pass args as sys.argv per run)

    Test cases for stdin/argv holes are randomised by code.golf on each call, so
    we store ONE sample during --fix-cache and use it consistently for local eval.
    The model just needs to handle the general case — if it passes our sample it
    will pass the judge's cases too.

    Returns None if the hole is unsupported or returns no meaningful output.
    """
    try:
        r = requests.post(
            "https://code.golf/solution",
            json={"hole": hole_id, "lang": "python", "code": "print(1)"},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        raw_runs = data.get("runs", [])
        if not raw_runs:
            return None
        runs = [{"args": run.get("args", []), "answer": run["answer"]} for run in raw_runs]
        if not any(rc["answer"] for rc in runs):
            return None
        return runs
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_problem(hole: dict) -> dict | None:
    """
    Build a complete problem dict for one code.golf hole.
    Returns None if the hole should be skipped or fails.
    Uses disk cache — re-run safe.
    """
    hole_id = hole["id"]
    cache_file = CACHE_DIR / f"{hole_id}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if hole_id in ALWAYS_SKIP:
        return None

    description = _strip_html(hole.get("preamble", ""))

    # Fetch one sample of test cases from the real judge
    runs = fetch_from_judge(hole_id)
    if not runs:
        return None

    # Human best score
    human_best = fetch_human_best(hole_id)
    if human_best is None:
        return None

    problem = {
        "name":         hole_id,
        "display_name": hole.get("name", hole_id),
        "category":     hole.get("category", ""),
        "description":  description,
        # runs: [{args, answer}] — one sample of test cases for local evaluation
        "runs":         runs,
        # expected: convenience alias for self-contained holes (1 run, no args)
        "expected":     runs[0]["answer"] if len(runs) == 1 and not runs[0]["args"] else None,
        "human_best":   human_best,
    }
    cache_file.write_text(json.dumps(problem, indent=2))
    return problem


def fix_problem_cache(hole_id: str) -> bool:
    """
    Re-fetch one sample of test cases from the judge for an already-cached hole
    and update the cache file in-place. Returns True if the cache was updated.
    """
    cache_file = CACHE_DIR / f"{hole_id}.json"
    if not cache_file.exists():
        return False

    problem = json.loads(cache_file.read_text())

    runs = fetch_from_judge(hole_id)
    if not runs:
        return False

    problem["runs"]     = runs
    problem["expected"] = runs[0]["answer"] if len(runs) == 1 and not runs[0]["args"] else None
    # Remove stale LLM-generated fields
    problem.pop("reference_code", None)
    problem.pop("hole_type", None)

    cache_file.write_text(json.dumps(problem, indent=2))
    return True


def load_all_problems(
    *,
    verbose: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch and cache all eligible code.golf holes.
    `limit` caps how many holes are attempted (useful for dry runs).
    Returns a list of successfully built problem dicts.
    """
    stubs = fetch_all_hole_stubs()
    problems = []
    attempted = 0
    for hole in stubs:
        if limit and attempted >= limit:
            break
        attempted += 1
        if verbose:
            print(f"  {hole['id']}...", end=" ", flush=True)
        prob = build_problem(hole)
        if prob:
            problems.append(prob)
            if verbose:
                print(f"ok  ({prob['human_best']}B)")
        else:
            if verbose:
                print("skip")
    return problems
