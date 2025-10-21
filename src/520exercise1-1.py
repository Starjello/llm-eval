#!/usr/bin/env python3
"""
HumanEval+ pass@k evaluator (simple, Windows-safe, no emojis).

What this does:
- Loads HumanEval+ problems with evalplus
- Samples N problems (default 20)
- For each problem:
  * Saves the prompt to prompts/<task_id>.txt
  * Calls OpenAI to generate code (K attempts)
  * Precomputes expected outputs via the canonical solution
  * Runs candidate code against base+plus inputs
  * Saves each attempt to generated_solutions/<task_id>_attempt<N>.py
- Writes a run summary to heplus_passk_results.json

Requirements:
  pip install openai evalplus
  Set OPENAI_API_KEY (see README / comments)
"""

import os, sys, re, json, time, random, argparse, tempfile, subprocess, shutil, textwrap, importlib.util, types
from typing import Any, Dict, List, Tuple
from evalplus.data import get_human_eval_plus

# ---------------- Configuration ----------------
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6         # >0 for diversity
MAX_TOKENS = 800
K = 3                     # pass@k attempts per problem
NUM_PROBLEMS = 20         # how many problems to sample

TEST_TIMEOUT = 10         # seconds for executing tests
SAVE_DIR = "generated_solutions"
PROMPT_DIR = "prompts"
RESULTS_JSON = "heplus_passk_results.json"
random.seed(1234)         # for reproducible sampling order
# ------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

# ---------------- OpenAI ----------------
def ask_openai(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "Write correct, minimal Python 3 code only. No explanations or markdown fences."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    return text

# ---------------- Helpers ----------------
def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._/-]+", "_", name.strip())

def save_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_attempt(task_id: str, attempt_idx: int, code: str) -> str:
    base = _sanitize(task_id).replace("/", "_")
    path = os.path.abspath(os.path.join(SAVE_DIR, f"{base}_attempt{attempt_idx}.py"))
    save_text(path, code.strip() or "# Empty response from model.\n")
    return path

def load_module_from_code(name: str, code: str) -> types.ModuleType:
    """Load code into a fresh module object."""
    mod = types.ModuleType(name)
    exec(compile(code, filename=f"<{name}>", mode="exec"), mod.__dict__)
    return mod

def compute_expected_outputs(canonical_code: str, entry_point: str, inputs: List[Any]) -> List[Any]:
    """Run the canonical solution to compute expected outputs for given inputs."""
    mod = load_module_from_code("canonical", canonical_code)
    fn = getattr(mod, entry_point)
    expected = []
    for args in inputs:
        if isinstance(args, (list, tuple)):
            expected.append(fn(*args))
        elif isinstance(args, dict):
            expected.append(fn(**args))
        else:
            # Some HE+ inputs are singletons (e.g., int/str); pass directly
            expected.append(fn(args))
    return expected

def build_runner(solution_path: str,
                 entry_point: str,
                 base_args: List[Any],
                 base_expected: List[Any],
                 plus_args: List[Any],
                 plus_expected: List[Any]) -> str:
    """
    Emit a test runner script that:
      - imports solution from solution_path,
      - calls entry_point on each base/plus input,
      - compares to expected outputs (already computed),
      - prints a summary and exits 0 on full pass.
    """
    # Serialize inputs and expected as Python literals
    def lit(x): return repr(x)
    base_cases = [(a, e) for a, e in zip(base_args, base_expected)]
    plus_cases = [(a, e) for a, e in zip(plus_args, plus_expected)]

    run_code = f"""\
import importlib.util, sys

# Load candidate solution
spec = importlib.util.spec_from_file_location("solution", r"{solution_path}")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
fn = getattr(solution, "{entry_point}")

# Test cases (args, expected) for base and plus
BASE_CASES = {[(a, e) for a, e in base_cases]}
PLUS_CASES = {[(a, e) for a, e in plus_cases]}

def call_fn(args):
    if isinstance(args, (list, tuple)):
        return fn(*args)
    elif isinstance(args, dict):
        return fn(**args)
    else:
        return fn(args)

failures = []

for i, (args, exp) in enumerate(BASE_CASES):
    try:
        out = call_fn(args)
        if out != exp:
            failures.append(("base", i, args, out, exp))
    except Exception as e:
        failures.append(("base", i, args, f"EXC: {{e}}", exp))

for i, (args, exp) in enumerate(PLUS_CASES):
    try:
        out = call_fn(args)
        if out != exp:
            failures.append(("plus", i, args, out, exp))
    except Exception as e:
        failures.append(("plus", i, args, f"EXC: {{e}}", exp))

if not failures:
    print("PASS: all base+plus tests passed.")
    sys.exit(0)
else:
    print("FAIL: {{}} failing case(s).".format(len(failures)))
    # print up to first 10 failures for brevity
    for rec in failures[:10]:
        kind, idx, args, out, exp = rec
        print(f"[{{kind}}][case {{idx}}] args={{args}}  got={{out}}  expected={{exp}}")
    sys.exit(1)
"""
    return run_code

def run_test_runner(runner_code: str, timeout_sec: int) -> Dict[str, Any]:
    """Execute the generated test runner in an isolated subprocess."""
    tmpdir = tempfile.mkdtemp(prefix="heplus_")
    test_path = os.path.join(tmpdir, "run_tests.py")
    save_text(test_path, runner_code)
    try:
        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", "-I", test_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "code": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "elapsed": time.time() - start,
        }
    except subprocess.TimeoutExpired:
        return {"code": 124, "stdout": "", "stderr": f"TIMEOUT after {timeout_sec}s", "elapsed": timeout_sec}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------- Main eval flow ----------------
def evaluate_problem(task_id: str, prob: Dict[str, Any], k: int) -> Dict[str, Any]:
    """
    For a single HumanEval+ problem:
      - save prompt
      - compute expected outputs via canonical solution
      - generate K attempts and test each
    """
    prompt_text = prob["prompt"]
    entry = prob["entry_point"]
    canonical = prob["canonical_solution"]
    base_inputs = prob["base_input"]
    plus_inputs = prob["plus_input"]

    # Save prompt
    prompt_path = os.path.abspath(os.path.join(PROMPT_DIR, f"{_sanitize(task_id)}.txt"))
    save_text(prompt_path, prompt_text)

    # Precompute expected outputs once
    try:
        base_expected = compute_expected_outputs(canonical, entry, base_inputs)
        plus_expected = compute_expected_outputs(canonical, entry, plus_inputs)
    except Exception as e:
        return {
            "task_id": task_id,
            "error": f"Failed to compute expected outputs: {e}",
            "passed": False,
            "attempts": [],
        }

    attempts = []
    passed_any = False
    first_pass_attempt = None

    for i in range(1, k + 1):
        # Get candidate code
        try:
            code = ask_openai(prompt_text)
        except Exception as e:
            attempts.append({
                "i": i, "status": "api_error", "error": str(e)
            })
            continue

        # Save code to .py
        sol_path = save_attempt(task_id, i, code)

        # Build & run the test runner that compares to expected
        runner_src = build_runner(sol_path, entry, base_inputs, base_expected, plus_inputs, plus_expected)
        res = run_test_runner(runner_src, TEST_TIMEOUT)

        att = {
            "i": i,
            "solution_path": sol_path,
            "exit_code": res["code"],
            "stdout": res["stdout"][-1200:],
            "stderr": res["stderr"][-1200:],
            "elapsed": res["elapsed"],
        }
        attempts.append(att)

        if res["code"] == 0 and not passed_any:
            passed_any = True
            first_pass_attempt = i
            break  # early stop once we have a pass

    return {
        "task_id": task_id,
        "entry_point": entry,
        "prompt_path": prompt_path,
        "passed": passed_any,
        "first_pass_attempt": first_pass_attempt,
        "attempts": attempts,
        "n_base": len(base_inputs),
        "n_plus": len(plus_inputs),
    }

def main():
    parser = argparse.ArgumentParser(description="HumanEval+ pass@k evaluator (simple).")
    parser.add_argument("--k", type=int, default=K, help="Attempts per problem (pass@k).")
    parser.add_argument("--n", type=int, default=NUM_PROBLEMS, help="Number of problems to evaluate.")
    parser.add_argument("--all", action="store_true", help="Evaluate all HumanEval+ problems.")
    args = parser.parse_args()

    # Load all problems
    problems = get_human_eval_plus()  # dict: task_id -> problem dict
    task_ids = sorted(problems.keys())

    if args.all:
        selected = task_ids
    else:
        selected = random.sample(task_ids, k=min(args.n, len(task_ids)))

    print(f"Evaluating {len(selected)} HumanEval+ problems with pass@{args.k} (model={MODEL}, temp={TEMPERATURE})")
    run_results = []
    passed_count = 0

    for idx, task_id in enumerate(selected, 1):
        print(f"\n[{idx}/{len(selected)}] Task: {task_id}")
        res = evaluate_problem(task_id, problems[task_id], args.k)
        run_results.append(res)
        if res.get("passed"):
            passed_count += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")
            # Show last attempt stderr/stdout (if any) for quick debugging
            atts = res.get("attempts", [])
            if atts:
                last = atts[-1]
                if last.get("stdout"):
                    print("\n--- stdout (last attempt) ---")
                    print(last["stdout"])
                if last.get("stderr"):
                    print("\n--- stderr (last attempt) ---")
                    print(last["stderr"])

    # Overall pass@k (per-problem: at least one pass among k attempts)
    total = len(run_results)
    pass_at_k = passed_count / total if total else 0.0
    print(f"\n=== Summary ===")
    print(f"Problems passed at least once out of {args.k} attempts: {passed_count}/{total}")
    print(f"pass@{args.k}: {pass_at_k:.3f}")

    summary = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "k": args.k,
        "num_problems": total,
        "pass_at_k": pass_at_k,
        "results": run_results,
        "save_dir": os.path.abspath(SAVE_DIR),
        "prompt_dir": os.path.abspath(PROMPT_DIR),
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {os.path.abspath(RESULTS_JSON)}")
    print(f"Generated solutions in: {os.path.abspath(SAVE_DIR)}")
    print(f"Prompts saved in: {os.path.abspath(PROMPT_DIR)}")

if __name__ == "__main__":
    main()
