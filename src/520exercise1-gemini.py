#!/usr/bin/env python3
"""
Minimal pass@k evaluator for code generation using Google Gemini (Windows-safe, no emojis).

What it does
- You define tasks (prompt + Python assertions) in TASKS below
- Calls Gemini to generate code for each task (K attempts = pass@K)
- Runs your tests in a subprocess with timeout
- Saves every attempt to generated_solutions/<task>_attempt<N>.py
- Writes a run summary to passk_results_gemini.json

Setup
  pip install google-generativeai
  # Set your key:
  #   Windows PowerShell:  setx GOOGLE_API_KEY "your-key"
  #   macOS/Linux:         export GOOGLE_API_KEY="your-key"
"""

import os, sys, time, json, tempfile, subprocess, shutil, textwrap, re
from typing import List, Dict, Any

# ------------------ CONFIG ------------------
TASKS: List[Dict[str, str]] = [
    {
        "name": "add",
        "question": "Write a function add(a, b) that returns the sum of a and b.",
        "tests": """
        assert solution.add(2, 3) == 5
        assert solution.add(-1, 1) == 0
        assert solution.add(0, 0) == 0
        """,
    },
    # Add more tasks here if you like
]

GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro"
K = 3                 # pass@k attempts per task
TEMPERATURE = 0.6     # >0 for diverse attempts; 0 for deterministic
TEST_TIMEOUT = 10     # seconds per test
SAVE_DIR = "generated_solutions"
RESULTS_JSON = "passk_results_gemini.json"

os.makedirs(SAVE_DIR, exist_ok=True)
# --------------------------------------------


# ------------------ Gemini client ------------------
def _init_gemini():
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. (PowerShell: setx GOOGLE_API_KEY \"your-key\")")
    genai.configure(api_key=api_key)
    return genai

def ask_gemini(prompt: str, model: str = GEMINI_MODEL, temperature: float = TEMPERATURE) -> str:
    """
    Query Gemini and return code-only text (no markdown fences).
    """
    genai = _init_gemini()
    # System-style instruction via "candidate control"—Gemini supports system prompts through safety/defs,
    # but simplest is to include instructions inline.
    system_hint = (
        "Write correct, minimal Python 3 code only. "
        "Return only runnable code; do not include explanations or markdown fences."
    )
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(
        [system_hint, "\n\nUSER PROMPT:\n", prompt],
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 1024,
        },
    )
    text = (resp.text or "").strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    return text


# ------------------ Utilities ------------------
def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or "task"

def save_attempt(task_name: str, attempt: int, code: str) -> str:
    """
    Save code to generated_solutions/<task>_attempt<N>.py and return absolute path.
    Guarantees a non-empty .py (adds a comment if model returned empty).
    """
    base = _sanitize(task_name)
    path = os.path.abspath(os.path.join(SAVE_DIR, f"{base}_attempt{attempt}.py"))
    with open(path, "w", encoding="utf-8") as f:
        f.write((code or "").strip() or "# Empty response from model.\n")
    return path


# ------------------ Test runner ------------------
def run_tests(solution_code: str, tests: str, timeout_sec: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """
    Write solution.py and run_tests.py to a temp dir; execute tests with a timeout.
    Returns dict with {code, stdout, stderr, elapsed}.
    """
    tmpdir = tempfile.mkdtemp(prefix="passk_gem_")
    sol_path = os.path.join(tmpdir, "solution.py")
    test_path = os.path.join(tmpdir, "run_tests.py")

    try:
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_code)

        clean_tests = textwrap.dedent(tests).strip()
        run_code = f"""\
import importlib.util, sys
spec = importlib.util.spec_from_file_location("solution", r"{sol_path}")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

{clean_tests}

print("PASS: Tests finished without assertion errors.")
"""
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(run_code)

        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", "-I", test_path],  # -I isolated, -u unbuffered
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


# ------------------ Evaluation logic ------------------
def evaluate_task_passk(task: Dict[str, str], k: int) -> Dict[str, Any]:
    """
    Try up to k generations for a single task with Gemini.
    Returns:
      {
        "name": task_name,
        "passed": bool,
        "attempts": used_attempts,
        "first_pass_attempt": int or None,
        "attempt_summaries": [ {i, code_path, exit_code, stdout_tail, stderr_tail, gen_time, run_time} ... ]
      }
    """
    name = task.get("name") or "unnamed_task"
    question = task["question"]
    tests = task["tests"]

    attempt_summaries = []
    passed = False
    first_pass_attempt = None

    for i in range(1, k + 1):
        t0 = time.time()
        try:
            code = ask_gemini(question)
        except Exception as e:
            attempt_summaries.append({
                "i": i, "code_path": None, "exit_code": -1,
                "stdout_tail": "", "stderr_tail": f"Gemini error: {e}",
                "gen_time": time.time() - t0, "run_time": 0.0
            })
            continue

        gen_time = time.time() - t0

        # Save the generated code
        py_path = save_attempt(name, i, code)

        # Run tests
        result = run_tests(code, tests)
        run_time = result["elapsed"]

        attempt_pass = (result["code"] == 0 and "PASS: Tests finished without assertion errors." in result["stdout"])
        if attempt_pass and not passed:
            passed = True
            first_pass_attempt = i

        tail = lambda s: s[-600:] if len(s) > 600 else s
        attempt_summaries.append({
            "i": i,
            "code_path": py_path,
            "exit_code": result["code"],
            "stdout_tail": tail(result["stdout"]),
            "stderr_tail": tail(result["stderr"]),
            "gen_time": gen_time,
            "run_time": run_time,
        })

        if passed:
            break  # early stop once we have a pass

    return {
        "name": name,
        "passed": passed,
        "attempts": len(attempt_summaries),
        "first_pass_attempt": first_pass_attempt,
        "attempt_summaries": attempt_summaries,
    }


def main():
    print(f"Running pass@{K} on {len(TASKS)} task(s). Model={GEMINI_MODEL}, temperature={TEMPERATURE}")
    per_task = []
    for task in TASKS:
        print(f"\n=== Task: {task.get('name', 'unnamed')} ===")
        res = evaluate_task_passk(task, K)
        status = "PASS" if res["passed"] else "FAIL"
        print(f"Result: {status} (attempts used: {res['attempts']}, first pass attempt: {res['first_pass_attempt']})")

        for info in res["attempt_summaries"]:
            if info.get("code_path"):
                print(f"Saved attempt {info['i']} to: {info['code_path']}")

        if not res["passed"]:
            last = res["attempt_summaries"][-1]
            if last.get("stdout_tail"):
                print("\n--- stdout (last attempt) ---")
                print(last["stdout_tail"])
            if last.get("stderr_tail"):
                print("\n--- stderr (last attempt) ---")
                print(last["stderr_tail"])
        per_task.append(res)

    total = len(per_task)
    passed = sum(1 for r in per_task if r["passed"])
    metric = passed / total if total else 0.0
    print(f"\n=== Summary ===")
    print(f"Tasks passed at least once out of {K} attempts: {passed}/{total}")
    print(f"pass@{K}: {metric:.3f}")

    out = {"k": K, "model": GEMINI_MODEL, "temperature": TEMPERATURE, "results": per_task, "pass_at_k": metric}
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {os.path.abspath(RESULTS_JSON)}")
    print(f"Generated solutions in: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
