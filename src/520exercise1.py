#!/usr/bin/env python3
"""
pass@k evaluator for code generation (Windows-safe, no emojis).
- Saves every model attempt as .py and raw .txt
- Prints absolute paths and sizes
- Adds --replay <task> <n> and --retest <path.py>

Requirements:
  pip install openai
  Set OPENAI_API_KEY:
    PowerShell: setx OPENAI_API_KEY "sk-..."
    macOS/Linux: export OPENAI_API_KEY="sk-..."
"""

import os, sys, time, json, tempfile, subprocess, shutil, textwrap, re, argparse
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
    # Add more tasks here with "name", "question", "tests"
]

MODEL = "gpt-4o-mini"
K = 3
TEMPERATURE = 0.6
MAX_TOKENS = 800
TEST_TIMEOUT = 10

SAVE_DIR = "generated_solutions"
os.makedirs(SAVE_DIR, exist_ok=True)
# --------------------------------------------


# ------------------ OpenAI ------------------
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
    # Strip ``` fences if present
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    return text


# ------------------ Utilities ------------------
def _sanitize(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return slug or "task"

def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_attempt_files(task_name: str, attempt: int, code: str, raw_text: str) -> Dict[str, str]:
    base = _sanitize(task_name)
    py_path = os.path.abspath(os.path.join(SAVE_DIR, f"{base}_attempt{attempt}.py"))
    raw_path = os.path.abspath(os.path.join(SAVE_DIR, f"{base}_attempt{attempt}.raw.txt"))
    # Always save raw text
    _write_text(raw_path, raw_text if raw_text is not None else "")
    # Save code; if empty, place a helpful comment so the file is never blank
    code_to_save = code.strip()
    if not code_to_save:
        code_to_save = "# Empty or non-code response from model.\n"
    _write_text(py_path, code_to_save)
    return {"py_path": py_path, "raw_path": raw_path}

def bytesize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ------------------ Test runner ------------------
def run_tests(solution_code: str, tests: str, timeout_sec: int = TEST_TIMEOUT) -> Dict[str, Any]:
    tmpdir = tempfile.mkdtemp(prefix="passk_")
    sol_path = os.path.join(tmpdir, "solution.py")
    test_path = os.path.join(tmpdir, "run_tests.py")

    try:
        _write_text(sol_path, solution_code)

        clean_tests = textwrap.dedent(tests).strip()
        run_code = f"""\
import importlib.util, sys
spec = importlib.util.spec_from_file_location("solution", r"{sol_path}")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

{clean_tests}

print("PASS: Tests finished without assertion errors.")
"""
        _write_text(test_path, run_code)

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


# ------------------ Evaluation ------------------
def evaluate_task_passk(task: Dict[str, str], k: int) -> Dict[str, Any]:
    name = task.get("name") or "unnamed_task"
    question = task["question"]
    tests = task["tests"]

    attempt_summaries = []
    passed = False
    first_pass_attempt = None

    for i in range(1, k + 1):
        t0 = time.time()
        try:
            raw_text = ask_openai(question)
        except Exception as e:
            attempt_summaries.append({
                "i": i, "code_path": None, "raw_path": None, "exit_code": -1,
                "stdout_tail": "", "stderr_tail": f"OpenAI error: {e}",
                "gen_time": time.time() - t0, "run_time": 0.0
            })
            continue

        gen_time = time.time() - t0
        code = raw_text.strip()
        # Save both code and raw text
        paths = save_attempt_files(name, i, code, raw_text)
        py_path, raw_path = paths["py_path"], paths["raw_path"]

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
            "raw_path": raw_path,
            "exit_code": result["code"],
            "stdout_tail": tail(result["stdout"]),
            "stderr_tail": tail(result["stderr"]),
            "gen_time": gen_time,
            "run_time": run_time,
            "code_bytes": bytesize(py_path),
            "raw_bytes": bytesize(raw_path),
        })

        if passed:
            break

    return {
        "name": name,
        "passed": passed,
        "attempts": len(attempt_summaries),
        "first_pass_attempt": first_pass_attempt,
        "attempt_summaries": attempt_summaries,
    }


# ------------------ Replay / Retest ------------------
def _get_task_by_name(name: str) -> Dict[str, str]:
    for t in TASKS:
        if t.get("name") == name:
            return t
    raise KeyError(f"No task named '{name}' in TASKS.")

def replay(task_name: str, attempt_num: int) -> None:
    task = _get_task_by_name(task_name)
    base = _sanitize(task_name)
    py_path = os.path.abspath(os.path.join(SAVE_DIR, f"{base}_attempt{attempt_num}.py"))
    if not os.path.exists(py_path):
        print(f"Not found: {py_path}")
        sys.exit(1)
    code = open(py_path, "r", encoding="utf-8").read()
    print(f"Replaying {task_name} attempt {attempt_num} from:\n  {py_path}\n")
    res = run_tests(code, task["tests"])
    print("Exit code:", res["code"])
    if res["stdout"]:
        print("\n--- stdout ---\n" + res["stdout"])
    if res["stderr"]:
        print("\n--- stderr ---\n" + res["stderr"])

def retest(py_path: str, task_name_for_tests: str) -> None:
    task = _get_task_by_name(task_name_for_tests)
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path):
        print(f"Not found: {py_path}")
        sys.exit(1)
    code = open(py_path, "r", encoding="utf-8").read()
    print(f"Retesting {py_path} with tests from task '{task_name_for_tests}'")
    res = run_tests(code, task["tests"])
    print("Exit code:", res["code"])
    if res["stdout"]:
        print("\n--- stdout ---\n" + res["stdout"])
    if res["stderr"]:
        print("\n--- stderr ---\n" + res["stderr"])


# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser(description="pass@k evaluator with saving and replay.")
    parser.add_argument("--replay", nargs=2, metavar=("TASK_NAME", "ATTEMPT_NUM"),
                        help="Replay a saved attempt for a task (e.g., --replay add 1)")
    parser.add_argument("--retest", nargs=2, metavar=("PATH_TO_PY", "TASK_NAME"),
                        help="Retest a saved .py with tests from TASK_NAME")
    args = parser.parse_args()

    if args.replay:
        task_name, attempt_s = args.replay
        replay(task_name, int(attempt_s))
        return

    if args.retest:
        path, task_name = args.retest
        retest(path, task_name)
        return

    print(f"Running pass@{K} on {len(TASKS)} task(s). Model={MODEL}, temperature={TEMPERATURE}")
    per_task = []
    for task in TASKS:
        print(f"\n=== Task: {task.get('name', 'unnamed')} ===")
        res = evaluate_task_passk(task, K)
        status = "PASS" if res["passed"] else "FAIL"
        print(f"Result: {status} (attempts used: {res['attempts']}, first pass attempt: {res['first_pass_attempt']})")

        for info in res["attempt_summaries"]:
            if info.get("code_path"):
                print(f"Saved attempt {info['i']} to: {info['code_path']}  ({info.get('code_bytes', 0)} bytes)")
                if info.get("raw_path"):
                    print(f"Raw text: {info['raw_path']}  ({info.get('raw_bytes', 0)} bytes)")

        if not res["passed"]:
            last = res["attempt_summaries"][-1]
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

    out = {"k": K, "model": MODEL, "temperature": TEMPERATURE, "results": per_task, "pass_at_k": metric}
    with open("passk_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {os.path.abspath('passk_results.json')}")
    print(f"Generated solutions in: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
