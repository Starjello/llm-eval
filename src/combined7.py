#!/usr/bin/env python3
"""
HumanEval evaluator for OpenAI (ChatGPT) and Google Gemini (2.5) with self-debugging.
Trimmed for minimal prompt tokens while keeping enough context.
"""

import os, sys, time, json, tempfile, subprocess, shutil, textwrap, re, ast, argparse
from typing import List, Dict, Any, Optional

# ------------------ CONFIG ------------------
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"   # override via --gemini-model

K = 2                   # attempts per task (base + one debug)
TEMPERATURE = 0.2
TEST_TIMEOUT = 15
DEBUG_ROUNDS = 1

BASE_SAVE_DIR = "generated_solutions"
RESULTS_JSON = "combo_passk_results.json"

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_SAVE_DIR, "openai"), exist_ok=True)
os.makedirs(os.path.join(BASE_SAVE_DIR, "gemini"), exist_ok=True)

PROMPTS_DIR = "prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROMPTS_DIR, "openai"), exist_ok=True)
os.makedirs(os.path.join(PROMPTS_DIR, "gemini"), exist_ok=True)
# --------------------------------------------


# ------------------ Prompt builders (ultra-compact first) ------------------
def base_system_hint() -> str:
    # Short, strong constraints. No markdown fences/explanations.
    return (
        "Think step-by-step internally. Do NOT reveal reasoning. "
        "Return ONLY valid Python 3 code for a single module. "
        "No markdown, no comments, no extra text."
         "Before writing, silently evaluate the question so that you are both following the exact instructions and fully understand the question"
    )

def _extract_signature_and_docstring(src: str, doc_chars: int = 220) -> str:
    """
    Extract the function signature and at most `doc_chars` of docstring.
    Fallback: first 400 chars of raw prompt.
    """
    try:
        tree = ast.parse(src)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if not func:
            return (src or "").strip()[:400]
        args = [a.arg for a in func.args.args]
        sig = f"def {func.name}({', '.join(args)}):"
        doc = (ast.get_docstring(func) or "").strip()
        if doc:
            doc = (doc[:doc_chars] + "…") if len(doc) > doc_chars else doc
            doc_block = f'"""{doc}"""'
        else:
            doc_block = '"""Implement this function correctly. Return value; do not print."""'
        return f"{sig}\n{doc_block}\n"
    except Exception:
        return (src or "").strip()[:400]

def strip_comments_and_blank_lines(code: str) -> str:
    out = []
    for line in (code or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in line:
            line = line[:line.index("#")]
        line = line.rstrip()
        if line:
            out.append(line)
    return "\n".join(out)

def compact_humaneval_prompt(question: str) -> str:
    # Signature + short instruction line
    core = _extract_signature_and_docstring(question, doc_chars=220)
    return core + "# Use Python 3 stdlib only. Return, do not print. Handle edge cases. Keep code minimal.\n"

def ultra_compact_humaneval_prompt(question: str) -> str:
    # Signature-only + one-liner instruction
    try:
        tree = ast.parse(question)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if not func:
            return "Return valid Python 3 code implementing the required function."
        args = [a.arg for a in func.args.args]
        sig = f"def {func.name}({', '.join(args)}):"
        return f'{sig}\n"""Implement correctly. Return value; do not print."""\n'
    except Exception:
        return "Return valid Python 3 code implementing the required function."

def user_prompt_from_question(question: str) -> str:
    # If prompt looks large, auto-fallback to ultra-compact
    q = question or ""
    return compact_humaneval_prompt(q) if len(q) <= 1200 else ultra_compact_humaneval_prompt(q)

def debug_user_prompt(previous_code: str, error_excerpt: str) -> str:
    # Very tight repair prompt, last N chars only
    prev_small = strip_comments_and_blank_lines(previous_code or "")[-500:]
    err_small  = (error_excerpt or "")[-240:]
    return (
        "Fix the module to pass tests.\n"
        "Return ONLY corrected Python code. No explanations or markdown.\n"
        "Keep the same function signatures.\n\n"
        "Error (trimmed):\n"
        f"{err_small}\n\n"
        "Current code (trimmed):\n"
        f"{prev_small}\n"
    )


# ------------------ Provider calls ------------------
def ask_openai_code(prompt: str, temperature: float = TEMPERATURE, model: str = OPENAI_MODEL) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. pip install openai") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": base_system_hint()},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=320,
        top_p=1,
        n=1,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    return text

# --- Thinking budget helper ---
def _thinking_budget_for(model: str) -> int:
    m = (model or "").lower()
    if "2.5-flash" in m:
        return 0       # disable thinking for flash (great for code gen)
    if "2.5-pro" in m:
        return 64      # keep small budget for pro (cannot fully disable)
    return -1          # unknown model -> don't set explicitly


# --- Gemini caller with thinking budget control + SDK fallbacks ---
def _thinking_budget_for(model: str) -> int:
    m = (model or "").lower()
    if "2.5-flash" in m:
        return 0      # no thinking for flash -> maximize output budget
    if "2.5-pro" in m:
        return 32     # tiny but nonzero for pro; can try 0 if SDK permits, else 32/64
    return -1

def ask_gemini_code(prompt: str, temperature: float = TEMPERATURE, model: str = GEMINI_MODEL) -> str:
    import os
    try:
        from google import genai
        from google.genai import types
        has_types = True
    except Exception:
        import google.genai as genai
        types = None
        has_types = False

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    client = genai.Client(api_key=api_key)

    sys_hint = (
        "Think silently. Output ONLY valid Python 3 code as a single module. "
        "No markdown fences. No comments. No explanations."
    )

    def extract_text(resp) -> str:
        t = getattr(resp, "text", None) or ""
        if t.strip():
            return t.strip()
        out = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                txt = getattr(part, "text", "")
                if txt:
                    out.append(txt)
        return "".join(out).strip()

    def debug_finish(resp, label: str):
        frs = [getattr(c, "finish_reason", None) for c in (getattr(resp, "candidates", []) or [])]
        usage = getattr(resp, "usage_metadata", None)
        print(f"[gemini:{label}] finish_reasons={frs} | usage={usage}")

    def try_generate(user_prompt: str, max_out_tokens: int) -> str:
        tb = _thinking_budget_for(model)

        # NOTE: stop_sequences removed to avoid early cutoffs
        # NOTE: we also avoid any max-newlines fences, etc.

        # 1) Preferred typed config
        if has_types:
            try:
                cfg_kwargs = dict(
                    system_instruction=sys_hint,
                    temperature=float(temperature),
                    max_output_tokens=int(max_out_tokens),
                    top_k=1,
                    candidate_count=1,
                )
                if tb >= 0:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(tb))
                cfg = types.GenerateContentConfig(**cfg_kwargs)
                resp = client.models.generate_content(model=model, contents=[user_prompt], config=cfg)
                txt = extract_text(resp)
                if not txt:
                    debug_finish(resp, f"typed_cfg@{max_out_tokens}")
                if txt.startswith("```"):
                    txt = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("```")).strip()
                return txt
            except TypeError:
                pass
            except Exception:
                pass

        # 2) Dict config (camelCase for thinkingConfig)
        try:
            cfg = {
                "system_instruction": sys_hint,
                "temperature": float(temperature),
                "max_output_tokens": int(max_out_tokens),
                "top_k": 1,
                "candidate_count": 1,
                # no stop_sequences here
            }
            if tb >= 0:
                cfg["thinkingConfig"] = {"thinkingBudget": int(tb)}
            resp = client.models.generate_content(model=model, contents=[user_prompt], config=cfg)
            txt = extract_text(resp)
            if not txt:
                debug_finish(resp, f"dict_config@{max_out_tokens}")
            if txt.startswith("```"):
                txt = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("```")).strip()
            return txt
        except TypeError:
            pass
        except Exception:
            pass

        # 3) generation_config (older SDKs)
        try:
            gen_cfg = {
                "temperature": float(temperature),
                "max_output_tokens": int(max_out_tokens),
                "top_k": 1,
            }
            if tb >= 0:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": int(tb)}
            resp = client.models.generate_content(model=model, contents=[user_prompt], generation_config=gen_cfg)
            txt = extract_text(resp)
            if not txt:
                debug_finish(resp, f"generation_config@{max_out_tokens}")
            if txt.startswith("```"):
                txt = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("```")).strip()
            return txt
        except Exception:
            pass

        # 4) Minimal
        try:
            resp = client.models.generate_content(model=model, contents=[user_prompt])
            txt = extract_text(resp)
            if not txt:
                debug_finish(resp, f"minimal@{max_out_tokens}")
            if txt.startswith("```"):
                txt = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("```")).strip()
            return txt
        except Exception:
            return ""

    # BIGGER output ladder — give it space to finish
    oc_ladder = [2000, 1500, 1200, 900, 600]

    # Try your prompt first
    for oc in oc_ladder:
        out = try_generate(prompt, oc)
        if out:
            return out

    # Ultra-compact fallback
    ultra = ultra_compact_humaneval_prompt(prompt)
    for oc in oc_ladder:
        out = try_generate(ultra, oc)
        if out:
            return out

    raise RuntimeError("Gemini produced no usable code even with large output caps and thinking disabled/limited.")


# ------------------ Utilities ------------------
def tail(s: str, n: int = 900) -> str:
    s = s or ""
    return s[-n:] if len(s) > n else s

def _problem_number_from_task(task_name: str) -> str:
    m = re.search(r"HumanEval/(\d+)", task_name or "")
    return m.group(1) if m else "unknown"

def _attempt_label_base() -> str:
    return "base"

def _attempt_label_fix(r: int) -> str:
    return f"fix{r:02d}"

def _provider_slug(provider: str) -> str:
    return "openai" if provider == "openai" else "gemini"

def _provider_save_dir(provider: str) -> str:
    prov_dir = os.path.join(BASE_SAVE_DIR, _provider_slug(provider))
    os.makedirs(prov_dir, exist_ok=True)
    return prov_dir

def _build_filename(problem_number: str, attempt: int, provider: str, label: str) -> str:
    prov = _provider_slug(provider)
    return f"humaneval_{problem_number}_{prov}_attempt_{attempt:03d}_{label}.py"

def _save_failed_variant_code(problem_number: str, attempt: int, provider: str, label: str, code: str) -> str:
    prov_dir = _provider_save_dir(provider)
    base = _build_filename(problem_number, attempt, provider, label)
    path = os.path.abspath(os.path.join(prov_dir, base))
    with open(path, "w", encoding="utf-8") as f:
        f.write((code or "").strip() or "# Empty response from model.\n")
    return path

def _save_prompt_text(problem_number: str, attempt: int, provider: str, label: str, prompt_text: str) -> str:
    prov_dir = os.path.join(PROMPTS_DIR, _provider_slug(provider))
    os.makedirs(prov_dir, exist_ok=True)
    base = f"humaneval_{problem_number}_{_provider_slug(provider)}_attempt_{attempt:03d}_{label}.txt"
    path = os.path.abspath(os.path.join(prov_dir, base))
    # Write the full prompt verbatim (no truncation)
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt_text or "")
    return path


# ------------------ HumanEval loaders ------------------
def load_humaneval_tasks(n: int = 20) -> List[Dict[str, Any]]:
    try:
        from human_eval.data import read_problems
    except Exception as e:
        raise RuntimeError("Install 'human-eval' first: pip install human-eval") from e

    problems = read_problems()
    task_ids = sorted(problems.keys())[:n]
    tasks: List[Dict[str, Any]] = []
    for tid in task_ids:
        prob = problems[tid]
        tasks.append({
            "name": tid,
            "question": prob["prompt"],
            "tests": prob["test"],
            "entry_point": prob["entry_point"],
            "humaneval": True,
        })
    return tasks

def load_humaneval_task_by_number(task_number: int) -> List[Dict[str, Any]]:
    try:
        from human_eval.data import read_problems
    except Exception as e:
        raise RuntimeError("Install 'human-eval' first: pip install human-eval") from e

    problems = read_problems()
    key = f"HumanEval/{task_number}"
    if key not in problems:
        raise RuntimeError(f"Task {task_number} not found in HumanEval.")
    prob = problems[key]
    return [{
        "name": key,
        "question": prob["prompt"],
        "tests": prob["test"],
        "entry_point": prob["entry_point"],
        "humaneval": True,
    }]


# ------------------ Test runner ------------------
def run_tests(solution_code: str, tests: str, timeout_sec: int = TEST_TIMEOUT,
              humaneval: bool = False, entry_point: Optional[str] = None) -> Dict[str, Any]:
    tmpdir = tempfile.mkdtemp(prefix="passk_combo_")
    sol_path = os.path.join(tmpdir, "solution.py")
    test_path = os.path.join(tmpdir, "run_tests.py")

    try:
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_code)

        clean_tests = textwrap.dedent(tests).strip()

        if humaneval:
            if not entry_point:
                raise ValueError("HumanEval test run requires 'entry_point'.")
            run_code = f"""\
import importlib.util

spec = importlib.util.spec_from_file_location("solution", r"{sol_path}")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

globals_ns = {{}}
locals_ns = globals_ns
exec({clean_tests!r}, globals_ns, locals_ns)

candidate = getattr(solution, {entry_point!r}, None)
if candidate is None:
    raise AttributeError("Entry point not found.")

if "check" in globals_ns:
    globals_ns["check"](candidate)
elif "test_check" in globals_ns:
    globals_ns["test_check"]()
else:
    found = False
    for name, obj in list(globals_ns.items()):
        if callable(obj):
            try:
                import inspect
                if len(inspect.signature(obj).parameters) == 1:
                    obj(candidate); found = True; break
            except Exception:
                pass
    if not found:
        raise RuntimeError("No suitable HumanEval check found.")

print("PASS: Tests finished without assertion errors.")
"""
        else:
            run_code = f"""\
import importlib.util
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


# ------------------ Core evaluation with self-debug ------------------
def evaluate_task(provider: str, task: Dict[str, Any], k: int, attempt_index_for_saving: int = 1) -> Dict[str, Any]:
    name = task.get("name") or "unnamed_task"
    question = task["question"]
    tests = task["tests"]
    humaneval = bool(task.get("humaneval", False))
    entry_point = task.get("entry_point")
    problem_number = _problem_number_from_task(name)

    variants = []
    passed = False
    final_variant = 0

    # Base attempt
    prompt = user_prompt_from_question(question)
    _save_prompt_text(problem_number, attempt_index_for_saving, provider, _attempt_label_base(), prompt)
    try:
        code = ask_openai_code(prompt) if provider == "openai" else ask_gemini_code(prompt, model=GEMINI_MODEL)
    except Exception as e:
        variants.append({
            "label": _attempt_label_base(),
            "code": "",
            "exit_code": -1,
            "stdout_tail": "",
            "stderr_tail": f"{provider} error: {e}",
            "elapsed": 0.0,
        })
        return {"name": name, "passed": False, "attempts_used": 1, "final_variant": 0, "variants": variants}

    res = run_tests(code, tests, humaneval=humaneval, entry_point=entry_point)
    variants.append({
        "label": _attempt_label_base(),
        "code": code,
        "exit_code": res["code"],
        "stdout_tail": tail(res["stdout"]),
        "stderr_tail": tail(res["stderr"]),
        "elapsed": res["elapsed"],
    })
    if res["code"] == 0 and "PASS: Tests finished without assertion errors." in res["stdout"]:
        passed = True
        final_variant = 0

    # Debug rounds
    prev_code = code
    r = 0
    while (not passed) and r < DEBUG_ROUNDS:
        r += 1
        err_excerpt = tail((variants[-1]["stderr_tail"] or "") + "\n" + (variants[-1]["stdout_tail"] or ""), 600)
        repair_prompt = debug_user_prompt(prev_code, err_excerpt)
        _save_prompt_text(problem_number, attempt_index_for_saving, provider, _attempt_label_fix(r), repair_prompt)
        try:
            fixed = ask_openai_code(repair_prompt) if provider == "openai" else ask_gemini_code(repair_prompt, model=GEMINI_MODEL)
        except Exception as e:
            variants.append({
                "label": _attempt_label_fix(r),
                "code": "",
                "exit_code": -1,
                "stdout_tail": "",
                "stderr_tail": f"{provider} debug error: {e}",
                "elapsed": 0.0,
            })
            break

        res_fix = run_tests(fixed, tests, humaneval=humaneval, entry_point=entry_point)
        variants.append({
            "label": _attempt_label_fix(r),
            "code": fixed,
            "exit_code": res_fix["code"],
            "stdout_tail": tail(res_fix["stdout"]),
            "stderr_tail": tail(res_fix["stderr"]),
            "elapsed": res_fix["elapsed"],
        })

        if res_fix["code"] == 0 and "PASS: Tests finished without assertion errors." in res_fix["stdout"]:
            passed = True
            final_variant = r
        prev_code = fixed

    return {
        "name": name,
        "passed": passed,
        "attempts_used": len(variants),
        "final_variant": final_variant,
        "variants": variants,
    }


def save_failed_task_variants(problem_number: str, provider: str, attempt_index: int, variants: List[Dict[str, Any]]) -> List[str]:
    saved = []
    for v in variants:
        failed = not (v["exit_code"] == 0 and "PASS: Tests finished without assertion errors." in (v["stdout_tail"] or ""))
        if failed:
            path = _save_failed_variant_code(problem_number, attempt_index, provider, v["label"], v["code"])
            saved.append(path)
    return saved


def evaluate_until_failures_or_exhaustion(provider: str, tasks: List[Dict[str, Any]], k: int, stop_after_failures: int) -> Dict[str, Any]:
    per_task = []
    failures = passes = total_evaluated = 0
    failed_task_counter = 0

    for task in tasks:
        if failures >= stop_after_failures:
            break

        name = task.get("name", "unnamed")
        problem_number = _problem_number_from_task(name)
        print(f"\n=== Task: {name} (entry_point={task.get('entry_point')}) [{provider}] ===")

        result = evaluate_task(provider, task, k, attempt_index_for_saving=(failed_task_counter + 1))
        per_task.append(result)

        total_evaluated += 1
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passes += 1
        else:
            failures += 1
            failed_task_counter += 1

        print(f"Result: {status} (attempts used: {result['attempts_used']}, final_variant: {result['final_variant']})")

        if not result["passed"]:
            saved_paths = save_failed_task_variants(problem_number, provider, failed_task_counter, result["variants"])
            if saved_paths:
                print("Saved failing variants to:")
                for p in saved_paths:
                    print("  ", p)

        last = result["variants"][-1] if result["variants"] else None
        if last and last.get("stderr_tail"):
            print("\n--- stderr (last variant) ---")
            print(last["stderr_tail"])

        current_pass_at_k = passes / total_evaluated if total_evaluated else 0.0
        print(f"Running pass@{k}: {current_pass_at_k:.3f} ({passes}/{total_evaluated})")

    provider_slug = _provider_slug(provider)
    pass_at_k = passes / total_evaluated if total_evaluated else 0.0

    print(f"\n[{provider_slug}] pass@{k} over this run: {pass_at_k:.3f} ({passes}/{total_evaluated})")
    print(f"[{provider_slug}] failures: {failures} | stopped_after_failures={failures >= stop_after_failures} | tasks_evaluated={total_evaluated}")

    return {
        "provider": provider_slug,
        "results": per_task,
        "failures": failures,
        "passes": passes,
        "tasks_evaluated": total_evaluated,
        "stopped_after_failures": failures >= stop_after_failures,
        "stop_after_failures": stop_after_failures,
        "pass_at_k": pass_at_k,
        "save_dir": os.path.abspath(_provider_save_dir(provider)),
    }


# ------------------ Main ------------------
def main():
    global GEMINI_MODEL
    parser = argparse.ArgumentParser(description="HumanEval evaluator for OpenAI/Gemini with failure-capped runs.")
    parser.add_argument("--stop-after-failures", type=int, default=2,
                        help="Stop each provider after this many failed tasks (default: 2).")
    parser.add_argument("--max-tasks", type=int, default=10,
                        help="Max number of HumanEval tasks to load (default: 10).")
    parser.add_argument("--gemini-model", type=str, default=GEMINI_MODEL,
                        help="Gemini model to use (e.g., gemini-2.5-pro or gemini-2.5-flash).")
    args, _ = parser.parse_known_args()

    GEMINI_MODEL = args.gemini_model

    print("Select provider:")
    print("  1) OpenAI (ChatGPT)")
    print("  2) Google Gemini 2.5")
    print("  3) Sequential (ChatGPT, then Gemini)")
    choice = input("Enter 1, 2, or 3: ").strip()

    # Optional single-question mode
    single_id_raw = input("Enter a HumanEval task number to run ONLY that task (e.g., 115), or press Enter to run a set: ").strip()
    single_task_mode = False
    single_task_list: List[Dict[str, Any]] = []
    if single_id_raw:
        try:
            single_id = int(single_id_raw)
            single_task_list = load_humaneval_task_by_number(single_id)
            single_task_mode = True
            print(f"Single-task mode: loaded HumanEval/{single_id}")
        except Exception as e:
            print(f"Could not load task '{single_id_raw}': {e}")
            sys.exit(2)

    if not single_task_mode:
        try:
            TASKS: List[Dict[str, Any]] = load_humaneval_tasks(n=args.max_tasks)
        except Exception as e:
            print(f"Error loading HumanEval tasks: {e}")
            sys.exit(2)

    runs_summary = []

    if choice == "1":
        provider = "openai"
        task_source = single_task_list if single_task_mode else TASKS
        print(f"\nRunning {provider} "
              f"{'(single task)' if single_task_mode else f'until {args.stop_after_failures} failures'} "
              f"(K={K}, debug_rounds={DEBUG_ROUNDS}, "
              f"{'task_id=' + single_id_raw if single_task_mode else 'max_tasks=' + str(args.max_tasks)})")
        summary = evaluate_until_failures_or_exhaustion(provider, task_source, K,
                                                        stop_after_failures=(1 if single_task_mode else args.stop_after_failures))
        runs_summary.append(summary)

    elif choice == "2":
        provider = "gemini"
        task_source = single_task_list if single_task_mode else TASKS
        print(f"\nRunning {provider} "
              f"{'(single task)' if single_task_mode else f'until {args.stop_after_failures} failures'} "
              f"(K={K}, debug_rounds={DEBUG_ROUNDS}, model={GEMINI_MODEL}, "
              f"{'task_id=' + single_id_raw if single_task_mode else 'max_tasks=' + str(args.max_tasks)})")
        summary = evaluate_until_failures_or_exhaustion(provider, task_source, K,
                                                        stop_after_failures=(1 if single_task_mode else args.stop_after_failures))
        runs_summary.append(summary)

    elif choice == "3":
        task_source = single_task_list if single_task_mode else TASKS

        print(f"\n[Phase 1] Running openai "
              f"{'(single task)' if single_task_mode else f'until {args.stop_after_failures} failures'} "
              f"(K={K}, debug_rounds={DEBUG_ROUNDS}, "
              f"{'task_id=' + single_id_raw if single_task_mode else 'max_tasks=' + str(args.max_tasks)})")
        summary_openai = evaluate_until_failures_or_exhaustion("openai", task_source, K,
                                                               stop_after_failures=(1 if single_task_mode else args.stop_after_failures))
        runs_summary.append(summary_openai)

        print(f"\n[Phase 2] Running gemini "
              f"{'(single task)' if single_task_mode else f'until {args.stop_after_failures} failures'} "
              f"(K={K}, debug_rounds={DEBUG_ROUNDS}, model={GEMINI_MODEL}, "
              f"{'task_id=' + single_id_raw if single_task_mode else 'max_tasks=' + str(args.max_tasks)})")
        summary_gemini = evaluate_until_failures_or_exhaustion("gemini", task_source, K,
                                                               stop_after_failures=(1 if single_task_mode else args.stop_after_failures))
        runs_summary.append(summary_gemini)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    print("\n=== Overall Summary ===")
    for run in runs_summary:
        prov = run["provider"]
        failed = run["failures"]
        passed = run["passes"]
        total = run["tasks_evaluated"]
        print(f"{prov}: pass@{K}={run['pass_at_k']:.3f} ({passed}/{total}), "
              f"failures={failed}, stopped_after_failures={run['stopped_after_failures']}, save_dir={run['save_dir']}")

    out = {
        "k": K,
        "debug_rounds": DEBUG_ROUNDS,
        "base_save_dir": os.path.abspath(BASE_SAVE_DIR),
        "runs": runs_summary,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary to {os.path.abspath(RESULTS_JSON)}")
    print(f"Failed solutions saved (only for failed tasks) under: {os.path.abspath(BASE_SAVE_DIR)}")


if __name__ == "__main__":
    main()
