"""Kaggle L4 pilot: Qwen3.8-27B proposes verified ARC solve(grid) programs."""

import ast
import json
import math
import multiprocessing
import re
import time
from pathlib import Path
from queue import Empty


TASK_ID = "05a7bcf2"
THINKING_CONDITIONS = (
    ("off", False, None, 4096),
    ("low", True, "low", 12288),
    ("medium", True, "medium", 12288),
    ("xhigh", True, "xhigh", 12288),
)
MAX_AGENTIC_TURNS = 10
PROGRAMS_PER_RESPONSE = 2
AGENTIC_MAX_NEW_TOKENS = 4096
PROGRAM_TIMEOUT_SECONDS = 2.0
SEED = 42
INPUT_DIR = Path("/kaggle/input")
RESULTS_PATH = Path("/kaggle/working/scaling_results.json")
SUMMARY_PATH = Path("/kaggle/working/scaling_summary.json")

SYSTEM_PROMPT = """You synthesize general ARC-AGI-2 transformation programs.
Return executable hypotheses, not predicted grids. Each hypothesis must define:
def solve(grid):
    ...
    return output
The input and output are rectangular integer grids, at most 30x30, with cells 0..9.
Infer all colors, sizes, positions and counts from the grid.
Allowed: functions, assignments, if, for, comprehensions, indexing, arithmetic,
safe builtins, and ordinary list/dict/set methods.
Forbidden: imports, while, recursion, exceptions, classes, files, networking,
eval/exec, lambda, yield, global state, or memorizing a full training input/output.
Nested helper functions and the conventional unused variable name _ are allowed.
Find a compact rule that exactly reproduces every training pair. Prefer object,
relation, panel, scanline, periodic, or local-template abstractions.
Return only JSON:
{"analysis":"concise abstraction","programs":[{"code":"def solve(grid):\\n..."}]}
Return at most two materially different programs."""

SAFE_BUILTINS = {
    name: value
    for name, value in {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len,
        "list": list, "map": map, "max": max, "min": min, "next": next,
        "range": range, "reversed": reversed, "round": round,
        "set": set, "sorted": sorted,
        "sum": sum, "tuple": tuple, "zip": zip,
    }.items()
}
SAFE_METHODS = {
    "add", "append", "copy", "count", "discard", "extend", "get", "index",
    "items", "keys", "pop", "remove", "reverse", "sort", "update", "values",
}
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.While,
    ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Global, ast.Nonlocal,
    ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await, ast.Delete,
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def find_file(name):
    matches = list(INPUT_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_DIR}")
    return matches[0]


def find_model_dir():
    candidates = [
        path.parent
        for path in INPUT_DIR.rglob("config.json")
        if (path.parent / "processor_config.json").exists()
        or (path.parent / "preprocessor_config.json").exists()
    ]
    qwen = [path for path in candidates if "qwen" in str(path).lower()]
    if not qwen:
        raise FileNotFoundError("Qwen3.8 model directory not found")
    return qwen[0]


def build_prompt(task):
    lines = ["Training examples:"]
    for index, pair in enumerate(task["train"], 1):
        lines.append(
            f"{index}. input={compact(pair['input'])} "
            f"output={compact(pair['output'])}"
        )
    lines.append("Visible test inputs (never hardcode these exact values):")
    for index, pair in enumerate(task["test"], 1):
        lines.append(f"{index}. input={compact(pair['input'])}")
    lines.append("Propose up to two compact solve(grid) programs.")
    return "\n".join(lines)


def parse_response(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("programs"), list):
            continue
        sources = []
        for item in value["programs"]:
            source = item.get("code") if isinstance(item, dict) else item
            normalized = source.strip() if isinstance(source, str) else ""
            if normalized and normalized not in sources:
                sources.append(normalized)
            if len(sources) >= PROGRAMS_PER_RESPONSE:
                break
        return value.get("analysis"), sources
    return None, []


def validate_grid(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise ValueError("invalid grid height")
    if not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30:
        raise ValueError("invalid grid width")
    width = len(grid[0])
    if any(not isinstance(row, list) or len(row) != width for row in grid):
        raise ValueError("ragged grid")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("invalid cell")


def validate_program(source):
    if not source.strip() or len(source) > 20_000:
        raise ValueError("empty or oversized source")
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 2_000 or any(isinstance(node, FORBIDDEN_NODES) for node in nodes):
        raise ValueError("forbidden or oversized AST")
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_functions) != len(tree.body) or not 1 <= len(top_functions) <= 16:
        raise ValueError("top level must contain only functions")
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef)]
    names = {function.name for function in functions}
    if "solve" not in names or any(name.startswith("_") or "__" in name for name in names):
        raise ValueError("invalid function names")
    solve = next(function for function in functions if function.name == "solve")
    args = [*solve.args.posonlyargs, *solve.args.args]
    if len(args) != 1 or args[0].arg != "grid" or solve.args.vararg or solve.args.kwarg:
        raise ValueError("solve must have signature solve(grid)")
    allowed_calls = set(SAFE_BUILTINS) | names
    for node in nodes:
        if isinstance(node, ast.Name) and (
            (node.id.startswith("_") and node.id != "_") or "__" in node.id
        ):
            raise ValueError("private identifiers are forbidden")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError("forbidden attribute")
        if isinstance(node, ast.Call):
            name_call = isinstance(node.func, ast.Name) and node.func.id in allowed_calls
            method_call = isinstance(node.func, ast.Attribute) and node.func.attr in SAFE_METHODS
            if not name_call and not method_call:
                raise ValueError("forbidden call")
    for function in functions:
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function.name
            for node in ast.walk(function)
        ):
            raise ValueError("recursion is forbidden")
    return tree


def worker(source, grid, queue):
    started = time.perf_counter()
    try:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            memory = 768 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except ImportError:
            pass
        namespace = {"__builtins__": SAFE_BUILTINS}
        exec(compile(source, "<generated>", "exec"), namespace)
        output = namespace["solve"]([row[:] for row in grid])
        validate_grid(output)
        queue.put({"output": output, "error": None, "elapsed": time.perf_counter() - started})
    except BaseException as exc:
        queue.put({"output": None, "error": f"{type(exc).__name__}: {exc}",
                   "elapsed": time.perf_counter() - started})


def run_program(source, grid):
    validate_program(source)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=worker, args=(source, grid, queue))
    process.start()
    process.join(PROGRAM_TIMEOUT_SECONDS + 0.25)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"output": None, "error": "execution timed out", "elapsed": PROGRAM_TIMEOUT_SECONDS}
    try:
        return queue.get_nowait()
    except Empty:
        return {"output": None, "error": f"worker exit {process.exitcode}", "elapsed": 0.0}
    finally:
        queue.close()


def cell_score(prediction, truth):
    predicted = sum(len(row) for row in prediction)
    expected = sum(len(row) for row in truth)
    matches = sum(
        prediction[row][column] == truth[row][column]
        for row in range(min(len(prediction), len(truth)))
        for column in range(min(len(prediction[row]), len(truth[row])))
    )
    return matches, max(predicted, expected)


def grid_diagnostics(prediction, truth, error=None, coordinate_limit=40):
    if prediction is None:
        return {"error": error or "no output"}
    predicted_shape = [
        len(prediction), len(prediction[0]) if prediction and prediction[0] else 0,
    ]
    expected_shape = [len(truth), len(truth[0]) if truth and truth[0] else 0]
    mismatches = []
    missing = extra = wrong_color = 0
    for row in range(max(len(prediction), len(truth))):
        predicted_row = prediction[row] if row < len(prediction) else []
        expected_row = truth[row] if row < len(truth) else []
        for column in range(max(len(predicted_row), len(expected_row))):
            got = predicted_row[column] if column < len(predicted_row) else None
            expected = expected_row[column] if column < len(expected_row) else None
            if got == expected:
                continue
            if expected not in (None, 0) and got in (None, 0):
                kind = "missing"
                missing += 1
            elif got not in (None, 0) and expected in (None, 0):
                kind = "extra"
                extra += 1
            else:
                kind = "wrong_color"
                wrong_color += 1
            if len(mismatches) < coordinate_limit:
                mismatches.append({
                    "row": row, "col": column, "got": got,
                    "expected": expected, "kind": kind,
                })
    mismatch_count = missing + extra + wrong_color
    return {
        "exact": mismatch_count == 0,
        "predicted_shape": predicted_shape,
        "expected_shape": expected_shape,
        "mismatch_count": mismatch_count,
        "missing": missing,
        "extra": extra,
        "wrong_color": wrong_color,
        "mismatches": mismatches,
        "coordinates_truncated": mismatch_count > len(mismatches),
    }


def feedback_payload(evaluations, limit=2):
    ranked = sorted(evaluations, key=sort_key)
    return [
        {
            "program": item["source"],
            "valid": item["valid"],
            "error": item["error"],
            "train_matches": item["train_matches"],
            "train_total": item["train_total"],
            "cell_accuracy": round(item["cell_accuracy"], 6),
            "train_diagnostics": item.get("train_diagnostics", []),
        }
        for item in ranked[:limit]
    ]


def build_repair_instruction(evaluations, round_index, force_reparse):
    strategy = (
        "The previous interpretation stopped improving. Discard its central "
        "assumption and infer a materially different scene decomposition and rule."
        if force_reparse
        else "Diagnose the failed assumption from the counterexamples and repair it."
    )
    return (
        f"\nThis is repair round {round_index}. {strategy}\n"
        "The verifier executed prior programs on every training pair. "
        f"Exact reports: {compact(feedback_payload(evaluations))}\n"
        "All training outputs were already provided, so using these differences "
        "is allowed. Return a revised general program, not a coordinate-specific "
        "patch. Obey the sandbox restrictions."
    )


def evaluate(source, task):
    started = time.perf_counter()
    try:
        tree = validate_program(source)
        complexity = len(list(ast.walk(tree))) + math.ceil(len(source) / 32)
    except BaseException as exc:
        return {
            "source": source, "valid": False, "error": f"{type(exc).__name__}: {exc}",
            "train_matches": 0, "train_total": len(task["train"]),
            "cell_accuracy": 0.0, "complexity": None, "predictions": [],
            "train_diagnostics": [],
            "test_exact": None, "test_total": None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    train_matches = matching = cells = 0
    errors = []
    train_diagnostics = []
    for index, pair in enumerate(task["train"]):
        result = run_program(source, pair["input"])
        if result["output"] is None:
            errors.append(f"train[{index}]: {result['error']}")
            train_diagnostics.append(
                grid_diagnostics(None, pair["output"], result["error"])
            )
            continue
        train_diagnostics.append(grid_diagnostics(result["output"], pair["output"]))
        train_matches += int(result["output"] == pair["output"])
        correct, total = cell_score(result["output"], pair["output"])
        matching += correct
        cells += total
    predictions = []
    test_exact = test_total = None
    if train_matches == len(task["train"]):
        test_exact = test_total = 0
        for index, pair in enumerate(task["test"]):
            result = run_program(source, pair["input"])
            if result["output"] is None:
                errors.append(f"test[{index}]: {result['error']}")
                continue
            predictions.append(result["output"])
            if "output" in pair:
                test_total += 1
                test_exact += int(result["output"] == pair["output"])
    return {
        "source": source, "valid": True, "error": errors or None,
        "train_matches": train_matches, "train_total": len(task["train"]),
        "cell_accuracy": matching / cells if cells else 0.0,
        "complexity": complexity, "predictions": predictions,
        "train_diagnostics": train_diagnostics,
        "test_exact": test_exact, "test_total": test_total,
        "elapsed_seconds": time.perf_counter() - started,
    }


def sort_key(item):
    return (
        item["train_matches"] != item["train_total"],
        -item["train_matches"], -item["cell_accuracy"],
        item["complexity"] if item["complexity"] is not None else 10**9,
    )


def generate_once(
    model, processor, torch, messages, enable_thinking,
    reasoning_effort, max_new_tokens, seed,
):
    template_options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": enable_thinking,
    }
    if reasoning_effort:
        template_options["reasoning_effort"] = reasoning_effort
    inputs = processor.apply_chat_template(
        messages, **template_options,
    ).to(model.device)
    torch.manual_seed(seed)
    started = time.perf_counter()
    generation_options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 20,
        "repetition_penalty": 1.0 if enable_thinking else 1.05,
    }
    if enable_thinking:
        generation_options.update({"temperature": 1.0, "top_p": 0.95})
    else:
        generation_options.update({"temperature": 0.7, "top_p": 0.8})
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_options)
    elapsed = time.perf_counter() - started
    tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw = processor.decode(tokens, skip_special_tokens=True)
    analysis, sources = parse_response(raw)
    return {
        "analysis": analysis,
        "sources": sources,
        "raw": raw,
        "output_tokens": int(tokens.shape[0]),
        "elapsed_seconds": round(elapsed, 3),
    }


def evaluate_sources(sources, task):
    unique = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    return sorted([evaluate(source, task) for source in unique], key=sort_key)


def condition_summary(evaluations, train_total):
    if not evaluations:
        return {
            "candidate_programs": 0, "valid_programs": 0,
            "best_train_matches": 0, "train_total": train_total,
            "best_cell_accuracy": 0.0, "train_exact": False,
            "test_exact": None, "test_total": None,
        }
    best = evaluations[0]
    return {
        "candidate_programs": len(evaluations),
        "valid_programs": sum(item["valid"] for item in evaluations),
        "best_train_matches": best["train_matches"],
        "train_total": best["train_total"],
        "best_cell_accuracy": best["cell_accuracy"],
        "train_exact": best["train_matches"] == best["train_total"],
        "test_exact": best["test_exact"],
        "test_total": best["test_total"],
    }


def main():
    import subprocess
    import sys

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        wheels = list(INPUT_DIR.rglob("bitsandbytes-*.whl"))
        if not wheels:
            raise FileNotFoundError("offline bitsandbytes wheel not found")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--find-links",
                str(wheels[0].parent),
                "bitsandbytes>=0.49.2",
            ]
        )
    import torch
    import transformers
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
    )

    challenge_path = find_file("arc-agi_training_challenges.json")
    solution_path = find_file("arc-agi_training_solutions.json")
    model_dir = find_model_dir()
    challenges = json.loads(challenge_path.read_text())
    solutions = json.loads(solution_path.read_text())
    for pair, truth in zip(challenges[TASK_ID]["test"], solutions[TASK_ID]):
        pair["output"] = truth

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, device_map="auto", quantization_config=quantization,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False,
    ).eval()

    task = challenges[TASK_ID]
    base_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": build_prompt(task)}]},
    ]
    results = {
        "task_id": TASK_ID,
        "model": str(model_dir),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "thinking_conditions": [],
        "agentic": {},
    }

    for name, thinking, effort, token_budget in THINKING_CONDITIONS:
        response = generate_once(
            model, processor, torch, base_messages,
            thinking, effort, token_budget, SEED,
        )
        evaluations = evaluate_sources(response.pop("sources"), task)
        record = {
            "name": name,
            "enable_thinking": thinking,
            "reasoning_effort": effort,
            "max_new_tokens": token_budget,
            "response": response,
            "evaluations": evaluations,
            "summary": condition_summary(evaluations, len(task["train"])),
        }
        results["thinking_conditions"].append(record)
        print(
            f"[thinking={name}] tokens={response['output_tokens']} "
            f"structured={bool(evaluations)} "
            f"best={record['summary']['best_train_matches']}/"
            f"{record['summary']['train_total']}",
            flush=True,
        )

    responses = []
    evaluations = []
    seen = set()
    best_history = []
    first_train_exact_turn = None
    first_test_exact_turn = None
    for turn in range(MAX_AGENTIC_TURNS):
        force_reparse = (
            turn >= 2
            and len(best_history) >= 2
            and best_history[-1] <= best_history[-2]
        )
        prompt = build_prompt(task)
        if turn:
            prompt += build_repair_instruction(evaluations, turn, force_reparse)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        response = generate_once(
            model, processor, torch, messages, False, None,
            AGENTIC_MAX_NEW_TOKENS, SEED + 100 + turn,
        )
        sources = response.pop("sources")
        new_evaluations = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                result = evaluate(source, task)
                evaluations.append(result)
                new_evaluations.append(result)
        evaluations.sort(key=sort_key)
        if evaluations:
            best = evaluations[0]
            best_history.append(best["train_matches"] + best["cell_accuracy"])
        else:
            best_history.append(-1.0)
        exact = [
            item for item in evaluations
            if item["train_matches"] == item["train_total"]
            and len(item["predictions"]) == len(task["test"])
        ]
        responses.append({
            "turn": turn,
            "force_reparse": force_reparse,
            "response": response,
            "new_evaluations": new_evaluations,
            "best": condition_summary(evaluations, len(task["train"])),
        })
        print(
            f"[agentic turn={turn}] tokens={response['output_tokens']} "
            f"new={len(new_evaluations)} "
            f"best={responses[-1]['best']['best_train_matches']}/"
            f"{responses[-1]['best']['train_total']}",
            flush=True,
        )
        if exact:
            first_train_exact_turn = turn
            if any(item["test_exact"] == item["test_total"] for item in exact):
                first_test_exact_turn = turn
            break

    results["agentic"] = {
        "max_turns": MAX_AGENTIC_TURNS,
        "responses": responses,
        "evaluations": evaluations,
        "best_history": best_history,
        "first_train_exact_turn": first_train_exact_turn,
        "first_test_exact_turn": first_test_exact_turn,
        "summary": condition_summary(evaluations, len(task["train"])),
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))

    summary = {
        "model": str(model_dir), "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "task_id": TASK_ID,
        "thinking": [
            {
                "name": item["name"],
                "reasoning_effort": item["reasoning_effort"],
                "max_new_tokens": item["max_new_tokens"],
                "output_tokens": item["response"]["output_tokens"],
                "seconds": item["response"]["elapsed_seconds"],
                **item["summary"],
            }
            for item in results["thinking_conditions"]
        ],
        "agentic": {
            "turns_executed": len(responses),
            "first_train_exact_turn": first_train_exact_turn,
            "first_test_exact_turn": first_test_exact_turn,
            "total_output_tokens": sum(
                item["response"]["output_tokens"] for item in responses
            ),
            "seconds": round(sum(
                item["response"]["elapsed_seconds"] for item in responses
            ), 3),
            **results["agentic"]["summary"],
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
