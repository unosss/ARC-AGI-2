#!/usr/bin/env python3
"""One-task REPL pilot with trace sealing before label access."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "public_notebook/output_files/agent"
if not HARNESS.exists():  # compact remote deployment
    HARNESS = HERE / "agent"
sys.path.insert(0, str(HARNESS))
from agent import score  # noqa: E402
from client import ChatClient  # noqa: E402
from llama_cpp_client import LlamaCppChatClient  # noqa: E402
from repl_agent import solve_repl  # noqa: E402


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def server_props(base_url):
    with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=30) as r:
        return json.load(r)


def compatibility_probe(client):
    messages = [
        {"role": "system", "content": "You must use the supplied function."},
        {"role": "user", "content": "Call python with code exactly: print(2+3)"},
    ]
    tool = [{"type": "function", "function": {"name": "python", "description": "Run code",
             "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                            "required": ["code"]}}}]
    result = client.chat(messages, tools=tool, max_tokens=256, seed=0, timeout=300)
    ok = not result.get("error") and any(
        c["name"] == "python" and "print(2+3)" in (c.get("arguments") or {}).get("code", "")
        for c in result.get("tool_calls", [])
    )
    return ok, result


def forced_submit_probe(client):
    messages = [
        {"role": "system", "content": "You must use the supplied function."},
        {"role": "user", "content": "Call submit with a transform that returns its input."},
    ]
    tool = [{"type": "function", "function": {
        "name": "submit",
        "description": "Submit a transform",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}},
            "required": ["code"],
        },
    }}]
    result = client.chat(
        messages, tools=tool, tool_choice="submit", max_tokens=256, seed=0, timeout=300,
    )
    ok = not result.get("error") and any(
        c["name"] == "submit" and (c.get("arguments") or {}).get("code")
        for c in result.get("tool_calls", [])
    )
    return ok, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "pilot_config.json")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_id = cfg["task_id"]
    tasks = json.loads((args.data_dir / "challenge.json").read_text())
    if list(tasks) != [task_id]:
        raise ValueError("pilot challenge must contain exactly the fixed task")
    model = server_props(args.base_url.rsplit("/v1", 1)[0]).get("model_path", cfg["model_file"])
    sampling = cfg["sampling"]
    client_class = LlamaCppChatClient if cfg.get("llama_cpp_tool_compat") else ChatClient
    client = client_class(
        args.base_url, model, max_tokens=cfg["turn_max_tokens"],
        temperature=sampling["temperature"], top_p=sampling["top_p"],
        top_k=sampling["top_k"], thinking=sampling["thinking"],
    )
    compatible, probe = compatibility_probe(client)
    atomic_json(args.output_dir / "tool_compatibility.json", {"compatible": compatible, "response": probe})
    if not compatible:
        raise RuntimeError("llama-server did not return a usable python tool call")
    force_submit_after = cfg.get("force_submit_after", 0)
    if force_submit_after:
        forced_compatible, forced_probe = forced_submit_probe(client)
        atomic_json(args.output_dir / "forced_submit_compatibility.json", {
            "compatible": forced_compatible,
            "request": getattr(client, "last_payload", None),
            "response": forced_probe,
        })
        if not forced_compatible:
            raise RuntimeError(
                "llama-server rejected or did not honor forced tool_choice=submit: "
                + forced_probe.get("error", repr(forced_probe))
            )

    start = time.time()
    deadline = start + cfg["wall_deadline_seconds"]
    answers, report = solve_repl(
        tasks, client, attempts=cfg["attempts"], concurrency=1, deadline=deadline,
        stop_after=1, max_turns=cfg["max_turns"], max_gen_tokens=cfg["max_gen_tokens"],
        context_budget=cfg["context_budget"], nudge_every=cfg["early_submit_nudge"],
        turn_max_tokens=cfg["turn_max_tokens"], force_submit_after=force_submit_after,
        on_done=lambda rec: atomic_json(args.output_dir / "attempt.json", rec),
        log=lambda line: print(line, flush=True),
    )
    candidates = answers[task_id]
    test_input = tasks[task_id]["test"][0]["input"]
    grids = [g for g, _ in candidates[0]]
    while len(grids) < 2:
        grids.append(test_input)
    submission = {task_id: [{"attempt_1": grids[0], "attempt_2": grids[1]}]}
    run = {
        "config": cfg, "resolved_model": model, "start_epoch": start,
        "end_epoch": time.time(), "report": report, "answers": answers,
        "submission": submission,
    }
    trace_path = args.output_dir / "sealed_trace.json"
    atomic_json(trace_path, run)
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (args.output_dir / "sealed_trace.sha256").write_text(f"{digest}  sealed_trace.json\n")

    # The held-out solution is first opened after immutable output/trace bytes exist.
    solutions = json.loads((args.data_dir / "sealed_labels/solution.json").read_text())
    total, pct, per = score(submission, solutions)
    result = {
        "task_id": task_id, "trace_sha256": digest, "score": total,
        "percent": pct, "per_task": per, "wall_s": time.time() - start,
        "completion_tokens": sum(a["gen_tokens"] for a in report["attempts"]),
        "prompt_tokens": sum(a["prompt_tokens"] for a in report["attempts"]),
    }
    result["completion_tokens_per_s"] = result["completion_tokens"] / max(result["wall_s"], 1e-9)
    atomic_json(args.output_dir / "score.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
