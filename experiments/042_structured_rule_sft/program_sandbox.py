"""Restrictive validator and subprocess runner for model-generated grid programs."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Any


Grid = list[list[int]]
MARKER = "@@EXP042_RESULT@@"
BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}
SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
SAFE_METHODS = {
    "add",
    "append",
    "astype",
    "clear",
    "copy",
    "count",
    "discard",
    "extend",
    "fill",
    "flatten",
    "get",
    "index",
    "insert",
    "items",
    "keys",
    "max",
    "min",
    "pop",
    "popleft",
    "ravel",
    "remove",
    "reshape",
    "reverse",
    "setdefault",
    "sort",
    "sum",
    "tolist",
    "transpose",
    "update",
    "values",
}
SAFE_NUMPY = {
    "abs",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argwhere",
    "array",
    "asarray",
    "bool_",
    "clip",
    "concatenate",
    "copy",
    "count_nonzero",
    "cumsum",
    "diag",
    "diff",
    "eye",
    "empty",
    "empty_like",
    "flip",
    "fliplr",
    "flipud",
    "full",
    "full_like",
    "hstack",
    "indices",
    "int8",
    "int16",
    "int32",
    "int64",
    "isin",
    "logical_and",
    "logical_not",
    "logical_or",
    "max",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "nonzero",
    "ones",
    "ones_like",
    "pad",
    "repeat",
    "roll",
    "rot90",
    "stack",
    "sum",
    "tile",
    "transpose",
    "uint8",
    "unique",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
}
SAFE_IMPORTED_CALLS = {"Counter", "defaultdict", "deque"}
BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


class ProgramRejected(ValueError):
    pass


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def validate_program(code: str) -> ast.Module:
    if len(code) > 20_000:
        raise ProgramRejected("program exceeds 20,000 characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ProgramRejected(f"syntax error: {error.msg}") from error
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "transform" not in functions:
        raise ProgramRejected("program has no transform(grid) function")
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise ProgramRejected(f"blocked syntax: {type(node).__name__}")
        if isinstance(node, ast.Import):
            if any(alias.name != "numpy" or alias.asname not in {None, "np"} for alias in node.names):
                raise ProgramRejected("only numpy may be imported")
        if isinstance(node, ast.ImportFrom):
            allowed = {
                "collections": {"Counter", "defaultdict", "deque"},
                "typing": {"Any", "Dict", "List", "Optional", "Set", "Tuple"},
            }
            names = {alias.name for alias in node.names}
            if node.module not in allowed or not names <= allowed[node.module]:
                raise ProgramRejected("unsupported from-import")
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                raise ProgramRejected(f"blocked name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ProgramRejected(f"blocked attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS | SAFE_IMPORTED_CALLS | functions:
                    raise ProgramRejected(f"blocked call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                root = _root_name(node.func)
                if root == "np":
                    if not isinstance(node.func.value, ast.Name) or node.func.attr not in SAFE_NUMPY:
                        raise ProgramRejected(f"blocked numpy call: {ast.unparse(node.func)}")
                elif node.func.attr not in SAFE_METHODS:
                    raise ProgramRejected(f"blocked method call: {node.func.attr}")
            else:
                raise ProgramRejected("indirect calls are blocked")
    return tree


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    return (
        1 <= width <= 30
        and all(len(row) == width for row in value)
        and all(
            isinstance(cell, int)
            and not isinstance(cell, bool)
            and 0 <= cell <= 9
            for row in value
            for cell in row
        )
    )


RUNNER = r'''
import contextlib
import io
import json
import traceback
import numpy as np
import resource

MARKER = "@@EXP042_RESULT@@"
job = json.loads(input())
try:
    memory_bytes = int(job["memory_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
except (KeyError, OSError, ValueError):
    pass
namespace = {"np": np, "__name__": "__arc_program__"}
sink = io.StringIO()
results = []
try:
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(job["code"], "<arc-program>", "exec"), namespace)
    transform = namespace["transform"]
    for grid in job["inputs"]:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            value = transform(grid)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        results.append({"output": value})
except Exception as error:
    message = "".join(traceback.format_exception_only(type(error), error)).strip()[-500:]
    results = [{"error": message} for _ in job["inputs"]]
print(MARKER + json.dumps(results, separators=(",", ":")))
'''


def run_program(
    code: str,
    inputs: list[Grid],
    timeout_seconds: float = 3.0,
    memory_bytes: int = 1 << 30,
) -> list[dict[str, Any]]:
    try:
        validate_program(code)
    except ProgramRejected as error:
        return [{"error": str(error)} for _ in inputs]
    try:
        with tempfile.TemporaryDirectory(prefix="exp042-sandbox-") as directory:
            process = subprocess.run(
                [sys.executable, "-c", RUNNER],
                input=json.dumps(
                    {"code": code, "inputs": inputs, "memory_bytes": memory_bytes}
                )
                + "\n",
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
    except subprocess.TimeoutExpired:
        return [{"error": "program timed out"} for _ in inputs]
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                values = json.loads(line[len(MARKER) :])
            except json.JSONDecodeError:
                break
            if isinstance(values, list) and len(values) == len(inputs):
                return [
                    value
                    if isinstance(value, dict)
                    and ("error" in value or valid_grid(value.get("output")))
                    else {"error": "program returned an invalid grid"}
                    for value in values
                ]
    stderr = " | ".join(process.stderr.strip().splitlines()[-2:])[-500:]
    return [{"error": stderr or "program subprocess failed"} for _ in inputs]
