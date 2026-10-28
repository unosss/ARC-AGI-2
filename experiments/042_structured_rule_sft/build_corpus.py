"""Build paired direct/program SFT episodes from public NVARC artifacts."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from config_fingerprint import corpus_config_sha256
from program_sandbox import run_program, validate_program


Grid = list[list[int]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    if not 1 <= width <= 30 or any(len(row) != width for row in value):
        return False
    return all(
        isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9
        for row in value
        for cell in row
    )


def grid_key(grid: Grid) -> str:
    return canonical_json(grid)


def load_pairs(path: Path) -> list[dict[str, Grid]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) < 4:
        raise ValueError("pair file must contain at least four pairs")
    result = []
    for pair in data:
        if not isinstance(pair, dict):
            raise ValueError("pair is not an object")
        input_grid, output_grid = pair.get("input"), pair.get("output")
        if not valid_grid(input_grid) or not valid_grid(output_grid):
            raise ValueError("invalid input/output grid")
        result.append({"input": input_grid, "output": output_grid})
    return result


def strip_markdown_fence(code: str) -> str:
    text = code.strip()
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


def canonical_program(code: str) -> str:
    """Normalize recorded code and expose the fixed transform(grid) interface."""
    tree = ast.parse(strip_markdown_fence(code))
    body: list[ast.stmt] = []
    has_generator = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                continue
            has_generator = has_generator or node.name == "generate_puzzle_output"
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            # Drop conventional ``if __name__ == ...`` blocks.
            if "__name__" in ast.unparse(node.test):
                continue
        body.append(node)
    if not has_generator:
        raise ValueError("program has no generate_puzzle_output function")
    wrapper = ast.parse(
        "\n\ndef transform(grid):\n"
        "    input_grid = np.array(grid, dtype=np.int8)\n"
        "    return generate_puzzle_output(input_grid)\n"
    ).body
    normalized = ast.Module(body=body + wrapper, type_ignores=[])
    ast.fix_missing_locations(normalized)
    program = ast.unparse(normalized).strip() + "\n"
    validate_program(program)
    return program


def select_matching_program(
    pairs: list[dict[str, Grid]],
    candidates: list[tuple[int, str]],
    timeout_seconds: float,
    memory_bytes: int,
) -> tuple[str, int]:
    inputs = [pair["input"] for pair in pairs]
    expected = [pair["output"] for pair in pairs]
    rejected = 0
    for sid, completion in sorted(candidates):
        try:
            program = canonical_program(completion)
        except (SyntaxError, ValueError):
            rejected += 1
            continue
        results = run_program(program, inputs, timeout_seconds, memory_bytes)
        if [item.get("output") for item in results] == expected:
            return program, sid
    raise ValueError(
        f"no published program replays every accepted pair ({rejected} rejected)"
    )


def pair_program_key(pair_path: Path, pairs_root: Path) -> tuple[str, str]:
    relative = pair_path.relative_to(pairs_root)
    puzzle_name1 = relative.parent.name
    stem = relative.stem
    prefix = puzzle_name1 + "_"
    if not stem.startswith(prefix) or len(stem) <= len(prefix):
        raise ValueError("pair path does not encode puzzle_name1/puzzle_name2")
    return puzzle_name1, stem[len(prefix) :]


def load_program_index_jsonl(
    path: Path, wanted: set[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[int, str]]]:
    result: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["puzzle_name1"]), str(row["puzzle_name2"]))
        if key in wanted:
            result.setdefault(key, []).append((int(row["sid"]), row["completion"]))
    return result


def load_program_parquet(
    paths: list[Path], wanted: set[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[int, str]]]:
    try:
        import pyarrow.dataset as parquet_dataset
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read NVARC output parquet") from error
    first_names = sorted({key[0] for key in wanted})
    dataset = parquet_dataset.dataset([str(path) for path in paths], format="parquet")
    scanner = dataset.scanner(
        columns=["puzzle_name1", "puzzle_name2", "sid", "completion"],
        filter=parquet_dataset.field("puzzle_name1").isin(first_names),
        batch_size=256,
    )
    result: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for batch in scanner.to_batches():
        values = batch.to_pylist()
        for row in values:
            key = (str(row["puzzle_name1"]), str(row["puzzle_name2"]))
            if key in wanted:
                result.setdefault(key, []).append(
                    (int(row["sid"]), str(row["completion"]))
                )
    return result


def deterministic_pair_order(
    pairs: list[dict[str, Grid]], seed: int, puzzle_key: str
) -> list[dict[str, Grid]]:
    ordered = list(pairs)
    digest = hashlib.sha256(f"{seed}:pairs:{puzzle_key}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def episode_from_paths(
    pair_path: Path,
    pairs_root: Path,
    candidates: list[tuple[int, str]],
    split: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    relative = pair_path.relative_to(pairs_root)
    puzzle_key = relative.with_suffix("").as_posix()
    pairs = load_pairs(pair_path)
    program, sid = select_matching_program(
        pairs,
        candidates,
        float(config["program_timeout_seconds"]),
        int(config["program_memory_bytes"]),
    )
    max_chars = int(config.get("max_program_characters", 16000))
    if len(program) > max_chars:
        raise ValueError(f"canonical program exceeds {max_chars} characters")
    ordered = deterministic_pair_order(pairs, int(config["split_seed"]), puzzle_key)
    demo_count = int(config["demo_count"])
    if len(ordered) <= demo_count:
        raise ValueError("not enough pairs for demonstrations plus query")
    demos = ordered[:demo_count]
    query = ordered[demo_count]
    return {
        "id": sha256_bytes(f"exp042:{puzzle_key}".encode())[:20],
        "puzzle_key": puzzle_key,
        "source_ids": list(pair_program_key(pair_path, pairs_root)),
        "split": split,
        "source_pair_sha256": file_sha256(pair_path),
        "source_program_sid": sid,
        "demos": demos,
        "query_input": query["input"],
        "query_output": query["output"],
        "program": program,
        "program_sha256": sha256_bytes(program.encode()),
    }


def stable_rule_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            sha256_bytes(f"{seed}:rule:{row['puzzle_key']}".encode()),
            row["puzzle_key"],
        ),
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text)


def challenge_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "puzzle_key": row["puzzle_key"],
        "demos": row["demos"],
        "query_input": row["query_input"],
    }


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    challenge = challenge_row(row)
    return {
        "id": row["id"],
        "challenge_sha256": sha256_bytes(canonical_json(challenge).encode()),
        "query_output": row["query_output"],
        "program": row["program"],
        "program_sha256": row["program_sha256"],
    }


def source_split(source_id: str, seed: int, thresholds: list[int]) -> str:
    value = int(sha256_bytes(f"{seed}:source-id:{source_id}".encode())[:8], 16) % 10_000
    if value < thresholds[0]:
        return "train"
    if value < thresholds[1]:
        return "validation"
    return "test"


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text())
    all_pair_paths = sorted(args.pairs_root.rglob("*.json"))
    required_by_split = {
        "train": max(int(value) for value in config["train_budgets"]),
        "validation": int(config["validation_rules"]),
        "test": int(config["test_rules"]),
    }
    required = sum(required_by_split.values())
    reserve_config = config.get("rule_reserve", 2000)
    if isinstance(reserve_config, dict):
        reserve_by_split = {
            split: int(reserve_config.get(split, 0)) for split in required_by_split
        }
    else:
        reserve_per_split = max(1, int(reserve_config) // 3)
        reserve_by_split = {
            split: reserve_per_split for split in required_by_split
        }
    thresholds = [int(value) for value in config["source_split_thresholds"]]
    if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1] < 10_000:
        raise ValueError("source_split_thresholds must be two increasing values")
    classified: dict[str, list[Path]] = {name: [] for name in required_by_split}
    cross_split_pairs = 0
    malformed_paths = 0
    for path in all_pair_paths:
        try:
            first, second = pair_program_key(path, args.pairs_root)
        except ValueError:
            malformed_paths += 1
            continue
        first_split = source_split(first, int(config["split_seed"]), thresholds)
        second_split = source_split(second, int(config["split_seed"]), thresholds)
        if first_split != second_split:
            cross_split_pairs += 1
            continue
        classified[first_split].append(path)
    pair_paths = []
    for split, needed in required_by_split.items():
        ordered_paths = sorted(
            classified[split],
            key=lambda path: (
                sha256_bytes(
                    f"{config['split_seed']}:{split}:"
                    f"{path.relative_to(args.pairs_root)}".encode()
                ),
                path.as_posix(),
            ),
        )
        pair_paths.extend(ordered_paths[: needed + reserve_by_split[split]])
    keyed_paths: dict[tuple[str, str], Path] = {}
    assigned_splits: dict[tuple[str, str], str] = {}
    for path in pair_paths:
        key = pair_program_key(path, args.pairs_root)
        if key in keyed_paths:
            raise ValueError(f"duplicate pair program key: {key}")
        keyed_paths[key] = path
        assigned_splits[key] = source_split(
            key[0], int(config["split_seed"]), thresholds
        )
    wanted = set(keyed_paths)
    if args.program_index is not None:
        programs = load_program_index_jsonl(args.program_index, wanted)
    else:
        parquet_paths = sorted(args.program_parquet_root.rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError("no output-program parquet files found")
        programs = load_program_parquet(parquet_paths, wanted)
    rows = []
    skipped: dict[str, int] = {}
    if malformed_paths:
        skipped["ValueError:pair path does not encode puzzle names"] = malformed_paths
    skipped["cross_source_split_pair"] = cross_split_pairs

    def process_keyed_path(item):
        key, pair_path = item
        try:
            candidates = programs.get(key)
            if not candidates:
                raise FileNotFoundError("matching published program rows are absent")
            return (
                episode_from_paths(
                    pair_path, args.pairs_root, candidates, assigned_splits[key], config
                ),
                None,
            )
        except (FileNotFoundError, ValueError, SyntaxError) as error:
            return None, type(error).__name__ + ":" + str(error)

    workers = int(config.get("corpus_workers", 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = executor.map(process_keyed_path, keyed_paths.items())
        for row, name in outcomes:
            if row is not None:
                rows.append(row)
            else:
                assert name is not None
                skipped[name] = skipped.get(name, 0) + 1

    program_splits: dict[str, set[str]] = {}
    for row in rows:
        program_splits.setdefault(row["program_sha256"], set()).add(row["split"])
    cross_split_programs = {
        digest for digest, splits in program_splits.items() if len(splits) > 1
    }
    if cross_split_programs:
        skipped["identical_program_cross_split"] = sum(
            row["program_sha256"] in cross_split_programs for row in rows
        )
    rows = [
        row for row in rows if row["program_sha256"] not in cross_split_programs
    ]
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, needed in required_by_split.items():
        seen_programs = set()
        used_source_ids = set()
        unique_rows = []
        for row in stable_rule_order(
            [value for value in rows if value["split"] == split],
            int(config["split_seed"]),
        ):
            if row["program_sha256"] in seen_programs:
                continue
            if split != "train" and any(
                source_id in used_source_ids for source_id in row["source_ids"]
            ):
                continue
            seen_programs.add(row["program_sha256"])
            used_source_ids.update(row["source_ids"])
            unique_rows.append(row)
        if len(unique_rows) < needed:
            raise RuntimeError(
                f"only {len(unique_rows)} usable {split} rules; requires {needed}. "
                "Increase rule_reserve after inspecting skip reasons."
            )
        selected[split] = unique_rows[:needed]
    train_rows = selected["train"]
    val_rows = selected["validation"]
    test_rows = selected["test"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.labels_dir.mkdir(parents=True, exist_ok=True)
    output_resolved = args.output_dir.resolve()
    labels_resolved = args.labels_dir.resolve()
    if (
        output_resolved == labels_resolved
        or labels_resolved.is_relative_to(output_resolved)
        or output_resolved.is_relative_to(labels_resolved)
    ):
        raise ValueError(
            "labels_dir and output_dir must be separate, non-nested directories"
        )
    files = {
        "train.jsonl": (args.output_dir / "train.jsonl", train_rows),
        "validation_challenges.jsonl": (
            args.output_dir / "validation_challenges.jsonl",
            map(challenge_row, val_rows),
        ),
        "validation_labels.jsonl": (
            args.labels_dir / "validation_labels.jsonl",
            map(label_row, val_rows),
        ),
        "test_challenges.jsonl": (
            args.output_dir / "test_challenges.jsonl",
            map(challenge_row, test_rows),
        ),
        "test_labels.jsonl": (
            args.labels_dir / "test_labels.jsonl",
            map(label_row, test_rows),
        ),
    }
    for path, values in files.values():
        write_jsonl(path, values)

    manifest = {
        "schema_version": 1,
        "experiment": 42,
        "config_sha256": file_sha256(args.config),
        "corpus_config_sha256": corpus_config_sha256(config),
        "source_pair_file_count": len(all_pair_paths),
        "candidate_pair_file_count": len(pair_paths),
        "usable_rule_count": len(rows),
        "selected_rule_count": required,
        "train_rule_count": len(train_rows),
        "validation_rule_count": len(val_rows),
        "test_rule_count": len(test_rows),
        "train_budgets": config["train_budgets"],
        "skipped": dict(sorted(skipped.items())),
        "files": {
            name: {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "sealed_label": path.parent.resolve() == args.labels_dir.resolve(),
            }
            for name, (path, _) in files.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--program-parquet-root", type=Path)
    source.add_argument("--program-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))
