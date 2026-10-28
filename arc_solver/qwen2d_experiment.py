"""Deterministic design, corpus, lifecycle, and statistics for Experiment 021."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

from arc_solver.qwen2d_metadata import serialize_grid


MAIN_ARMS = ("text_control", "rowcol", "rowcol_component")
NEGATIVE_ARMS = ("shuffled_rowcol", "shuffled_component")
SYNTHETIC_FAMILIES = (
    "reconstruction",
    "indexed_cell",
    "row_extraction",
    "column_extraction",
    "transpose",
    "d4",
    "translation",
    "component_count",
    "component_bbox",
    "component_membership",
    "copy_component",
    "recolor_component",
)

# Experiment 034 freezes this partition before generating any corpus.  The
# tuples (rather than sets) also define deterministic within-stage row order.
PERCEPTION_FAMILIES = (
    "reconstruction",
    "indexed_cell",
    "row_extraction",
    "column_extraction",
    "component_count",
    "component_bbox",
    "component_membership",
)
INTERPRETATION_FAMILIES = (
    "transpose",
    "d4",
    "translation",
    "copy_component",
    "recolor_component",
)
PERCEPTION_STAGE_TOKEN_BUDGET = 600_000
INTERPRETATION_STAGE_TOKEN_BUDGET = 900_000
STAGE_TOKEN_BUDGETS = {
    "perception": PERCEPTION_STAGE_TOKEN_BUDGET,
    "interpretation": INTERPRETATION_STAGE_TOKEN_BUDGET,
}
REFINER_UNKNOWN = 10
REFINER_OUTSIDE = 11


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def corrupt_refiner_draft(
    target: Sequence[Sequence[int]],
    *,
    seed: int,
    severity: float = 0.35,
    max_size: int = 30,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Create a deterministic, possibly wrong-shaped draft for denoising.

    Values 0..9 are ARC colors and 10 denotes an unknown cell.  The returned
    rectangle is intentionally allowed to differ from the target dimensions;
    the refiner's shape heads must repair that error.
    """
    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must be in (0, 1]")
    if not target or not target[0]:
        raise ValueError("target must be a non-empty rectangular grid")
    height, width = len(target), len(target[0])
    if height > max_size or width > max_size:
        raise ValueError("target exceeds max_size")
    if any(len(row) != width for row in target):
        raise ValueError("target must be rectangular")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9
        for row in target
        for value in row
    ):
        raise ValueError("target colors must be integers in 0..9")

    rng = random.Random(seed)
    draft = [list(row) for row in target]
    operations: list[str] = []

    # Shape corruption is independent of cell corruption and remains valid for
    # 1x1 and 30x30 edge cases.
    shape_choices = []
    if height > 1:
        shape_choices.append("crop_row")
    if width > 1:
        shape_choices.append("crop_col")
    if height < max_size:
        shape_choices.append("add_row")
    if width < max_size:
        shape_choices.append("add_col")
    if shape_choices and rng.random() < severity:
        shape_op = rng.choice(shape_choices)
        operations.append(shape_op)
        if shape_op == "crop_row":
            draft.pop()
        elif shape_op == "crop_col":
            draft = [row[:-1] for row in draft]
        elif shape_op == "add_row":
            draft.append([rng.randrange(10) for _ in range(len(draft[0]))])
        else:
            for row in draft:
                row.append(rng.randrange(10))

    cell_count = len(draft) * len(draft[0])
    edit_count = max(1, round(cell_count * severity))
    positions = [(row, col) for row in range(len(draft)) for col in range(len(draft[0]))]
    rng.shuffle(positions)
    unknown_count = wrong_color_count = 0
    for row, col in positions[:edit_count]:
        if rng.random() < 0.5:
            draft[row][col] = REFINER_UNKNOWN
            unknown_count += 1
        else:
            previous = draft[row][col]
            choices = [color for color in range(10) if color != previous]
            draft[row][col] = rng.choice(choices)
            wrong_color_count += 1
    operations.append("cell_corruption")
    return draft, {
        "seed": seed,
        "severity": severity,
        "operations": operations,
        "unknown_cells": unknown_count,
        "wrong_color_cells": wrong_color_count,
        "target_shape": [height, width],
        "draft_shape": [len(draft), len(draft[0])],
    }


def refiner_prompt_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return the exact label-blind object available to a refiner worker."""
    required = {"train", "query", "draft"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"refiner row missing prompt fields: {missing}")
    return {key: copy.deepcopy(row[key]) for key in ("train", "query", "draft")}


def make_refiner_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    variants_per_row: int = 3,
    severities: Sequence[float] = (0.15, 0.35, 0.60),
) -> list[dict[str, Any]]:
    """Expand supervised rows into deterministic draft-to-clean examples."""
    if variants_per_row <= 0:
        raise ValueError("variants_per_row must be positive")
    if not rows:
        raise ValueError("rows cannot be empty")
    if not severities:
        raise ValueError("severities cannot be empty")
    output = []
    for row in rows:
        if not {"row_id", "train", "query", "target"} <= set(row):
            raise ValueError("source row lacks required refiner fields")
        for variant in range(variants_per_row):
            digest = hashlib.sha256(
                f"{seed}:{row['row_id']}:{variant}".encode()
            ).digest()
            draft_seed = int.from_bytes(digest[:8], "big")
            severity = float(severities[variant % len(severities)])
            draft, corruption = corrupt_refiner_draft(
                row["target"], seed=draft_seed, severity=severity
            )
            item = copy.deepcopy(row)
            item.update(
                {
                    "row_id": f"{row['row_id']}:refiner:{variant}",
                    "source_row_id": row["row_id"],
                    "draft": draft,
                    "corruption": corruption,
                }
            )
            output.append(item)
    return output


def refiner_corpus_audit(
    rows: Sequence[dict[str, Any]],
    *,
    global_task_ids: Iterable[str],
    heldout_task_ids: Iterable[str],
) -> dict[str, Any]:
    """Audit task partitioning and prove clean labels are absent from prompts."""
    overlap = sorted(set(global_task_ids) & set(heldout_task_ids))
    prompt_exposes_target_field = []
    duplicate_ids = []
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("row_id")
        if row_id in seen:
            duplicate_ids.append(row_id)
        seen.add(row_id)
        prompt = refiner_prompt_payload(row)
        if "target" in prompt or set(prompt) != {"train", "query", "draft"}:
            prompt_exposes_target_field.append(row_id)
    report = {
        "task_id_overlap": overlap,
        "prompt_exposes_target_field": sorted(prompt_exposes_target_field),
        "duplicate_row_ids": sorted(duplicate_ids),
    }
    report["passed"] = not any(report.values())
    if not report["passed"]:
        raise AssertionError(f"refiner corpus audit failed: {report}")
    return report


def _components(grid: list[list[int]]) -> list[list[tuple[int, int]]]:
    height, width = len(grid), len(grid[0])
    unseen = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != 0
    }
    output = []
    while unseen:
        root = min(unseen)
        color = grid[root[0]][root[1]]
        component, stack = [], [root]
        unseen.remove(root)
        while stack:
            cell = stack.pop()
            component.append(cell)
            r, c = cell
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in unseen and grid[nxt[0]][nxt[1]] == color:
                    unseen.remove(nxt)
                    stack.append(nxt)
        output.append(sorted(component))
    return sorted(output, key=lambda component: (-len(component), component[0]))


def task_stratum(task: dict[str, Any]) -> str:
    """A label-blind descriptor derived only from published training pairs."""
    pairs = task["train"]
    max_dim = max(
        max(len(pair[key]), len(pair[key][0]))
        for pair in pairs
        for key in ("input", "output")
    )
    dimension = "small" if max_dim <= 10 else "medium" if max_dim <= 20 else "large"
    component_count = int(
        round(sum(len(_components(pair["input"])) for pair in pairs) / len(pairs))
    )
    components = "few" if component_count <= 2 else "some" if component_count <= 6 else "many"
    relations = set()
    for pair in pairs:
        in_shape = (len(pair["input"]), len(pair["input"][0]))
        out_shape = (len(pair["output"]), len(pair["output"][0]))
        relations.add("same" if in_shape == out_shape else "changed")
    relation = next(iter(relations)) if len(relations) == 1 else "mixed"
    return f"{dimension}:{components}:{relation}"


def make_task_manifest(
    challenges: dict[str, Any],
    excluded_ids: Iterable[str],
    *,
    seed: int,
    reserve_count: int = 100,
    primary_count: int = 32,
    negative_count: int = 12,
    replay_count: int = 4,
) -> dict[str, Any]:
    """Reserve tasks and select primary ids before any arm executes."""
    if not 1 <= primary_count <= reserve_count:
        raise ValueError("primary_count must be in 1..reserve_count")
    excluded = sorted(set(excluded_ids))
    eligible = sorted(set(challenges) - set(excluded))
    if len(eligible) < reserve_count:
        raise ValueError(f"need {reserve_count} eligible tasks, found {len(eligible)}")
    rng = random.Random(seed)
    keyed = sorted(eligible, key=lambda task_id: (rng.random(), task_id))
    reserved = sorted(keyed[:reserve_count])
    global_train = sorted(keyed[reserve_count:])
    strata: dict[str, list[str]] = defaultdict(list)
    for task_id in reserved:
        strata[task_stratum(challenges[task_id])].append(task_id)
    # Round-robin strata avoids an outcome-dependent hand selection.
    queues = {key: sorted(values) for key, values in strata.items()}
    primary: list[str] = []
    while len(primary) < primary_count and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(primary) < primary_count:
                primary.append(queues[key].pop(0))
    if len(primary) != primary_count:
        raise ValueError(
            f"primary set must contain {primary_count} tasks, found {len(primary)}"
        )
    negative = sorted(primary[:negative_count])
    replay = sorted(primary[:replay_count])
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "excluded_ids": excluded,
        "global_train_ids": global_train,
        "reserved_ids": reserved,
        "primary_ids": sorted(primary),
        "negative_control_ids": negative,
        "replay_ids": replay,
        "strata": {key: sorted(value) for key, value in strata.items()},
        "selection_uses_test_outputs": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def make_arc_loo_rows(
    challenges: dict[str, Any], task_ids: Iterable[str]
) -> list[dict[str, Any]]:
    rows = []
    for task_id in sorted(task_ids):
        pairs = challenges[task_id]["train"]
        for held_index, held in enumerate(pairs):
            rows.append(
                {
                    "row_id": f"arc:{task_id}:{held_index}",
                    "source": "arc_training_loo",
                    "family": "arc_loo",
                    "source_task_id": task_id,
                    "seed": held_index,
                    "train": copy.deepcopy(
                        [pair for index, pair in enumerate(pairs) if index != held_index]
                    ),
                    "query": copy.deepcopy(held["input"]),
                    "target": copy.deepcopy(held["output"]),
                }
            )
    return rows


def _base_grid(
    rng: random.Random, shape: tuple[int, int] | None = None
) -> list[list[int]]:
    height, width = shape or (rng.randint(4, 9), rng.randint(4, 9))
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for _ in range(rng.randint(2, 4)):
        color = rng.randint(1, 9)
        row, col = rng.randrange(height), rng.randrange(width)
        for _ in range(rng.randint(1, 5)):
            grid[row][col] = color
            row = min(height - 1, max(0, row + rng.choice((-1, 0, 1))))
            col = min(width - 1, max(0, col + rng.choice((-1, 0, 1))))
    return grid


def _transform_family(
    family: str, grid: list[list[int]], rng: random.Random
) -> tuple[list[list[int]], dict[str, Any]]:
    height, width = len(grid), len(grid[0])
    if family == "reconstruction":
        return copy.deepcopy(grid), {}
    if family == "indexed_cell":
        row, col = rng.randrange(height), rng.randrange(width)
        return [[grid[row][col]]], {"row": row, "col": col}
    if family == "row_extraction":
        row = rng.randrange(height)
        return [grid[row][:]], {"row": row}
    if family == "column_extraction":
        col = rng.randrange(width)
        return [[grid[row][col]] for row in range(height)], {"col": col}
    if family == "transpose":
        return [list(row) for row in zip(*grid)], {}
    if family == "d4":
        operation = rng.choice(("rot90", "rot180", "flip"))
        if operation == "rot90":
            target = [list(row) for row in zip(*grid[::-1])]
        elif operation == "rot180":
            target = [row[::-1] for row in grid[::-1]]
        else:
            target = [row[::-1] for row in grid]
        return target, {"operation": operation}
    if family == "translation":
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        target = [[0 for _ in range(width)] for _ in range(height)]
        for row in range(height):
            for col in range(width):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    target[nr][nc] = grid[row][col]
        return target, {"dr": dr, "dc": dc}
    components = _components(grid)
    selected = components[0] if components else [(0, 0)]
    if family == "component_count":
        return [[min(9, len(components))]], {}
    if family == "component_bbox":
        rows, cols = zip(*selected)
        return [
            grid[row][min(cols) : max(cols) + 1]
            for row in range(min(rows), max(rows) + 1)
        ], {}
    if family == "component_membership":
        selected_set = set(selected)
        return [
            [1 if (row, col) in selected_set else 0 for col in range(width)]
            for row in range(height)
        ], {}
    if family == "copy_component":
        selected_set = set(selected)
        target = [row[:] for row in grid]
        shift = 1 if max(col for _, col in selected) + 1 < width else -1
        for row, col in selected:
            if 0 <= col + shift < width:
                target[row][col + shift] = grid[row][col]
        return target, {"dc": shift}
    if family == "recolor_component":
        selected_set = set(selected)
        target = [row[:] for row in grid]
        color = next(value for value in range(1, 10) if value != grid[selected[0][0]][selected[0][1]])
        for row, col in selected_set:
            target[row][col] = color
        return target, {"color": color}
    raise ValueError(f"unknown synthetic family {family}")


def make_synthetic_rows(
    *,
    seeds_per_family: int,
    seed: int,
    heldout_modulus: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic train/probe rows split by family-local seed."""
    train, heldout = [], []
    for family_index, family in enumerate(SYNTHETIC_FAMILIES):
        for local_seed in range(seeds_per_family):
            row_seed = seed + family_index * 1_000_003 + local_seed
            rng = random.Random(row_seed)
            shape = (rng.randint(4, 9), rng.randint(4, 9))
            grids = [_base_grid(rng, shape) for _ in range(3)]
            transformed = [
                _transform_family(family, grid, random.Random(row_seed + 991))
                for grid in grids
            ]
            target, params = transformed[-1]
            row = {
                "row_id": f"synthetic:{family}:{local_seed:06d}",
                "source": "deterministic_synthetic",
                "family": family,
                "seed": local_seed,
                "train": [
                    {"input": grid, "output": pair[0]}
                    for grid, pair in zip(grids[:2], transformed[:2])
                ],
                "query": grids[-1],
                "target": target,
                "parameters": params,
            }
            (heldout if local_seed % heldout_modulus == 0 else train).append(row)
    return train, heldout


def loss_token_count(row: dict[str, Any]) -> int:
    """Number of assistant tokens bearing loss, including each output EOS."""
    demo_tokens = sum(
        len(serialize_grid(pair["output"])) + 1 for pair in row.get("train", [])
    )
    return demo_tokens + len(serialize_grid(row["target"])) + 1


def fixed_token_budget(
    rows: Sequence[dict[str, Any]], token_budget: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically cycle rows and mask the final excess answer tokens.

    The source row remains intact; ``loss_token_limit`` controls the label mask,
    making the loss-bearing budget exact without changing prompt tokens.
    """
    if token_budget <= 0 or not rows:
        raise ValueError("positive budget and non-empty rows required")
    selected, used, index = [], 0, 0
    while used < token_budget:
        row = copy.deepcopy(rows[index % len(rows)])
        count = loss_token_count(row)
        take = min(count, token_budget - used)
        row["epoch_cycle"] = index // len(rows)
        row["loss_token_limit"] = take
        row["row_instance_id"] = f"{row['row_id']}@{row['epoch_cycle']}"
        selected.append(row)
        used += take
        index += 1
    ids = [row["row_instance_id"] for row in selected]
    manifest = {
        "token_budget": token_budget,
        "loss_tokens": used,
        "row_count": len(selected),
        "unique_source_rows": len(set(row["row_id"] for row in selected)),
        "ordered_row_ids_sha256": sha256_json(ids),
        "rows_sha256": sha256_json(selected),
    }
    return selected, manifest


def make_staged_corpora(
    train_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
    *,
    perception_token_budget: int = PERCEPTION_STAGE_TOKEN_BUDGET,
    interpretation_token_budget: int = INTERPRETATION_STAGE_TOKEN_BUDGET,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Partition synthetic rows and materialize exact Experiment 034 budgets.

    Probe rows remain disjoint from the cycled, loss-bearing training corpora.
    The returned manifest is sufficient to verify the family assignment,
    source rows, probe partition, and exact token budgets without labels.
    """
    assignments = {
        "perception": PERCEPTION_FAMILIES,
        "interpretation": INTERPRETATION_FAMILIES,
    }
    assigned = [family for families in assignments.values() for family in families]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("synthetic family is assigned to more than one stage")
    if set(assigned) != set(SYNTHETIC_FAMILIES):
        missing = sorted(set(SYNTHETIC_FAMILIES) - set(assigned))
        extra = sorted(set(assigned) - set(SYNTHETIC_FAMILIES))
        raise AssertionError(
            f"staged family assignment differs from known families: "
            f"missing={missing}, extra={extra}"
        )

    # ARC leave-one-out rows are a second interpretation source rather than a
    # synthetic family.  They are allowed only in training, never in probes.
    known = set(SYNTHETIC_FAMILIES) | {"arc_loo"}
    for split_name, rows in (("train", train_rows), ("probe", probe_rows)):
        unknown = sorted(
            {
                row.get("family")
                for row in rows
                if row.get("family") not in known
            },
            key=str,
        )
        if unknown:
            raise ValueError(f"{split_name} rows contain unknown families: {unknown}")
        if split_name == "probe" and any(
            row.get("family") == "arc_loo" for row in rows
        ):
            raise ValueError("ARC LOO rows cannot be used as held-out synthetic probes")

    train_ids = [row.get("row_id") for row in train_rows]
    probe_ids = [row.get("row_id") for row in probe_rows]
    if any(row_id is None for row_id in train_ids + probe_ids):
        raise ValueError("every synthetic row must have a row_id")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("duplicate row_id in synthetic train rows")
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("duplicate row_id in synthetic probe rows")
    overlap = sorted(set(train_ids) & set(probe_ids))
    if overlap:
        raise ValueError(f"synthetic train/probe row overlap: {overlap}")

    budgets = {
        "perception": perception_token_budget,
        "interpretation": interpretation_token_budget,
    }
    corpora: dict[str, dict[str, list[dict[str, Any]]]] = {}
    stages: dict[str, Any] = {}
    for stage, families in assignments.items():
        family_set = set(families)
        if stage == "interpretation":
            family_set.add("arc_loo")
        source_train = [
            copy.deepcopy(row) for row in train_rows if row["family"] in family_set
        ]
        stage_probes = [
            copy.deepcopy(row) for row in probe_rows if row["family"] in family_set
        ]
        missing_rows = sorted(
            family for family in families
            if not any(row["family"] == family for row in source_train)
        )
        if missing_rows:
            raise ValueError(f"{stage} train rows missing families: {missing_rows}")
        budgeted_train, budget_manifest = fixed_token_budget(
            source_train, budgets[stage]
        )
        corpora[stage] = {"train": budgeted_train, "probe": stage_probes}
        stages[stage] = {
            "families": list(families),
            "source_train_row_count": len(source_train),
            "source_train_rows_sha256": sha256_json(source_train),
            "probe_row_count": len(stage_probes),
            "ordered_probe_row_ids_sha256": sha256_json(
                [row["row_id"] for row in stage_probes]
            ),
            "probe_rows_sha256": sha256_json(stage_probes),
            "train": budget_manifest,
        }

    manifest = {
        "schema_version": 1,
        "family_assignment": {
            stage: list(families) for stage, families in assignments.items()
        },
        "known_families_assigned_exactly_once": True,
        "train_probe_row_id_overlap": [],
        "stages": stages,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return corpora, manifest


# Descriptive alias retained for callers that prefer to make the input domain
# explicit; both names intentionally expose the same stable API.
make_staged_synthetic_corpora = make_staged_corpora


def collision_audit(
    global_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
    global_task_ids: Iterable[str],
    heldout_task_ids: Iterable[str],
) -> dict[str, Any]:
    train_keys = {
        sha256_json({"query": row["query"], "target": row["target"]})
        for row in global_rows
    }
    probe_keys = {
        sha256_json({"query": row["query"], "target": row["target"]})
        for row in probe_rows
    }
    task_overlap = sorted(set(global_task_ids) & set(heldout_task_ids))
    row_overlap = sorted(train_keys & probe_keys)
    report = {
        "task_id_overlap": task_overlap,
        "synthetic_pair_hash_overlap": row_overlap,
        "passed": not task_overlap and not row_overlap,
    }
    if not report["passed"]:
        raise AssertionError(f"corpus collision audit failed: {report}")
    return report


class LabelLifecycle:
    """Filesystem barrier preventing held-out labels from entering the solver."""

    def __init__(self, root: str | Path, arms: Sequence[str], task_ids: Sequence[str]):
        self.root = Path(root)
        self.arms = tuple(arms)
        self.task_ids = tuple(task_ids)

    def record_candidates(self, arm: str, task_id: str, candidate_file: str | Path) -> str:
        if arm not in self.arms or task_id not in self.task_ids:
            raise ValueError("candidate is outside preregistered arm/task set")
        path = Path(candidate_file)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        marker = self.root / "candidate_hashes" / arm
        marker.mkdir(parents=True, exist_ok=True)
        (marker / f"{task_id}.sha256").write_text(digest + "\n")
        return digest

    def assert_ready_for_labels(self) -> None:
        missing = [
            f"{arm}/{task_id}"
            for arm, task_id in itertools.product(self.arms, self.task_ids)
            if not (self.root / "candidate_hashes" / arm / f"{task_id}.sha256").is_file()
        ]
        if missing:
            raise RuntimeError(f"labels remain sealed; missing candidate hashes: {missing}")

    def materialize_labels(self, labels: dict[str, Any]) -> Path:
        self.assert_ready_for_labels()
        if set(labels) != set(self.task_ids):
            raise ValueError("label ids differ from preregistered task ids")
        path = self.root / "parent_evaluator_only" / "labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(labels))
        return path


def paired_randomization(
    differences: Sequence[float], *, exact_limit: int = 20, draws: int = 100_000, seed: int = 21021
) -> float:
    values = [float(value) for value in differences if value != 0]
    if not values:
        return 1.0
    observed = abs(sum(values))
    if len(values) <= exact_limit:
        totals = (
            abs(sum(sign * value for sign, value in zip(signs, values)))
            for signs in itertools.product((-1, 1), repeat=len(values))
        )
        extreme = sum(total >= observed - 1e-12 for total in totals)
        return extreme / (2 ** len(values))
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(rng.choice((-1, 1)) * value for value in values)) >= observed - 1e-12
        for _ in range(draws)
    )
    return (extreme + 1) / (draws + 1)


def bootstrap_mean_ci(
    differences: Sequence[float], *, draws: int = 20_000, seed: int = 21021
) -> tuple[float, float]:
    if not differences:
        raise ValueError("differences cannot be empty")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(float(differences[rng.randrange(n)]) for _ in range(n)) / n
        for _ in range(draws)
    )
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def paired_contrast(
    task_metrics: dict[str, dict[str, float]], arm_a: str, arm_b: str
) -> dict[str, Any]:
    task_ids = sorted(set(task_metrics[arm_a]) & set(task_metrics[arm_b]))
    differences = [
        float(task_metrics[arm_b][task_id]) - float(task_metrics[arm_a][task_id])
        for task_id in task_ids
    ]
    wins = [task_id for task_id, difference in zip(task_ids, differences) if difference > 0]
    losses = [task_id for task_id, difference in zip(task_ids, differences) if difference < 0]
    return {
        "contrast": f"{arm_b}-{arm_a}",
        "n_tasks": len(task_ids),
        "mean_difference": sum(differences) / len(differences) if differences else None,
        "bootstrap_95_ci": bootstrap_mean_ci(differences) if differences else None,
        "paired_randomization_p": paired_randomization(differences),
        "wins": wins,
        "losses": losses,
        "ties": len(task_ids) - len(wins) - len(losses),
        "low_power_warning": len(task_ids) < 50,
    }
