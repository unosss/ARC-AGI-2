"""Generate fixed counterfactual relation artifacts and a private answer key."""

from __future__ import annotations

import json
from pathlib import Path


EXP = Path(__file__).resolve().parent
ARTIFACTS = EXP / "artifacts"


INCIDENCE_CASES = [
    {
        "id": "incidence_0",
        "dimensions": [12, 14],
        "guides": [
            {"orientation": "horizontal", "fixed": 5, "span": [0, 8]},
            {"orientation": "vertical", "fixed": 3, "span": [0, 11]},
        ],
        "components": [
            {"id": "C0", "rows": [4, 6], "cols": [5, 6]},
            {"id": "C1", "rows": [3, 4], "cols": [1, 2]},
            {"id": "C2", "rows": [7, 8], "cols": [4, 5]},
            {"id": "C3", "rows": [3, 4], "cols": [9, 10]},
            {"id": "C4", "rows": [9, 10], "cols": [10, 12]},
        ],
    },
    {
        "id": "incidence_1",
        "dimensions": [10, 10],
        "guides": [
            {"orientation": "horizontal", "fixed": 2, "span": [2, 7]},
            {"orientation": "vertical", "fixed": 7, "span": [2, 9]},
        ],
        "components": [
            {"id": "C0", "rows": [1, 2], "cols": [4, 5]},
            {"id": "C1", "rows": [3, 4], "cols": [2, 3]},
            {"id": "C2", "rows": [6, 7], "cols": [5, 6]},
            {"id": "C3", "rows": [0, 1], "cols": [8, 9]},
            {"id": "C4", "rows": [7, 8], "cols": [1, 2]},
        ],
    },
    {
        "id": "incidence_2",
        "dimensions": [9, 13],
        "guides": [
            {"orientation": "horizontal", "fixed": 6, "span": [4, 12]}
        ],
        "components": [
            {"id": "C0", "rows": [5, 7], "cols": [9, 10]},
            {"id": "C1", "rows": [3, 5], "cols": [5, 6]},
            {"id": "C2", "rows": [4, 5], "cols": [2, 3]},
            {"id": "C3", "rows": [7, 8], "cols": [11, 12]},
            {"id": "C4", "rows": [0, 1], "cols": [8, 9]},
        ],
    },
]

REFLECTION_CASES = [
    {
        "id": "reflection_0",
        "dimensions": [7, 8],
        "anchor": 3,
        "source_runs": [
            {"row": 1, "cols": [2, 2]},
            {"row": 2, "cols": [1, 2]},
            {"row": 3, "cols": [0, 2]},
            {"row": 4, "cols": [1, 1]},
            {"row": 5, "cols": [2, 2]},
        ],
    },
    {
        "id": "reflection_1",
        "dimensions": [8, 10],
        "anchor": 4,
        "source_runs": [
            {"row": 1, "cols": [3, 3]},
            {"row": 2, "cols": [2, 3]},
            {"row": 3, "cols": [1, 3]},
            {"row": 4, "cols": [0, 2]},
            {"row": 5, "cols": [2, 2]},
            {"row": 6, "cols": [1, 2]},
        ],
    },
]

SHAPE_CASES = [
    {
        "id": "shape_0",
        "dimensions": [11, 16],
        "prototypes": [
            {"name": "L3", "cells": [[0, 0], [1, 0], [1, 1]], "target_color": 4},
            {"name": "Z4", "cells": [[0, 0], [0, 1], [1, 1], [1, 2]], "target_color": 8},
            {"name": "I3", "cells": [[0, 0], [0, 1], [0, 2]], "target_color": 2},
        ],
        "queries": [
            {"id": "Q0", "cells": [[1, 1], [2, 1], [2, 2]]},
            {"id": "Q1", "cells": [[1, 5], [1, 6], [2, 5]]},
            {"id": "Q2", "cells": [[5, 1], [5, 2], [6, 2], [6, 3]]},
            {"id": "Q3", "cells": [[5, 6], [5, 7], [6, 5], [6, 6]]},
            {"id": "Q4", "cells": [[1, 12], [2, 12], [3, 12]]},
            {"id": "Q5", "cells": [[8, 10], [8, 11], [8, 12]]},
        ],
    },
    {
        "id": "shape_1",
        "dimensions": [10, 14],
        "prototypes": [
            {"name": "L3", "cells": [[0, 0], [1, 0], [1, 1]], "target_color": 3},
            {"name": "Z4", "cells": [[0, 0], [0, 1], [1, 1], [1, 2]], "target_color": 9},
        ],
        "queries": [
            {"id": "Q0", "cells": [[1, 1], [2, 1], [2, 2]]},
            {"id": "Q1", "cells": [[1, 6], [1, 7], [2, 7]]},
            {"id": "Q2", "cells": [[5, 1], [6, 1], [6, 2], [7, 2]]},
            {"id": "Q3", "cells": [[5, 7], [5, 8], [6, 6], [6, 7]]},
        ],
    },
]


def rectangle_cells(component: dict) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(component["rows"][0], component["rows"][1] + 1)
        for c in range(component["cols"][0], component["cols"][1] + 1)
    }


def guide_cells(guide: dict) -> set[tuple[int, int]]:
    if guide["orientation"] == "horizontal":
        return {(guide["fixed"], c) for c in range(guide["span"][0], guide["span"][1] + 1)}
    return {(r, guide["fixed"]) for r in range(guide["span"][0], guide["span"][1] + 1)}


def incidence_output(case: dict, contact: bool) -> list[list[int]]:
    height, width = case["dimensions"]
    grid = [[0] * width for _ in range(height)]
    guides = set().union(*(guide_cells(guide) for guide in case["guides"]))
    for component in case["components"]:
        for r, c in rectangle_cells(component):
            grid[r][c] = 2
    for r, c in guides:
        grid[r][c] = 1
    for component in case["components"]:
        cells = rectangle_cells(component)
        overlap = bool(cells & guides)
        adjacent = any(
            (r + dr, c + dc) in guides
            for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if overlap or (contact and adjacent):
            for r, c in cells:
                grid[r][c] = 1
    return grid


def reflection_output(case: dict, endpoint: bool) -> list[list[int]]:
    height, width = case["dimensions"]
    grid = [[0] * width for _ in range(height)]
    for run in case["source_runs"]:
        row = run["row"]
        left, right = run["cols"]
        start = right + 1
        end = 2 * case["anchor"] - left if endpoint else width - 2 - right
        for col in range(max(0, start), min(width - 1, end) + 1):
            grid[row][col] = 1
    return grid


def normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def d4_signatures(cells: set[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    variants = set()
    current = cells
    for _ in range(4):
        variants.add(normalize(current))
        variants.add(normalize({(r, -c) for r, c in current}))
        current = {(c, -r) for r, c in current}
    return variants


def shape_output(case: dict, d4: bool) -> list[list[int]]:
    height, width = case["dimensions"]
    grid = [[7] * width for _ in range(height)]
    prototypes = []
    for proto in case["prototypes"]:
        cells = {tuple(cell) for cell in proto["cells"]}
        signatures = d4_signatures(cells) if d4 else {normalize(cells)}
        prototypes.append((signatures, proto["target_color"]))
    for query in case["queries"]:
        cells = {tuple(cell) for cell in query["cells"]}
        signature = normalize(cells)
        color = 6
        for signatures, target_color in prototypes:
            if signature in signatures:
                color = target_color
                break
        for r, c in cells:
            grid[r][c] = color
    return grid


def artifact(family: str, condition: str, cases: list[dict], rule: str, construction: list[str]) -> dict:
    return {
        "experiment": 32,
        "family": family,
        "condition": condition,
        "coordinate_convention": "zero-based [row,column], inclusive bounds where ranges are used",
        "colors": {
            "incidence": {"background": 0, "guide": 1, "payload": 2},
            "reflection": {"background": 0, "fill": 1},
            "shape": {"background": 7, "source": 6},
        }[family],
        "rule": rule,
        "cases": cases,
        "construction": construction,
        "precomputed_memberships_included": False,
        "literal_output_grids_included": False,
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "incidence_overlap": artifact(
            "incidence",
            "overlap",
            INCIDENCE_CASES,
            "Select a component iff at least one of its cells is also a guide cell.",
            [
                "Initialize the output to background.",
                "Paint every component rectangle payload color.",
                "Paint every guide segment guide color.",
                "Compute the guide cell set and each component cell set.",
                "Repaint every selected component in full with guide color.",
            ],
        ),
        "incidence_contact": artifact(
            "incidence",
            "contact",
            INCIDENCE_CASES,
            "Select a component iff it overlaps a guide or at least one component cell has Manhattan distance exactly one from a guide cell. Diagonal contact does not count.",
            [
                "Initialize the output to background.",
                "Paint every component rectangle payload color.",
                "Paint every guide segment guide color.",
                "Compute overlap and four-neighbor contact from coordinates.",
                "Repaint every selected component in full with guide color.",
            ],
        ),
        "reflection_center_strict": artifact(
            "reflection",
            "center_strict",
            REFLECTION_CASES,
            "For each source run [l,u] in width W, paint fill color from u+1 through W-2-u inclusive. Emit nothing when the interval is empty. Source cells are not retained.",
            [
                "Initialize the output to background.",
                "For every source run, calculate the output interval from the rule.",
                "Clip only to grid bounds and paint that interval fill color.",
            ],
        ),
        "reflection_endpoint_interval": artifact(
            "reflection",
            "endpoint_interval",
            REFLECTION_CASES,
            "For each source run [l,u] with anchor a, paint fill color from u+1 through 2a-l inclusive. Emit nothing when the interval is empty. Source cells are not retained.",
            [
                "Initialize the output to background.",
                "For every source run, calculate the output interval from the rule.",
                "Clip only to grid bounds and paint that interval fill color.",
            ],
        ),
        "shape_translation": artifact(
            "shape",
            "translation",
            SHAPE_CASES,
            "Normalize a cell set only by subtracting its minimum row and column. Match a query to a prototype iff these normalized sets are exactly equal. Recolor matched queries to the prototype target; retain source color for unmatched queries.",
            [
                "Initialize the output to background.",
                "Compute translation-normalized coordinates for each prototype and query.",
                "Apply the first exact prototype match, otherwise retain source color.",
                "Paint every query cell with its resulting color.",
            ],
        ),
        "shape_d4": artifact(
            "shape",
            "d4",
            SHAPE_CASES,
            "A query matches a prototype iff translation-normalized cell sets become equal after any sequence of 90-degree rotations and reflection. Recolor matched queries to the prototype target; retain source color for unmatched queries.",
            [
                "Initialize the output to background.",
                "Generate and translation-normalize all D4 transforms of each prototype.",
                "Normalize each query and test set equality against the transformed prototypes.",
                "Apply the first match, otherwise retain source color, then paint query cells.",
            ],
        ),
    }
    for name, data in artifacts.items():
        (ARTIFACTS / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )

    answer_key = {
        "incidence_overlap": [incidence_output(case, False) for case in INCIDENCE_CASES],
        "incidence_contact": [incidence_output(case, True) for case in INCIDENCE_CASES],
        "reflection_center_strict": [
            reflection_output(case, False) for case in REFLECTION_CASES
        ],
        "reflection_endpoint_interval": [
            reflection_output(case, True) for case in REFLECTION_CASES
        ],
        "shape_translation": [shape_output(case, False) for case in SHAPE_CASES],
        "shape_d4": [shape_output(case, True) for case in SHAPE_CASES],
    }
    (EXP / "private_answer_key.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

