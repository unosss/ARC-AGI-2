import inspect
from pathlib import Path

import arc_solver.hybrid2d as hybrid2d
from arc_solver.hybrid2d.component_programs import (
    MarkerStamp,
    PeriodicRepair,
    TopologyRecolor,
    search_periodic,
)
from arc_solver.hybrid2d.scene_graph import (
    ComponentScene,
    PairedSceneGraph,
    ViewSpec,
    background_hypotheses,
)
from arc_solver.hybrid2d.solver import solve


def test_segmentation_views_expose_background_connectivity_and_topology():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    assert ("zero", 0) in background_hypotheses(grid)
    scene = ComponentScene(grid, ViewSpec(0, 4))
    component = scene.components[0]
    assert component.hole_count == 1
    assert component.topology == "holed"
    assert component.area == 8
    assert component.mask == component.d4_mask
    assert component.border_touch == ()


def test_paired_scene_graph_records_only_same_shape_delta():
    before = [[0, 1], [0, 0]]
    after = [[0, 2], [0, 0]]
    pair = PairedSceneGraph.build(before, after, ViewSpec(0, 4))
    assert pair.changed == frozenset({(0, 1)})
    assert pair.transitions == ((1, 2, 1),)


def test_local_cross_family_is_shared_and_exact():
    before = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    after = [row[:] for row in before]
    for row, column in ((2, 2), (1, 2), (2, 1), (2, 3), (3, 2)):
        after[row][column] = 8
    result = solve([{"input": before, "output": after}], before)
    assert result.candidates == [after]
    assert result.report["candidates"][0]["family"] == "local_motif"


def test_topology_family_outlines_and_recolors_holed_component():
    before = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    program = TopologyRecolor(ViewSpec(0, 4), 1, 2, 8, 6)
    after = [
        [2, 2, 2, 2, 2],
        [2, 8, 8, 8, 2],
        [2, 8, 6, 8, 2],
        [2, 8, 8, 8, 2],
        [2, 2, 2, 2, 2],
    ]
    assert program.execute(before) == after
    result = solve([{"input": before, "output": after}], before)
    assert result.candidates[0] == after
    assert result.report["candidates"][0]["family"] == "component_recolor"


def test_marker_indexed_stamp_uses_frame_relations_and_erases_source():
    before = [
        [0, 4, 0, 0, 3, 3, 4, 3, 3, 3],
        [4, 4, 4, 0, 3, 0, 0, 0, 0, 3],
        [0, 4, 0, 0, 3, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3],
    ]
    after = [row[:] for row in before]
    for row, column in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
        after[row][column] = 0
    for row, column in ((2, 6), (3, 5), (3, 6), (3, 7), (4, 6)):
        after[row][column] = 4
    assert MarkerStamp().execute(before) == after
    result = solve([{"input": before, "output": after}], before)
    assert result.candidates[0] == after


def test_periodic_repair_requires_three_demos_and_survives_loo():
    outputs = [
        [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]],
        [[5, 6, 5, 6, 5, 6], [7, 8, 7, 8, 7, 8]],
        [[2, 3, 2, 3, 2, 3], [4, 5, 4, 5, 4, 5]],
    ]
    inputs = [[row[:] for row in grid] for grid in outputs]
    for index, grid in enumerate(inputs):
        grid[index % 2][index + 2] = 9
    programs, diagnostics = search_periodic(list(zip(inputs, outputs)))
    assert len(programs) == 1
    assert isinstance(programs[0], PeriodicRepair)
    assert (programs[0].axis, programs[0].period) == ("rows", 2)
    assert diagnostics["loo_survival"] is True
    assert diagnostics["ambiguous"] is False


def test_periodic_ambiguity_and_two_demo_overfit_abstain():
    grid = [[1, 1, 1, 1], [1, 1, 1, 1]]
    programs, diagnostics = search_periodic([(grid, grid), (grid, grid), (grid, grid)])
    assert programs == []
    assert diagnostics["ambiguous"] is True
    programs, diagnostics = search_periodic([(grid, grid), (grid, grid)])
    assert programs == []
    assert diagnostics["reason"] == "requires_at_least_three_demonstrations_for_loo"


def test_train_all_gate_rejects_one_demo_only_cross_rule():
    cross = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    recolored = [
        [0, 8, 0],
        [8, 8, 8],
        [0, 8, 0],
    ]
    contradictory = [row[:] for row in cross]
    result = solve(
        [
            {"input": cross, "output": recolored},
            {"input": cross, "output": contradictory},
        ],
        cross,
    )
    assert result.candidates == []


def test_production_has_no_diagnostic_task_dispatch_or_model_dependency():
    package = Path(inspect.getfile(hybrid2d)).parent
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    for task_id in (
        "135a2760",
        "16b78196",
        "1818057f",
        "247ef758",
        "981571dc",
        "aa4ec2a5",
    ):
        assert task_id not in source
    assert "qwen" not in source.lower()
