from arc_solver.structural_novelty import (
    ALL_FLAGS,
    analyze_target,
    d4_transform,
    input_orbit_match,
    nonzero_binary_shape_signature,
    pair_orbit_match,
    same_color_component_signature,
    summarize_records,
)


def _pair(input_grid, output_grid):
    return {"input": input_grid, "output": output_grid}


def test_pair_orbit_accepts_rotation_and_shared_injective_color_mapping():
    reference = _pair([[0, 1, 1], [2, 0, 1]], [[2, 1], [0, 2]])
    target = _pair(
        d4_transform([[7 if x == 0 else 4 if x == 1 else 9 for x in row] for row in reference["input"]], "rot90"),
        d4_transform([[7 if x == 0 else 4 if x == 1 else 9 for x in row] for row in reference["output"]], "rot90"),
    )
    witness = pair_orbit_match(reference, target)
    assert witness is not None
    assert witness["transform"] == "rot90"
    assert witness["color_mapping"] == {0: 7, 1: 4, 2: 9}
    assert len(target["input"]) == 3  # rectangular rotation swaps dimensions


def test_pair_orbit_rejects_shared_mapping_conflict_but_input_orbit_accepts():
    reference = _pair([[1, 0], [0, 1]], [[1]])
    target = _pair([[8, 3], [3, 8]], [[6]])
    assert input_orbit_match(reference["input"], target["input"]) is not None
    assert pair_orbit_match(reference, target) is None


def test_orbit_rejects_noninjective_and_repeated_color_mismatch():
    assert input_orbit_match([[1, 2]], [[8, 8]]) is None
    assert input_orbit_match([[1, 1]], [[8, 9]]) is None


def test_zero_is_permutable_in_exact_orbit():
    witness = input_orbit_match([[0, 1], [1, 0]], [[5, 0], [0, 5]])
    assert witness is not None
    assert witness["color_mapping"] == {0: 5, 1: 0}


def test_component_signatures_include_zero_but_binary_shapes_do_not():
    grid = [
        [0, 1, 0, 2],
        [0, 1, 0, 2],
        [3, 0, 0, 0],
    ]
    count, sizes = same_color_component_signature(grid)
    assert count == 5
    assert sizes == (1, 2, 2, 2, 5)
    shapes = nonzero_binary_shape_signature(grid)
    assert len(shapes) == 3
    assert sorted(len(shape) for shape in shapes) == [1, 2, 2]


def test_binary_shape_signature_normalizes_translation_and_d4():
    first = [[1, 0, 0], [1, 1, 0]]
    second = [[0, 7], [7, 7], [0, 0]]
    assert nonzero_binary_shape_signature(first) == nonzero_binary_shape_signature(second)


def _record(key, flag, top2, coverage, accuracy):
    flags = {name: False for name in ALL_FLAGS}
    flags["pair_orbit_novel"] = flag
    return {
        "item_key": key,
        "novelty_flags": flags,
        "performance": {
            "arm": {
                "top2": top2,
                "truth_coverage": coverage,
                "best_cell_accuracy": accuracy,
            }
        },
    }


def test_report_assembly_has_denominators_conditionals_and_safe_correlations():
    summary = summarize_records(
        [
            _record("a", False, True, True, 1.0),
            _record("b", True, False, False, 0.5),
        ],
        ["arm"],
    )
    rate = summary["flag_rates"]["pair_orbit_novel"]
    assert rate == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
        "examples": ["b"],
        "counterexamples": ["a"],
    }
    conditioned = summary["performance_by_flag"]["pair_orbit_novel"]["arm"]
    assert conditioned["false"]["top2_count"] == 1
    assert conditioned["true"]["count"] == 1
    associations = summary["associations"]["pair_orbit_novel"]["arm"]
    assert associations["top2_contingency_phi"]["phi"] == -1.0
    assert associations["best_cell_accuracy_point_biserial_r"] == -1.0
    assert (
        summary["associations"]["input_orbit_novel"]["arm"][
            "top2_contingency_phi"
        ]["phi"]
        is None
    )


def test_analyze_target_separates_input_and_pair_orbits():
    reference = _pair([[1]], [[1]])
    target = _pair([[8]], [[9]])
    analysis = analyze_target(target, [reference])
    assert analysis["novelty_flags"]["input_orbit_novel"] is False
    assert analysis["novelty_flags"]["pair_orbit_novel"] is True
