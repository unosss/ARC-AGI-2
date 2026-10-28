import json

import pytest

from arc_solver.perception import (
    build_probe_prompt,
    connected_components,
    describe_grid,
    parse_json_object,
    render_grid_image,
    score_description,
    validate_grid,
)


GRID = [
    [0, 1, 0, 2],
    [1, 0, 0, 2],
    [0, 0, 3, 0],
]


def test_validate_grid_rejects_ragged_and_invalid_cells():
    with pytest.raises(ValueError):
        validate_grid([[0], [0, 1]])
    with pytest.raises(ValueError):
        validate_grid([[10]])


def test_connectivity_changes_diagonal_component_count():
    four = connected_components(GRID, 4)
    eight = connected_components(GRID, 8)

    assert len(four) == 4
    assert len(eight) == 3
    assert four[0] == {
        "color": 2,
        "size": 2,
        "bbox": [0, 3, 1, 3],
        "shape": [[0, 0], [1, 0]],
    }


def test_describe_grid_has_object_and_relation_ground_truth():
    description = describe_grid(GRID)

    assert description["dimensions"] == [3, 4]
    assert description["color_counts"] == {"0": 7, "1": 2, "2": 2, "3": 1}
    assert description["nonzero_bbox"] == [0, 0, 2, 3]
    assert description["component_count_4"] == 4
    assert description["component_count_8"] == 3
    assert description["symmetry"] == {
        "horizontal": False,
        "vertical": False,
        "rotation_180": False,
    }


def test_probe_prompt_separates_modalities():
    text_prompt = build_probe_prompt(GRID, "text")
    image_prompt = build_probe_prompt(GRID, "image")
    both_prompt = build_probe_prompt(GRID, "both")

    assert "Grid as JSON:" in text_prompt
    assert "image is attached" not in text_prompt
    assert "Grid as JSON:" not in image_prompt
    assert "image is attached" in image_prompt
    assert "Grid as JSON:" in both_prompt
    assert "image is attached" in both_prompt


def test_parse_and_score_exact_description():
    expected = describe_grid(GRID)
    raw = "```json\n" + json.dumps(expected) + "\n```"
    prediction = parse_json_object(raw)
    score = score_description(prediction, expected)

    assert score == {
        "structured": True,
        "reading": 1.0,
        "objects": 1.0,
        "relations": 1.0,
        "overall": 1.0,
        "exact": True,
    }


def test_score_reports_reading_separately():
    expected = describe_grid(GRID)
    prediction = json.loads(json.dumps(expected))
    prediction["sampled_cells"][next(iter(prediction["sampled_cells"]))] = 9
    prediction["component_count_4"] = 99
    score = score_description(prediction, expected)

    assert 0 < score["reading"] < 1
    assert score["objects"] == pytest.approx(2 / 3)
    assert score["relations"] == 1
    assert not score["exact"]


def test_render_grid_image_has_exact_pixel_size():
    pytest.importorskip("PIL")
    image = render_grid_image(GRID, cell_size=10)

    assert image.size == (40, 30)
    assert image.mode == "RGB"
