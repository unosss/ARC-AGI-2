from arc_solver.agentic import (
    build_repair_instruction,
    feedback_payload,
    grid_diagnostics,
)


def test_grid_diagnostics_classifies_cell_errors():
    truth = [[1, 2, 0], [0, 3, 4]]
    prediction = [[1, 0, 5], [0, 6, 4]]

    result = grid_diagnostics(prediction, truth)

    assert result["mismatch_count"] == 3
    assert result["missing"] == 1
    assert result["extra"] == 1
    assert result["wrong_color"] == 1
    assert [item["kind"] for item in result["mismatches"]] == [
        "missing",
        "extra",
        "wrong_color",
    ]


def test_grid_diagnostics_reports_shape_and_runtime_error():
    shape_error = grid_diagnostics([[1]], [[1, 0]])
    runtime_error = grid_diagnostics(None, [[1]], error="execution timed out")

    assert shape_error["predicted_shape"] == [1, 1]
    assert shape_error["expected_shape"] == [1, 2]
    assert shape_error["mismatch_count"] == 1
    assert runtime_error == {"error": "execution timed out"}


def test_feedback_prefers_valid_high_accuracy_candidate():
    evaluations = [
        {
            "source": "invalid",
            "valid": False,
            "error": "forbidden import",
            "train_matches": 0,
            "train_total": 2,
            "cell_accuracy": 0,
        },
        {
            "source": "best",
            "valid": True,
            "error": None,
            "train_matches": 1,
            "train_total": 2,
            "cell_accuracy": 0.9,
            "complexity": 20,
            "train_diagnostics": [{"exact": True}, {"mismatch_count": 2}],
        },
    ]

    payload = feedback_payload(evaluations, limit=1)

    assert payload[0]["program"] == "best"
    assert payload[0]["cell_accuracy"] == 0.9


def test_repair_instruction_can_force_new_parse():
    instruction = build_repair_instruction([], round_index=2, force_reparse=True)

    assert "repair round 2" in instruction
    assert "materially different scene decomposition" in instruction
    assert "[]" in instruction
