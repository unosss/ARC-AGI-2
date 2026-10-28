import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "042_structured_rule_sft"
sys.path.insert(0, str(EXP))

import build_corpus
import grade
import model_io
import program_sandbox


CORRECT_CODE = """
import numpy as np

def generate_puzzle_output(input_grid):
    return np.flipud(input_grid)
"""
WRONG_CODE = """
import numpy as np

def generate_puzzle_output(input_grid):
    return np.array(input_grid)
"""


def flipped(grid):
    return list(reversed(grid))


class StructuredRuleSftTest(unittest.TestCase):
    def test_canonical_program_replays_grid(self):
        program = build_corpus.canonical_program(CORRECT_CODE)
        result = program_sandbox.run_program(program, [[[1, 2], [3, 4]]])
        self.assertEqual(result, [{"output": [[3, 4], [1, 2]]}])

    def test_sandbox_rejects_file_access(self):
        code = "def transform(grid):\n    return open('/etc/passwd').read()\n"
        with self.assertRaises(program_sandbox.ProgramRejected):
            program_sandbox.validate_program(code)

    def test_parsers(self):
        self.assertEqual(model_io.parse_grid("answer [[1,2],[3,4]] end"), [[1, 2], [3, 4]])
        self.assertEqual(
            model_io.parse_program("<program>\ndef transform(grid):\n return grid\n</program>"),
            "def transform(grid):\n return grid\n",
        )

    def test_matching_program_uses_recorded_outputs(self):
        grids = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        pairs = [{"input": grid, "output": flipped(grid)} for grid in grids]
        program, sid = build_corpus.select_matching_program(
            pairs,
            [(0, CORRECT_CODE), (1, WRONG_CODE)],
            timeout_seconds=3.0,
            memory_bytes=1 << 30,
        )
        self.assertEqual(sid, 0)
        self.assertIn("def transform", program)

    def test_builder_seals_test_labels_and_splits_by_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_root = root / "pairs"
            destination = root / "corpus"
            pair_root.mkdir()
            index_rows = []
            thresholds = [5000, 7500]
            sources = {"train": [], "validation": [], "test": []}
            candidate = 0
            while any(len(values) < (2 if split == "train" else 1) for split, values in sources.items()):
                source_id = f"{candidate:08x}"
                split = build_corpus.source_split(source_id, 42, thresholds)
                needed = 2 if split == "train" else 1
                if len(sources[split]) < needed:
                    sources[split].append(source_id)
                candidate += 1
            source_pairs = []
            for split, count in (("train", 2), ("validation", 1), ("test", 1)):
                first_ids = sources[split][:count]
                for first in first_ids:
                    second_index = candidate
                    while True:
                        second = f"{second_index:08x}"
                        second_index += 1
                        if build_corpus.source_split(second, 42, thresholds) == split:
                            candidate = second_index
                            source_pairs.append((first, second))
                            break
            for puzzle_index, (first, second) in enumerate(source_pairs):
                grids = [
                    [[puzzle_index, value], [value + 1, puzzle_index]]
                    for value in range(1, 6)
                ]
                pairs = [{"input": grid, "output": flipped(grid)} for grid in grids]
                relative = Path("nvarc_full") / first / f"{first}_{second}.json"
                (pair_root / relative).parent.mkdir(parents=True, exist_ok=True)
                (pair_root / relative).write_text(json.dumps(pairs))
                index_rows.extend(
                    [
                        {
                            "puzzle_name1": first,
                            "puzzle_name2": second,
                            "sid": 0,
                            "completion": f"TAG = {puzzle_index}\n" + CORRECT_CODE,
                        },
                        {
                            "puzzle_name1": first,
                            "puzzle_name2": second,
                            "sid": 1,
                            "completion": WRONG_CODE,
                        },
                    ]
                )
            program_index = root / "programs.jsonl"
            program_index.write_text(
                "".join(json.dumps(row) + "\n" for row in index_rows)
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "split_seed": 42,
                        "demo_count": 3,
                        "validation_rules": 1,
                        "test_rules": 1,
                        "train_budgets": [2],
                        "rule_reserve": 3,
                        "source_split_thresholds": thresholds,
                        "program_timeout_seconds": 3.0,
                        "program_memory_bytes": 1073741824,
                    }
                )
            )
            manifest = build_corpus.build(
                argparse.Namespace(
                    pairs_root=pair_root,
                    program_index=program_index,
                    program_parquet_root=None,
                    output_dir=destination,
                    labels_dir=root / "sealed",
                    config=config,
                )
            )
            self.assertEqual(manifest["usable_rule_count"], 4)
            challenge = json.loads(
                (destination / "test_challenges.jsonl").read_text().strip()
            )
            label = json.loads((root / "sealed" / "test_labels.jsonl").read_text().strip())
            self.assertFalse((destination / "test_labels.jsonl").exists())
            self.assertNotIn("query_output", challenge)
            self.assertNotIn("program", challenge)
            self.assertEqual(challenge["id"], label["id"])

    def test_paired_p_value(self):
        self.assertEqual(grade.paired_p_value(0, 0), 1.0)
        self.assertAlmostEqual(grade.paired_p_value(5, 0), 0.0625)

    def test_duplicate_candidate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            grade.unique_by_id([{"id": "x"}, {"id": "x"}], "test")


if __name__ == "__main__":
    unittest.main()
