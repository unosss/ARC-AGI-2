# Experiment 020: Component scene programs

## Scope

Experiment 020 extends, rather than replaces, the leakage-safe Experiment 019
solver in `arc_solver/hybrid2d/`. It adds deterministic same-shape paired scene
graphs and a compact program registry while retaining the affine-D4 occlusion
reconstructor. It is CPU-only and uses no learned model, GPU, task dispatch,
pinned answer, or test label during search.

The production boundary remains:

```python
solve(train_pairs, test_input)
```

The loader strips accidental test outputs. The CLI fully materializes and
writes predictions before optional evaluator labels are opened. Post-hoc
scoring never invokes the solver.

## Architecture

`scene_graph.py` defines the dual-scene representation. A segmentation view is
part of a program and records:

- background inferred from literal zero, global majority, or border majority;
- 4- or 8-connectivity;
- monochrome or multicolor foreground grouping;
- color, area, bounding box, normalized and D4-normalized masks;
- hole count/topology, border contact, coarse relative position, and repeated
  shape/equivalence class.

`component_programs.py` contains shared typed executors:

- `outline_and_recolor_holed_components`: selects by topology and exterior/
  enclosed-background relations;
- `recolor_solid_cardinal_crosses`: selects a local neighborhood template;
- `copy_external_shape_to_frame_marker_intersections`: derives row/column
  marker intersections and translates one shape by a shared relative anchor;
- `repair_residue_classes_by_unique_majority`: accepts a period only with at
  least three demonstrations, leave-one-out survival, and a unique
  minimum-description-length choice;
- the existing occlusion family supplies affine-D4 completion from globally
  and locally supported witnesses.

Every executor must replay every training output exactly. Programs are ranked
by description length and cross-example support. Equivalent programs and test
outputs are deduplicated, and at most two distinct outputs are retained.
Diagnostics record family rejection reasons, train support, operations,
periodic ambiguity, and leave-one-out status.

Whole-component dispatch and arbitrary per-object coordinate translations are
not in the grammar. This deliberately rejects the earlier `16b78196` overfit.
Constants are limited to colors/periods and typed view parameters that are
shared by all demonstrations.

## Reproduction

Label-free inference:

```bash
python -m arc_solver.hybrid2d \
  kaggle_runs/nvarc_single_pass_l4/output_v007/experiment_015/solver_input/challenges.json \
  --task-ids 0934a4d8 135a2760 136b0064 13e47133 142ca369 \
  16b78196 16de56c4 1818057f 195c6913 1ae2feb7 20270e3b 20a9e565 \
  21897d95 221dfab4 247ef758 269e22fb 271d71e2 36a08778 \
  981571dc aa4ec2a5 \
  --report experiments/020_component_scene_programs/diagnostic20_label_free_report.json \
  --submission experiments/020_component_scene_programs/diagnostic20_submission.json
```

`diagnostic20_report.json` was produced afterward by applying
`evaluate_posthoc` to that already-written submission and evaluator labels.

Tests:

```bash
python -m pytest -q
```

## Measured diagnostic-20 result

Label-free inference completed in 12.03 seconds:

- 20 tasks and 33 test items;
- exact train-fit and prediction for 4/20 tasks;
- 5/33 predicted items and 28/33 abstentions;
- post-hoc top-1 exact: 4/33 items and 3/20 complete tasks;
- post-hoc top-2 exact: 4/33 items and 3/20 complete tasks.

The selected programs and outcomes were:

- `0934a4d8`: affine-D4 occlusion reconstruction; 1/1 exact. This is the
  unchanged Experiment 019 behavior and is a fair geometric prediction.
- `1818057f`: cardinal-cross local motif recolor; 1/1 exact. The template and
  color operation are shared across all three demonstrations; fair/general.
- `aa4ec2a5`: topology-aware outline/hole recolor; 1/1 exact. The selector uses
  holes and exterior adjacency rather than component size; fair/general.
- `247ef758`: marker-indexed relative stamp; 1/2 exact. Both predictions use
  the same inferred frame/marker relation. The second honest failure shows the
  family is incomplete, not task-dispatched.
- `135a2760`: abstained. Two demonstrations are insufficient for the required
  periodic leave-one-out gate, so the known test-wrong periodic interpretation
  is not emitted.
- `981571dc`: abstained because no family replays every training pair exactly.
- `16b78196`: abstained; the forbidden demonstration-specific object mapping
  is not representable.

Experiment 019 achieved 1/20 complete tasks and 1/33 exact items on the same
diagnostic set. Experiment 020 improves this to 3/20 and 4/33 while preserving
the Experiment 019 exact result. The referenced NVARC run achieved 4/20
complete tasks, one more than this deterministic solver. These are local
post-hoc diagnostic measurements, not public-evaluation claims; checkpoint
contamination is irrelevant because Experiment 020 uses no checkpoint.

## Artifacts

- `diagnostic20_submission.json`: predictions written before labels were read.
- `diagnostic20_label_free_report.json`: full search diagnostics without
  evaluator results.
- `diagnostic20_report.json`: the same report plus post-hoc exact-match scores.

The reports preserve rejected families and abstentions. No failed or ambiguous
case is replaced with an oracle-selected answer.
