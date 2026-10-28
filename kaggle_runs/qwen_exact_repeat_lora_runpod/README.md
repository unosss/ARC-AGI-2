# Experiment 038: exact repeat augmentation

This corrects Experiment 037's invalid capped warp.

- Task: exposed hard task `20270e3b`; every train input and output axis is at
  most 13 cells and therefore remains legal after exact 2× repetition.
- Both arms first train on the unchanged 128-view color/D4 schedule.
- `repeat_control` adds a second identical 128-view cycle.
- `exact_repeat` adds a matched 128-view cycle alternating between exact row
  repetition and exact column repetition.
- No row or column is partially selected: every position on the chosen axis is
  repeated exactly twice.
- Decode and rescore settings are identical and use only the original grids.
