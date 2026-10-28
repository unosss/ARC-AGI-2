# Experiment 024 RunPod runner

This is the reproducible RunPod adaptation of the accepted Kaggle production
path. It uses the latest local `arc_solver` and `gpu_arm_worker.py` files and
runs `text_control` followed by `rowcol_component` on GPU 0.

The experiment config is loaded unchanged from
`kaggle_runs/qwen_2d_hard_paired_l4/full-config.json`. In particular,
`merge_for_inference` and `rescore_batch_size` remain absent (false and 1),
each task uses 128 adaptation steps, 16 decode views, the registered DFS
budgets, 8 sequential rescore views, and each arm gets exactly 1,500,000
global loss tokens.

RunPod-only infrastructure adaptations:

- merge the pod's ARC 2026 training files with the exact 16 historical
  challenge payloads and labels before execution;
- expose that dedicated merged directory through `/kaggle/input`, because the
  unchanged production worker discovers its challenge file there;
- run the two arms sequentially with `CUDA_VISIBLE_DEVICES=0`;
- preserve global checkpoints, completed candidate JSON files, logs, and arm
  completion markers so rerunning the command resumes safely;
- expand only the operational emergency deadline from Kaggle's 41,400 seconds
  to seven days by default. No model, data, training, decoding, rescoring, or
  search budget is changed.

The final report is a paired result on a deliberately difficult, enriched
16-task set. It is directional evidence, not an unbiased ARC accuracy estimate.
