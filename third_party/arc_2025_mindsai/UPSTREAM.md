# MindsAI ARC 2025 upstream

- Repository: https://github.com/jcole75/arc_2025_mindsai
- Commit: `d4ec950b11e31a9114efe82c86c389808224edc9`
- License: MIT; see `LICENSE`.
- Scope: Python runtime and configuration files required for TTT/AIRV.
- Excluded: notebooks, reports, bundled ARC datasets, TPU-only scripts, and git metadata.
- Local changes:
  - `src/modules/config/model_configs/phase1_codet5.py` is an Experiment 013
    integration profile.
  - `training_worker.py`, `inference_worker.py`, and `dataset_generator.py`
    make the optional Flax conversion import compatible with Transformers 5
    and load the published BPE tokenizer through the generic fast-tokenizer
    backend. Transformers 5 otherwise selects the incompatible T5 Unigram
    constructor from the legacy tokenizer class metadata.
