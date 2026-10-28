# NVARC single-pass upstream provenance

- Source notebook: <https://www.kaggle.com/code/mayankraaz/arc-agi-2-triple-pass-nvarc-sota-35-9-lb>
- Kaggle notebook slug: `mayankraaz/arc-agi-2-triple-pass-nvarc-sota-35-9-lb`
- Kaggle notebook ID: `139041342`
- Source file: `arc-agi-2-triple-pass-nvarc-sota-35-9-lb.ipynb`

## Extraction

The Python bodies of notebook cells 5–8 were extracted verbatim by removing
only each cell's leading `%%writefile <filename>` notebook magic:

- cell 5: `arc_loader.py`
- cell 6: `arc_decoder.py`
- cell 7: `arc_solver.py`
- cell 8: `starter.py`

No patches were made to these four files. Their SHA-256 digests are:

```text
2c62526daba3fdd9da317e3dcb689acca8e24018e56e4c7f2c33ed15381ed8c6  arc_loader.py
20ec4d4d31f5225cfd54d1ca2e92b73211800cdba413a1bec00c338ce95811a3  arc_decoder.py
2438f66910a1f3ebdfea4e166807c2ce9df534790129f6ad86ceb26088a52131  arc_solver.py
fe4f1b27af75a2c32ead5b61e73135bcd7ee0ba86228e2aa608e9dfd6a243c69  starter.py
```

## Included and excluded components

Included:

- ARC loading and augmentation
- NVARC task-local RSLoRA training
- constrained DFS candidate generation and rescoring
- `score_kgmon` candidate selection
- the upstream process launcher

Excluded:

- `symbolic_loo.py` and its notebook execution cell
- Pass B and Pass C
- multi-pass candidate pooling
- the adaptive controller
- P4 structural-distinctness selection
- checkpointing and final orchestration cells

The local Kaggle runner supplies its own leakage-safe Pass-A orchestration,
telemetry collection, and finalization around these unmodified modules.

## License

The Kaggle notebook page declares the notebook under the Apache License 2.0.
The extracted cells contain no separate copyright header. The notebook's
separate Unsloth dependency contains its own licensing notices and is not
vendored in this directory.

During kernel packaging only, `arc_solver.py` is renamed to
`nvarc_arc_solver.py` and the matching import in `starter.py` is rewritten.
This avoids a module-name collision with this repository's `arc_solver`
package; the four files stored in this directory remain the verbatim
notebook-cell bodies represented by the hashes above.
