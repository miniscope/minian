# MiniAn pipeline notebook

`pipeline.ipynb` is the main MiniAn analysis pipeline: load videos →
pre-processing → motion correction → seed initialization → CNMF (spatial /
temporal updates) → visualization.

## Running it

```bash
pip install minian
minian-notebooks copy pipeline      # copies this folder to ./minian-notebooks/pipeline/
cd minian-notebooks/pipeline
jupyter notebook pipeline.ipynb
```

The notebook fetches its demo movie automatically via `minian.data.fetch`
(cached + checksum-verified, no manual download step). The demo is a mouse CA1
recording acquired on a Miniscope V3, 5x temporally downsampled (10 `msCam`
`.avi` files, 2000 frames at 480x752). To run it on your own data instead,
edit the `dpath` cell near the top.

## Contents

- `pipeline.ipynb` — the notebook.
- `assets/` — figures referenced inline (`assets/workflow.png`, the
  `assets/param_*.png` parameter-tuning examples). These travel with the
  notebook so the `copy` keeps every image working.
