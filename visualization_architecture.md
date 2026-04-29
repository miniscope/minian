# Visualization module architecture

This document describes the current `minian/visualization.py` monolith, recent dependency choices, and a **practical plan to split it into smaller modules** without breaking notebooks or downstream imports.

## Current state

- **Single file**: `minian/visualization.py` (~2.2k lines) holds:
  - **Interactive viewers** (`VArrayViewer`, `CNMFViewer`, `AlignViewer`) — Panel, HoloViews, Datashader.
  - **Video I/O** (`write_video`, `generate_videos`, helpers).
  - **HoloViews helpers** (`datashade_ndcurve`, `visualize_motion`, …).
  - **Array / CNMF helpers** used by viewers and steps (`norm`, `construct_G`, `centroid`, …).
  - **Pipeline-oriented plots** (`visualize_preprocess`, `visualize_seeds`, `visualize_spatial_update`, …).
- **Backend shim**: `minian/viz_backend.py` centralizes HoloViews/Datashader imports and error messages when the optional `viz` extra is missing.
- **Optional install**: Heavy stack (HoloViews, Panel, Datashader, Matplotlib, …) lives under **`[project.optional-dependencies] viz`** in `pyproject.toml`. Core `minian` must not require it for non-viz code paths.
- **Colormaps**: `AlignViewer` false-colors three sessions as RGB. That path uses **Matplotlib `LinearSegmentedColormap`** (black → channel color → white) so we do **not** need a direct `colorcet` import in our code. (Other libraries in the viz stack may still pull `colorcet` transitively; that is fine.)

Public API today is **imported from the package** (`from minian.visualization import CNMFViewer`, etc.). Any split should **preserve those names** on `minian.visualization` (thin `__init__.py` re-exports).

## Why split

- **Navigation and review**: Easier ownership (viewers vs export vs pipeline plots vs math helpers).
- **Import cost**: Optional lazy imports become possible (e.g. defer `cv2` / `skvideo` until video export).
- **Testing**: Smaller units can be covered with lighter fixtures; keep one integration test that imports the full public surface.
- **Circular imports**: A tree of modules forces explicit dependency direction (helpers → backend; viewers → helpers; no helpers → viewers).

## Proposed layout

Use a **package** `minian/visualization/` with a stable facade, not many loose scripts at repo root.

```text
minian/
  visualization/
    __init__.py          # re-export public API (same symbols as today)
    _colormaps.py        # AlignViewer RGB ramps + any shared cmap tables
    _numeric.py          # norm, normalize, construct_G, convolve_G, centroid, …
    _hv_utils.py         # datashade_ndcurve, small HoloViews-only helpers
    export.py            # write_vid_blk, write_video, concat_video_recursive, generate_videos
    viewers_varray.py    # VArrayViewer (or viewers/varray.py if you prefer a subfolder)
    viewers_cnmf.py      # CNMFViewer
    viewers_align.py     # AlignViewer
    pipeline_plots.py    # visualize_preprocess, visualize_seeds, …, visualize_motion
```

Naming is indicative; adjust to taste (`viewers/` subpackage vs `viewers_*.py` flat files).

### Dependency direction (must not cycle)

1. **`_colormaps.py` / `_numeric.py`** — depend only on NumPy/SciPy/xarray (and Matplotlib for colormaps), not on HoloViews.
2. **`_hv_utils.py`** — depends on `viz_backend` + small numerics if needed.
3. **`export.py`** — may depend on OpenCV, skvideo, ffmpeg, xarray; avoid importing viewers.
4. **`viewers_*.py`** — depend on `viz_backend`, `_hv_utils`, `_numeric`, `_colormaps`, Panel/HoloViews stack.
5. **`pipeline_plots.py`** — depends on `viz_backend`, `_numeric`, `_hv_utils` as needed; keep imports minimal at module top where possible.

### What stays outside

- **`minian/viz_backend.py`** can remain where it is, or move to `minian/visualization/_backend.py` and have a one-line compatibility shim at the old path during migration.

## Migration phases (recommended)

Work in **small PR-sized steps**; run `uv run --extra viz python -c "from minian.visualization import …"` after each step.

| Phase | Move | Risk |
|-------|------|------|
| 0 | Add `docs/visualization_architecture.md` (this file); no code move | None |
| 1 | Extract `_numeric.py` (pure helpers); `visualization.py` imports from it | Low — few external imports of internals |
| 2 | Extract `_colormaps.py` + `_hv_utils.py` | Low |
| 3 | Extract `export.py` | Medium — binary/video deps |
| 4 | Extract one viewer at a time (e.g. `AlignViewer` first — smallest surface) | Medium — Panel streams |
| 5 | Replace `visualization.py` with `visualization/__init__.py` re-exports only | Medium — ensure `git grep` for `minian.visualization` internal imports |

Do **not** delete the old module until `__init__.py` re-exports match the previous public namespace and tests/notebooks pass.

## Backwards compatibility

- **Public imports**: `from minian.visualization import X` must keep working indefinitely (or follow a deprecation cycle with warnings).
- **Private uses**: If anything does `from minian.visualization import _something`, grep and fix or re-export explicitly.

## Testing checklist after a split

- `uv run --extra viz` — import all public symbols from `minian.visualization`.
- Run targeted tests that touch visualization (if any).
- Optional: execute `pipeline.ipynb` / `pipeline_test.ipynb` cells that use `CNMFViewer`, `generate_videos`, etc.

## Related files

- `minian/viz_backend.py` — HoloViews entrypoint and `ImportError` message for missing viz deps.
- `pyproject.toml` — `[project.optional-dependencies] viz`.
