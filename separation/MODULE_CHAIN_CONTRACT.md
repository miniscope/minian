# Module Chain Contract (4-Module Separation)

This repository validates a 4-stage separated chain for MiniAn CNMF calcium imaging:

1. `preprocessing`: raw video directory -> background-removed video + chunk sizes
2. `motion_correction`: preprocessed video -> motion-corrected video in two chunk layouts
3. `source_detection`: motion-corrected video -> CNMF spatial/temporal components (A, C, S, b, f, b0, c0)
4. `component_filtering`: CNMF outputs -> filtered components with quality metrics

## Module Signatures

### Module 1: preprocessing

```python
def preprocess_video(dpath: str, config: dict) -> Tuple[xr.DataArray, dict, dict]:
    """
    Args:
        dpath: Path to directory containing raw video files (AVI).
        config: Dict with keys: param_load_videos, param_denoise,
                param_background_removal, intpath, subset.

    Returns:
        Y_bg: Background-removed, denoised video (xr.DataArray, dims: frame/height/width).
        chk: Chunk size dict with keys 'frame', 'height', 'width'.
        config_out: Merged config including chk, intpath, dpath for downstream modules.
    """
```

### Module 2: motion_correction

```python
def correct_motion(
    Y_bg: xr.DataArray, config: dict
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, dict]:
    """
    Args:
        Y_bg: Preprocessed video from Module 1.
        config: Dict with keys: param_estimate_motion, subset_mc, chk, intpath,
                param_save_minian.

    Returns:
        Y_hw_chk: Motion-corrected video, spatial-chunked (frame=-1, height=chk, width=chk).
        Y_fm_chk: Motion-corrected video, frame-chunked (frame=chk, height=-1, width=-1).
        motion: Estimated motion shifts (xr.DataArray, dims: frame/shift_dim).
        max_proj: Maximum projection of corrected video (xr.DataArray, dims: height/width).
        config_out: Merged config for downstream.
    """
```

### Module 3: source_detection

```python
def detect_sources(
    Y_hw_chk: xr.DataArray,
    Y_fm_chk: xr.DataArray,
    max_proj: xr.DataArray,
    config: dict,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
           xr.DataArray, xr.DataArray, xr.DataArray, dict]:
    """
    Args:
        Y_hw_chk: Spatial-chunked video from Module 2.
        Y_fm_chk: Frame-chunked video from Module 2.
        max_proj: Maximum projection from Module 2.
        config: Dict with all CNMF parameters (param_seeds_init, param_pnr_refine,
                param_ks_refine, param_seeds_merge, param_initialize, param_init_merge,
                param_get_noise, param_first_spatial, param_first_temporal,
                param_first_merge, param_second_spatial, param_second_temporal,
                chk, intpath, param_save_minian).

    Returns:
        A: Spatial footprints (unit_id x height x width).
        C: Temporal traces (unit_id x frame).
        S: Deconvolved spikes (unit_id x frame).
        b: Background spatial component (height x width).
        f: Background temporal component (frame).
        b0: Baseline (unit_id x frame).
        c0: Initial calcium (unit_id x frame).
        config_out: Merged config for downstream.

    Raises:
        RuntimeError: If no Dask distributed client is active.
    """
```

### Module 4: component_filtering

```python
def filter_components(
    A: xr.DataArray,
    C: xr.DataArray,
    S: xr.DataArray,
    b0: xr.DataArray,
    c0: xr.DataArray,
    config: dict,
) -> FilterResult:
    """
    Args:
        A, C, S, b0, c0: CNMF outputs from Module 3.
        config: Dict with optional param_final_merge, quality thresholds.

    Returns:
        FilterResult dataclass with fields:
            A, C, S: Filtered arrays.
            labels: Per-unit accept (1) / reject (-1) labels.
            metrics: Per-unit quality metrics dict.
            config_out: Final config.
    """
```

## Required Metadata Handoff

`config_out` from each module is merged into the next module's input config.
Critical fields that must be threaded through:

- `chk` (chunk sizes) — set by preprocessing, used by all downstream modules
- `intpath` — intermediate zarr storage path, used by all modules
- `param_save_minian` — final output save parameters
- `dpath` — absolute path to data directory

## Enforced Behavior

- **Dask client**: Caller-managed (not module-managed). Each module documents the
  requirement. `source_detection` raises `RuntimeError` if no active Dask client.
- **Dask config**: Each module entry point sets the MiniAn Dask optimizations
  (`custom_arr_optimize`, `custom_delay_optimize`, memory settings) to ensure
  deterministic graph execution regardless of import order.
- **MINIAN_INTERMEDIATE**: Each module manages `os.environ["MINIAN_INTERMEDIATE"]`
  internally. The `intpath` is passed via config.
- **Intermediate saves**: Each module uses `save_minian()` at the exact same points
  as the monolithic pipeline to ensure Dask graph materialization matches.

## Example Chaining Pattern

```python
from preprocessing import preprocess_video
from motion_correction import correct_motion
from source_detection import detect_sources
from component_filtering import filter_components

# Caller manages Dask client
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=4, memory_limit="2GB",
                       resources={"MEM": 1}, threads_per_worker=2)
client = Client(cluster)

Y_bg, chk, cfg_pp = preprocess_video(dpath, config)
Y_hw, Y_fm, motion, max_proj, cfg_mc = correct_motion(Y_bg, cfg_pp)
A, C, S, b, f, b0, c0, cfg_sd = detect_sources(Y_hw, Y_fm, max_proj, cfg_mc)
result = filter_components(A, C, S, b0, c0, cfg_sd)

client.close()
cluster.close()
```

## Ground Truth Assertions (from test_pipeline.py)

```python
assert sizes["frame"] == 2000
assert sizes["height"] == 480
assert sizes["width"] == 752
assert sizes["unit_id"] == 282
assert motion.sum("frame") == [423, -239]
assert int(max_proj.sum()) == 1501505
assert int(C.sum()) == 478444
assert int(S.sum()) == 3943
assert int(A.sum()) == 41755
```
