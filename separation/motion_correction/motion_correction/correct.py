"""Motion correction module entry point.

Wraps MiniAn estimate_motion and apply_transform to replicate
the exact motion correction steps from pipeline.ipynb.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Tuple

import dask as da
import xarray as xr

from minian.motion_correction import apply_transform, estimate_motion
from minian.utilities import (
    custom_arr_optimize,
    custom_delay_optimize,
    save_minian,
)

from .defaults import get_defaults


def _setup_dask_config() -> None:
    """Replicate Dask config from minian/__init__.py."""
    da.config.set(
        array_optimize=custom_arr_optimize,
        delayed_optimize=custom_delay_optimize,
    )
    da.config.set(
        **{
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.85,
            "distributed.worker.memory.pause": 0.9,
            "distributed.worker.memory.terminate": 0.95,
            "distributed.admin.log-length": 100,
            "distributed.scheduler.transition-log-length": 100,
            "optimization.fuse.ave-width": 3,
            "array.slicing.split_large_chunks": False,
        }
    )
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "16384"


def correct_motion(
    Y_bg: xr.DataArray,
    config: dict,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, Dict]:
    """Run motion correction: estimate motion, apply transform, rechunk.

    Args:
        Y_bg: Preprocessed video from preprocessing module.
        config: Configuration dict. Must include 'chk', 'intpath'.
                Optional: param_estimate_motion, subset_mc, param_save_minian.

    Returns:
        Y_hw_chk: Motion-corrected video, spatial-chunked.
        Y_fm_chk: Motion-corrected video, frame-chunked.
        motion: Estimated motion shifts.
        max_proj: Maximum projection of corrected video.
        config_out: Merged config for downstream modules.
    """
    _setup_dask_config()

    defaults = get_defaults()
    cfg = deepcopy(defaults)
    cfg.update(deepcopy(config))

    chk = cfg["chk"]
    intpath = cfg["intpath"]
    subset_mc = cfg.get("subset_mc", None)
    param_save_minian = cfg.get("param_save_minian", {
        "dpath": os.path.join(cfg.get("dpath", "."), "minian"),
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    })

    os.environ["MINIAN_INTERMEDIATE"] = intpath

    # Step 1: Estimate motion
    varr_sel = Y_bg.sel(subset_mc) if subset_mc is not None else Y_bg
    motion = estimate_motion(varr_sel, **cfg["param_estimate_motion"])

    # Step 2: Save motion to final output
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}),
        **param_save_minian,
    )

    # Step 3: Apply transform
    Y = apply_transform(Y_bg, motion, fill=0)

    # Step 4: Save frame-chunked and spatial-chunked versions
    Y_fm_chk = save_minian(
        Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True
    )
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )

    # Step 5: Compute and save max projection
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()

    # Build config_out for downstream
    config_out = deepcopy(cfg)
    config_out["param_save_minian"] = param_save_minian

    return Y_hw_chk, Y_fm_chk, motion, max_proj, config_out
