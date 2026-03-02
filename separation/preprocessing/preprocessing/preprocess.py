"""Preprocessing module entry point.

Wraps MiniAn load_videos, denoise, and remove_background to replicate
the exact preprocessing steps from pipeline.ipynb.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Tuple

import dask as da
import xarray as xr

from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    custom_arr_optimize,
    custom_delay_optimize,
    get_optimal_chk,
    load_videos,
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


def preprocess_video(
    dpath: str,
    config: dict,
) -> Tuple[xr.DataArray, Dict, Dict]:
    """Run preprocessing: load, glow removal, denoise, background removal.

    Args:
        dpath: Path to directory containing raw video files.
        config: Configuration dict. Missing keys are filled from defaults.

    Returns:
        Y_bg: Background-removed video (xr.DataArray).
        chk: Chunk size dict with keys 'frame', 'height', 'width'.
        config_out: Merged config including chk, intpath, dpath.
    """
    _setup_dask_config()

    defaults = get_defaults()
    cfg = deepcopy(defaults)
    cfg.update(deepcopy(config))

    intpath = cfg["intpath"]
    subset = cfg.get("subset", dict(frame=slice(0, None)))

    os.environ["MINIAN_INTERMEDIATE"] = intpath

    # Step 1: Load videos
    dpath = os.path.abspath(dpath)
    varr = load_videos(dpath, **cfg["param_load_videos"])

    # Step 2: Get optimal chunk sizes
    chk, _ = get_optimal_chk(varr, dtype=float)

    # Step 3: Chunk and save raw video to intermediate
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )

    # Step 4: Apply subset selection
    varr_ref = varr.sel(subset)

    # Step 5: Glow removal (subtract per-pixel minimum)
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min

    # Step 6: Denoise
    varr_ref = denoise(varr_ref, **cfg["param_denoise"])

    # Step 7: Background removal
    varr_ref = remove_background(varr_ref, **cfg["param_background_removal"])

    # Step 8: Save preprocessed video to intermediate
    varr_ref = save_minian(
        varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True
    )

    # Build config_out for downstream modules
    config_out = deepcopy(cfg)
    config_out["chk"] = chk
    config_out["dpath"] = dpath

    return varr_ref, chk, config_out
