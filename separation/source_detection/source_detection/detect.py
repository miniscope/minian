"""Source detection module entry point.

Wraps the full MiniAn CNMF pipeline: seed initialization, component
initialization, noise estimation, two rounds of spatial/temporal updates
with merging — replicating the exact sequence from pipeline.ipynb.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Tuple

import dask as da
import xarray as xr

from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import (
    initA,
    initC,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
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


def _check_dask_client() -> None:
    """Raise RuntimeError if no active Dask distributed client."""
    try:
        from dask.distributed import get_client
        get_client()
    except (ImportError, ValueError):
        raise RuntimeError(
            "source_detection requires an active Dask distributed client. "
            "Create one with: client = Client(LocalCluster(...))"
        )


def detect_sources(
    Y_hw_chk: xr.DataArray,
    Y_fm_chk: xr.DataArray,
    max_proj: xr.DataArray,
    config: dict,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
           xr.DataArray, xr.DataArray, xr.DataArray, Dict]:
    """Run full CNMF source detection pipeline.

    Args:
        Y_hw_chk: Spatial-chunked motion-corrected video.
        Y_fm_chk: Frame-chunked motion-corrected video.
        max_proj: Maximum projection of corrected video.
        config: Configuration dict with all CNMF parameters, chk, intpath,
                param_save_minian.

    Returns:
        A: Spatial footprints.
        C: Temporal traces.
        S: Deconvolved spikes.
        b: Background spatial component.
        f: Background temporal component.
        b0: Baseline.
        c0: Initial calcium.
        config_out: Merged config for downstream.

    Raises:
        RuntimeError: If no active Dask distributed client.
    """
    _setup_dask_config()
    _check_dask_client()

    defaults = get_defaults()
    cfg = deepcopy(defaults)
    cfg.update(deepcopy(config))

    chk = cfg["chk"]
    intpath = cfg["intpath"]
    param_save_minian = cfg.get("param_save_minian", {
        "dpath": os.path.join(cfg.get("dpath", "."), "minian"),
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    })

    os.environ["MINIAN_INTERMEDIATE"] = intpath

    # ================================================================
    # SEED INITIALIZATION
    # ================================================================

    # Step 1: Initial seed detection
    seeds = seeds_init(Y_fm_chk, **cfg["param_seeds_init"])

    # Step 2: PNR refinement
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **cfg["param_pnr_refine"])

    # Step 3: KS refinement
    seeds = ks_refine(Y_hw_chk, seeds, **cfg["param_ks_refine"])

    # Step 4: Filter and merge seeds
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **cfg["param_seeds_merge"])

    # ================================================================
    # COMPONENT INITIALIZATION
    # ================================================================

    # Step 5: Initialize spatial components
    A_init = initA(
        Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **cfg["param_initialize"]
    )
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    # Step 6: Initialize temporal components
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    # Step 7: Initial merge
    A, C = unit_merge(A_init, C_init, **cfg["param_init_merge"])
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )

    # ================================================================
    # INITIAL BACKGROUND
    # ================================================================

    # Step 8: Initial background estimation
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    # ================================================================
    # NOISE ESTIMATION
    # ================================================================

    # Step 9: Noise estimation
    sn_spatial = get_noise_fft(Y_hw_chk, **cfg["param_get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    # ================================================================
    # FIRST SPATIAL UPDATE
    # ================================================================

    # Step 10: First spatial update
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **cfg["param_first_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )

    # Step 11: First background update
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    # Step 12: Save first spatial/background update results
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # ================================================================
    # FIRST TEMPORAL UPDATE
    # ================================================================

    # Step 13: Compute trace for temporal update
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    # Step 14: First temporal update
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **cfg["param_first_temporal"]
    )

    # Step 15: Save first temporal update results
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # ================================================================
    # FIRST MERGE
    # ================================================================

    # Step 16: First merge
    A_mrg, C_mrg, [sig_mrg] = unit_merge(
        A, C, [C + b0 + c0], **cfg["param_first_merge"]
    )

    # Step 17: Save merge results
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # ================================================================
    # SECOND SPATIAL UPDATE
    # ================================================================

    # Step 18: Second spatial update
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **cfg["param_second_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )

    # Step 19: Second background update
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    # Step 20: Save second spatial/background update results
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # ================================================================
    # SECOND TEMPORAL UPDATE
    # ================================================================

    # Step 21: Compute trace for second temporal update
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    # Step 22: Second temporal update
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **cfg["param_second_temporal"]
    )

    # Step 23: Save final temporal update results
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # ================================================================
    # SAVE FINAL OUTPUTS
    # ================================================================

    # Step 24: Save final results to minian output directory
    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)

    # Build config_out
    config_out = deepcopy(cfg)
    config_out["param_save_minian"] = param_save_minian

    return A, C, S, b, f, b0, c0, config_out
