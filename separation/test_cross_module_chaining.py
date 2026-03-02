"""Cross-module chaining test: preprocessing -> motion_correction -> source_detection -> component_filtering.

Validates that the four standalone packages can be chained together
to form the complete MiniAn CNMF pipeline, and that running them
produces the same results as the monolithic inline sequence.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "motion_correction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source_detection"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "component_filtering"))

from component_filtering import FilterResult, filter_components
from motion_correction import correct_motion
from preprocessing import preprocess_video
from source_detection import detect_sources


def _make_temp_videos(
    tmpdir: str, n_frames: int = 100, h: int = 50, w: int = 50, n_cells: int = 3
) -> str:
    """Create minimal AVI video files with synthetic cell signals."""
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available for creating test videos")

    rng = np.random.RandomState(42)
    vpath = os.path.join(tmpdir, "msCam1.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(vpath, fourcc, 30.0, (w, h), isColor=False)

    for t in range(n_frames):
        frame = (10 + 5 * rng.rand(h, w)).astype(np.uint8)
        # Add cell signals
        for i in range(n_cells):
            cy = 10 + i * 15
            cx = 10 + i * 15
            if cy >= h or cx >= w:
                continue
            signal = 50 * (1 + 0.5 * np.sin(2 * np.pi * t / 30 * (i + 1)))
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        r2 = dy**2 + dx**2
                        frame[yy, xx] = min(
                            255, int(frame[yy, xx] + signal * np.exp(-r2 / 4))
                        )
        out.write(frame)
    out.release()
    return tmpdir


def _run_monolithic(dpath, intpath, minian_path, chk_override=None):
    """Run the pipeline inline (monolithic) for comparison."""
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
    from minian.motion_correction import apply_transform, estimate_motion
    from minian.preprocessing import denoise, remove_background
    from minian.utilities import get_optimal_chk, load_videos, save_minian

    os.environ["MINIAN_INTERMEDIATE"] = intpath

    param_load_videos = {
        "pattern": r"msCam[0-9]+\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }
    param_denoise = {"method": "median", "ksize": 7}
    param_background_removal = {"method": "tophat", "wnd": 15}
    param_estimate_motion = {"dim": "frame"}
    param_save_minian = {
        "dpath": minian_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }
    param_seeds_init = {
        "wnd_size": 1000, "method": "rolling", "stp_size": 500,
        "max_wnd": 15, "diff_thres": 3,
    }
    param_pnr_refine = {"noise_freq": 0.06, "thres": 1}
    param_ks_refine = {"sig": 0.05}
    param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06}
    param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
    param_init_merge = {"thres_corr": 0.8}
    param_get_noise = {"noise_range": (0.06, 0.5)}
    param_first_spatial = {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)}
    param_first_temporal = {
        "noise_freq": 0.06, "sparse_penal": 1, "p": 1, "add_lag": 20, "jac_thres": 0.2,
    }
    param_first_merge = {"thres_corr": 0.8}
    param_second_spatial = {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)}
    param_second_temporal = {
        "noise_freq": 0.06, "sparse_penal": 1, "p": 1, "add_lag": 20, "jac_thres": 0.4,
    }

    # Preprocessing
    dpath = os.path.abspath(dpath)
    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath, overwrite=True,
    )
    varr_ref = varr
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    varr_ref = denoise(varr_ref, **param_denoise)
    varr_ref = remove_background(varr_ref, **param_background_removal)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    # Motion correction
    motion = estimate_motion(varr_ref, **param_estimate_motion)
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian,
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()

    # Source detection
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )

    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )

    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    # First spatial update
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True,
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"), intpath, overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # First temporal update
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_first_temporal,
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # First merge
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"), intpath, overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # Second spatial update
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True,
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"), intpath, overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # Second temporal update
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_second_temporal,
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True,
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # Final saves
    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)

    return A, C, S, b, f, b0, c0, motion, max_proj


def _run_separated(dpath, intpath, minian_path):
    """Run the 4-module separated pipeline chain."""
    param_save_minian = {
        "dpath": minian_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    config_pp = {
        "intpath": intpath,
        "subset": dict(frame=slice(0, None)),
    }

    # Module 1: Preprocessing
    Y_bg, chk, cfg_pp = preprocess_video(dpath, config_pp)

    # Module 2: Motion correction
    cfg_mc_in = deepcopy(cfg_pp)
    cfg_mc_in["param_save_minian"] = param_save_minian
    Y_hw_chk, Y_fm_chk, motion, max_proj, cfg_mc = correct_motion(Y_bg, cfg_mc_in)

    # Module 3: Source detection
    cfg_sd_in = deepcopy(cfg_mc)
    A, C, S, b, f, b0, c0, cfg_sd = detect_sources(
        Y_hw_chk, Y_fm_chk, max_proj, cfg_sd_in
    )

    # Module 4: Component filtering
    result = filter_components(A, C, S, b0, c0, cfg_sd)

    return result.A, result.C, result.S, b, f, b0, c0, motion, max_proj


@pytest.fixture
def dask_client():
    """Set up a Dask LocalCluster for testing."""
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=2,
        memory_limit="1GB",
        threads_per_worker=1,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.mark.slow
def test_full_pipeline_chaining(dask_client):
    """Chain all 4 modules and verify monolithic parity."""
    tmpdir = tempfile.mkdtemp(prefix="minian_test_chain_")

    try:
        dpath = _make_temp_videos(tmpdir)

        # Run separated chain
        intpath_sep = os.path.join(tmpdir, "int_sep")
        minian_sep = os.path.join(tmpdir, "minian_sep")
        os.makedirs(minian_sep, exist_ok=True)
        A_sep, C_sep, S_sep, b_sep, f_sep, b0_sep, c0_sep, motion_sep, mp_sep = (
            _run_separated(dpath, intpath_sep, minian_sep)
        )

        # Run monolithic
        intpath_mono = os.path.join(tmpdir, "int_mono")
        minian_mono = os.path.join(tmpdir, "minian_mono")
        os.makedirs(minian_mono, exist_ok=True)
        A_mono, C_mono, S_mono, b_mono, f_mono, b0_mono, c0_mono, motion_mono, mp_mono = (
            _run_monolithic(dpath, intpath_mono, minian_mono)
        )

        # Verify output types
        assert isinstance(A_sep, xr.DataArray)
        assert isinstance(C_sep, xr.DataArray)
        assert isinstance(S_sep, xr.DataArray)

        # Verify exact equality of gate metrics
        assert A_sep.sizes == A_mono.sizes, (
            f"A shape mismatch: {A_sep.sizes} != {A_mono.sizes}"
        )

        np.testing.assert_array_equal(
            motion_sep.values, motion_mono.values,
            err_msg="Motion estimates differ",
        )
        np.testing.assert_array_equal(
            mp_sep.values, mp_mono.values,
            err_msg="Max projections differ",
        )
        np.testing.assert_array_equal(
            A_sep.values, A_mono.values,
            err_msg="Spatial footprints (A) differ",
        )
        np.testing.assert_array_equal(
            C_sep.values, C_mono.values,
            err_msg="Temporal traces (C) differ",
        )
        np.testing.assert_array_equal(
            S_sep.values, S_mono.values,
            err_msg="Spikes (S) differ",
        )

        print(f"Units detected: {A_sep.sizes.get('unit_id', 0)}")
        print(f"A sum: {int(A_sep.sum())}")
        print(f"C sum: {int(C_sep.sum())}")
        print(f"S sum: {int(S_sep.sum())}")
        print("Cross-module chaining test PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.slow
def test_chain_output_shapes(dask_client):
    """Verify that chained modules produce reasonable output shapes."""
    tmpdir = tempfile.mkdtemp(prefix="minian_test_shapes_")

    try:
        dpath = _make_temp_videos(tmpdir)
        intpath = os.path.join(tmpdir, "intermediate")
        minian_path = os.path.join(tmpdir, "minian")
        os.makedirs(minian_path, exist_ok=True)

        A, C, S, b, f, b0, c0, motion, max_proj = _run_separated(
            dpath, intpath, minian_path
        )

        # Basic shape checks
        assert "unit_id" in A.dims
        assert "height" in A.dims
        assert "width" in A.dims
        assert "unit_id" in C.dims
        assert "frame" in C.dims
        assert "frame" in motion.dims
        assert set(max_proj.dims) == {"height", "width"}

        print("Chain output shapes test PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=2, memory_limit="1GB", threads_per_worker=1)
    client = Client(cluster)
    try:
        tmpdir = tempfile.mkdtemp(prefix="minian_test_chain_")
        try:
            dpath = _make_temp_videos(tmpdir)
            intpath = os.path.join(tmpdir, "intermediate")
            minian_path = os.path.join(tmpdir, "minian")
            os.makedirs(minian_path, exist_ok=True)
            A, C, S, b, f, b0, c0, motion, max_proj = _run_separated(
                dpath, intpath, minian_path
            )
            print(f"Units: {A.sizes.get('unit_id', 0)}")
            print("Quick chain test PASSED!")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        client.close()
        cluster.close()
