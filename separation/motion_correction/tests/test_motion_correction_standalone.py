"""Standalone tests for the motion correction module."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motion_correction import correct_motion


def _make_synthetic_video(n_frames: int = 50, h: int = 32, w: int = 32) -> xr.DataArray:
    """Create a synthetic preprocessed video DataArray."""
    rng = np.random.RandomState(42)
    data = rng.rand(n_frames, h, w).astype(np.float64)
    return xr.DataArray(
        data,
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(n_frames),
            "height": np.arange(h),
            "width": np.arange(w),
        },
    )


def test_correct_motion_output_types():
    """Verify correct_motion returns correct types and shapes."""
    tmpdir = tempfile.mkdtemp(prefix="minian_test_mc_")
    intpath = os.path.join(tmpdir, "intermediate")
    dpath = os.path.join(tmpdir, "data")
    minian_path = os.path.join(dpath, "minian")
    os.makedirs(minian_path, exist_ok=True)

    try:
        Y_bg = _make_synthetic_video()
        config = {
            "param_estimate_motion": {"dim": "frame"},
            "subset_mc": None,
            "chk": {"frame": 25, "height": 16, "width": 16},
            "intpath": intpath,
            "dpath": dpath,
            "param_save_minian": {
                "dpath": minian_path,
                "meta_dict": dict(session=-1, animal=-2),
                "overwrite": True,
            },
        }

        Y_hw_chk, Y_fm_chk, motion, max_proj, config_out = correct_motion(
            Y_bg, config
        )

        # Type checks
        assert isinstance(Y_hw_chk, xr.DataArray)
        assert isinstance(Y_fm_chk, xr.DataArray)
        assert isinstance(motion, xr.DataArray)
        assert isinstance(max_proj, xr.DataArray)
        assert isinstance(config_out, dict)

        # Shape checks
        assert set(Y_hw_chk.dims) == {"frame", "height", "width"}
        assert set(Y_fm_chk.dims) == {"frame", "height", "width"}
        assert "frame" in motion.dims
        assert set(max_proj.dims) == {"height", "width"}

        print("test_correct_motion_output_types PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_correct_motion_output_types()
