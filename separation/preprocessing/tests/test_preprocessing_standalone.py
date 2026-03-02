"""Standalone tests for the preprocessing module."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import preprocess_video


def _make_temp_videos(tmpdir: str, n_frames: int = 30, h: int = 32, w: int = 32) -> str:
    """Create a minimal AVI video file for testing."""
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available for creating test videos")

    vpath = os.path.join(tmpdir, "msCam1.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(vpath, fourcc, 30.0, (w, h), isColor=False)
    rng = np.random.RandomState(42)
    for _ in range(n_frames):
        frame = rng.randint(0, 256, (h, w), dtype=np.uint8)
        out.write(frame)
    out.release()
    return tmpdir


def test_preprocess_video_output_types():
    """Verify preprocess_video returns correct types and shapes."""
    tmpdir = tempfile.mkdtemp(prefix="minian_test_pp_")
    intpath = os.path.join(tmpdir, "intermediate")

    try:
        dpath = _make_temp_videos(tmpdir)
        config = {
            "param_load_videos": {
                "pattern": r"msCam[0-9]+\.avi$",
                "dtype": np.uint8,
                "downsample": dict(frame=1, height=1, width=1),
                "downsample_strategy": "subset",
            },
            "param_denoise": {"method": "median", "ksize": 7},
            "param_background_removal": {"method": "tophat", "wnd": 15},
            "intpath": intpath,
            "subset": dict(frame=slice(0, None)),
        }

        Y_bg, chk, config_out = preprocess_video(dpath, config)

        # Type checks
        assert isinstance(Y_bg, xr.DataArray)
        assert isinstance(chk, dict)
        assert isinstance(config_out, dict)

        # Shape checks
        assert "frame" in Y_bg.dims
        assert "height" in Y_bg.dims
        assert "width" in Y_bg.dims

        # Chunk dict
        assert "frame" in chk
        assert "height" in chk
        assert "width" in chk

        # config_out must include chk and dpath
        assert "chk" in config_out
        assert "dpath" in config_out

        print("test_preprocess_video_output_types PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_preprocess_video_output_types()
