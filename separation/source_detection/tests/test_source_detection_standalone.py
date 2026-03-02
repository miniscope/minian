"""Standalone tests for the source detection module."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_synthetic_movie(
    n_frames: int = 200, h: int = 50, w: int = 50, n_cells: int = 3
) -> xr.DataArray:
    """Create a synthetic movie with bright spots for source detection."""
    rng = np.random.RandomState(42)
    data = 0.01 * rng.rand(n_frames, h, w).astype(np.float64)

    # Add synthetic cells
    for i in range(n_cells):
        cy = 10 + i * 15
        cx = 10 + i * 15
        if cy >= h or cx >= w:
            continue
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < h and 0 <= xx < w:
                    r2 = dy**2 + dx**2
                    signal = np.exp(-r2 / 8.0)
                    temporal = 1 + 0.5 * np.sin(
                        np.linspace(0, 4 * np.pi * (i + 1), n_frames)
                    )
                    data[:, yy, xx] += 0.5 * signal * temporal

    return xr.DataArray(
        data,
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(n_frames),
            "height": np.arange(h),
            "width": np.arange(w),
        },
    )


def test_detect_sources_requires_dask_client():
    """Verify detect_sources raises RuntimeError without Dask client."""
    from source_detection import detect_sources

    movie = _make_synthetic_movie(n_frames=10, h=20, w=20)
    max_proj = movie.max("frame").compute()

    with pytest.raises(RuntimeError, match="Dask distributed client"):
        detect_sources(movie, movie, max_proj, {
            "chk": {"frame": 5, "height": 10, "width": 10},
            "intpath": "/tmp/test_sd",
        })


def test_detect_sources_output_types():
    """Verify detect_sources returns correct types (requires Dask)."""
    pytest.importorskip("dask.distributed")
    from dask.distributed import Client, LocalCluster

    from source_detection import detect_sources

    tmpdir = tempfile.mkdtemp(prefix="minian_test_sd_")
    intpath = os.path.join(tmpdir, "intermediate")
    minian_path = os.path.join(tmpdir, "minian")
    os.makedirs(minian_path, exist_ok=True)

    cluster = LocalCluster(
        n_workers=2,
        memory_limit="1GB",
        threads_per_worker=1,
    )
    client = Client(cluster)

    try:
        movie = _make_synthetic_movie()
        chk = {"frame": 100, "height": 25, "width": 25}

        Y_fm_chk = movie.chunk({"frame": chk["frame"], "height": -1, "width": -1})
        Y_hw_chk = movie.chunk({"frame": -1, "height": chk["height"], "width": chk["width"]})
        max_proj = movie.max("frame").compute()

        config = {
            "chk": chk,
            "intpath": intpath,
            "param_save_minian": {
                "dpath": minian_path,
                "meta_dict": dict(session=-1, animal=-2),
                "overwrite": True,
            },
        }

        A, C, S, b, f, b0, c0, config_out = detect_sources(
            Y_hw_chk, Y_fm_chk, max_proj, config
        )

        assert isinstance(A, xr.DataArray)
        assert isinstance(C, xr.DataArray)
        assert isinstance(S, xr.DataArray)
        assert isinstance(b, xr.DataArray)
        assert isinstance(f, xr.DataArray)
        assert isinstance(b0, xr.DataArray)
        assert isinstance(c0, xr.DataArray)
        assert isinstance(config_out, dict)

        assert "unit_id" in A.dims
        assert "unit_id" in C.dims
        assert "frame" in C.dims

        print(f"Detected {A.sizes['unit_id']} units")
        print("test_detect_sources_output_types PASSED")

    finally:
        client.close()
        cluster.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_detect_sources_output_types()
