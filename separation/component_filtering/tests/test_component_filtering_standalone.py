"""Standalone tests for the component filtering module."""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from component_filtering import FilterResult, filter_components


def _make_synthetic_cnmf_outputs(n_units: int = 5, n_frames: int = 100, h: int = 30, w: int = 30):
    """Create synthetic CNMF output arrays for testing."""
    rng = np.random.RandomState(42)

    A = xr.DataArray(
        np.abs(rng.randn(n_units, h, w)).astype(np.float64),
        dims=["unit_id", "height", "width"],
        coords={"unit_id": np.arange(n_units), "height": np.arange(h), "width": np.arange(w)},
    )
    C = xr.DataArray(
        np.abs(rng.randn(n_units, n_frames)).astype(np.float64),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(n_units), "frame": np.arange(n_frames)},
    )
    S = xr.DataArray(
        np.abs(rng.randn(n_units, n_frames)).astype(np.float64) * 0.1,
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(n_units), "frame": np.arange(n_frames)},
    )
    b0 = xr.DataArray(
        rng.randn(n_units, n_frames).astype(np.float64) * 0.01,
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(n_units), "frame": np.arange(n_frames)},
    )
    c0 = xr.DataArray(
        rng.randn(n_units, n_frames).astype(np.float64) * 0.01,
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(n_units), "frame": np.arange(n_frames)},
    )
    return A, C, S, b0, c0


def test_filter_components_passthrough():
    """Test that default config passes all units through."""
    A, C, S, b0, c0 = _make_synthetic_cnmf_outputs()
    result = filter_components(A, C, S, b0, c0, {})

    assert isinstance(result, FilterResult)
    assert len(result.labels) == 5
    assert np.all(result.labels == 1)
    assert "snr" in result.metrics
    assert "spatial_contiguity" in result.metrics
    assert "temporal_stability" in result.metrics
    assert result.A is A
    assert result.C is C
    assert result.S is S

    print("test_filter_components_passthrough PASSED")


def test_filter_components_with_thresholds():
    """Test filtering with explicit thresholds."""
    A, C, S, b0, c0 = _make_synthetic_cnmf_outputs()
    config = {"snr_threshold": 100.0}  # Very high threshold to reject all
    result = filter_components(A, C, S, b0, c0, config)

    assert isinstance(result, FilterResult)
    assert np.all(result.labels == -1)

    print("test_filter_components_with_thresholds PASSED")


def test_filter_components_empty():
    """Test filtering with zero units."""
    A = xr.DataArray(
        np.empty((0, 10, 10), dtype=np.float64),
        dims=["unit_id", "height", "width"],
    )
    C = xr.DataArray(
        np.empty((0, 50), dtype=np.float64),
        dims=["unit_id", "frame"],
    )
    S = C.copy()
    b0 = C.copy()
    c0 = C.copy()
    result = filter_components(A, C, S, b0, c0, {})

    assert len(result.labels) == 0
    assert len(result.metrics["snr"]) == 0

    print("test_filter_components_empty PASSED")


if __name__ == "__main__":
    test_filter_components_passthrough()
    test_filter_components_with_thresholds()
    test_filter_components_empty()
