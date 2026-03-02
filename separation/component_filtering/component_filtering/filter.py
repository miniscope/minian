"""Component filtering module entry point.

Provides explicit post-CNMF quality filtering with per-unit metrics.
For exact parity with the monolithic notebook (which has no explicit
filtering), the default behavior passes all units through with labels=1.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import dask as da
import numpy as np
import xarray as xr

from minian.utilities import custom_arr_optimize, custom_delay_optimize

from .defaults import get_defaults


@dataclass
class FilterResult:
    """Result of component filtering."""

    A: xr.DataArray
    C: xr.DataArray
    S: xr.DataArray
    labels: np.ndarray  # 1=accepted, -1=rejected
    metrics: Dict[str, np.ndarray]
    config_out: Dict


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


def _compute_snr(C: np.ndarray) -> np.ndarray:
    """Compute signal-to-noise ratio per unit from temporal traces."""
    n_units = C.shape[0]
    snr = np.zeros(n_units, dtype=np.float64)
    for i in range(n_units):
        trace = C[i]
        signal = np.max(trace) - np.min(trace)
        noise = np.std(trace)
        snr[i] = signal / noise if noise > 0 else 0.0
    return snr


def _compute_spatial_contiguity(A: np.ndarray) -> np.ndarray:
    """Compute spatial contiguity per unit (fraction of non-zero pixels in bounding box)."""
    n_units = A.shape[0]
    contiguity = np.zeros(n_units, dtype=np.float64)
    for i in range(n_units):
        footprint = A[i]
        nonzero = footprint > 0
        if not np.any(nonzero):
            contiguity[i] = 0.0
            continue
        rows = np.any(nonzero, axis=1)
        cols = np.any(nonzero, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        contiguity[i] = np.sum(nonzero) / bbox_area if bbox_area > 0 else 0.0
    return contiguity


def _compute_temporal_stability(C: np.ndarray) -> np.ndarray:
    """Compute temporal stability per unit (inverse coefficient of variation)."""
    n_units = C.shape[0]
    stability = np.zeros(n_units, dtype=np.float64)
    for i in range(n_units):
        trace = C[i]
        mean_val = np.mean(trace)
        std_val = np.std(trace)
        if std_val > 0 and mean_val != 0:
            stability[i] = abs(mean_val) / std_val
        else:
            stability[i] = 0.0
    return stability


def filter_components(
    A: xr.DataArray,
    C: xr.DataArray,
    S: xr.DataArray,
    b0: xr.DataArray,
    c0: xr.DataArray,
    config: dict,
) -> FilterResult:
    """Filter CNMF components based on quality metrics.

    For exact parity with the monolithic notebook (which has no explicit
    filtering), the default behavior passes all units through with labels=1.
    Quality metrics are computed but only used for filtering when thresholds
    are set in config.

    Args:
        A: Spatial footprints from source detection.
        C: Temporal traces from source detection.
        S: Deconvolved spikes from source detection.
        b0: Baseline from source detection.
        c0: Initial calcium from source detection.
        config: Configuration dict with optional thresholds.

    Returns:
        FilterResult with filtered arrays, labels, and metrics.
    """
    _setup_dask_config()

    defaults = get_defaults()
    cfg = deepcopy(defaults)
    cfg.update(deepcopy(config))

    # Compute arrays for metric calculation
    A_np = A.values if hasattr(A, "values") else np.asarray(A.compute())
    C_np = C.values if hasattr(C, "values") else np.asarray(C.compute())

    n_units = A_np.shape[0] if A_np.ndim == 3 else 0

    # Compute quality metrics
    metrics = {}
    if n_units > 0:
        metrics["snr"] = _compute_snr(C_np)
        metrics["spatial_contiguity"] = _compute_spatial_contiguity(A_np)
        metrics["temporal_stability"] = _compute_temporal_stability(C_np)
    else:
        metrics["snr"] = np.array([], dtype=np.float64)
        metrics["spatial_contiguity"] = np.array([], dtype=np.float64)
        metrics["temporal_stability"] = np.array([], dtype=np.float64)

    # Default: accept all units (for exact parity with notebook)
    labels = np.ones(n_units, dtype=np.int8)

    # Apply thresholds if set
    if cfg.get("snr_threshold") is not None and n_units > 0:
        labels[metrics["snr"] < cfg["snr_threshold"]] = -1
    if cfg.get("spatial_contiguity_threshold") is not None and n_units > 0:
        labels[metrics["spatial_contiguity"] < cfg["spatial_contiguity_threshold"]] = -1
    if cfg.get("temporal_stability_threshold") is not None and n_units > 0:
        labels[metrics["temporal_stability"] < cfg["temporal_stability_threshold"]] = -1

    # Build config_out
    config_out = deepcopy(cfg)

    return FilterResult(
        A=A,
        C=C,
        S=S,
        labels=labels,
        metrics=metrics,
        config_out=config_out,
    )
