"""Tests for irregular ``unit_id`` handling in plots and CNMF viewer ``update_AC``."""

from __future__ import annotations

import dask
import holoviews as hv
import numpy as np
import xarray as xr

from minian.visualization.pipeline_plots import (
    _regularize_unit_id_coord_for_image_grid,
    visualize_spatial_update,
)
from minian.visualization.viewers_cnmf import CNMFViewer

hv.extension("bokeh")


def test_regularize_unit_id_maps_irregular_spacing_to_arange() -> None:
    da = xr.DataArray(
        np.zeros((4, 3)),
        dims=("unit_id", "frame"),
        coords={
            "unit_id": np.array([0.0, 1.0, 100.5, 200.9]),
            "frame": [0, 1, 2],
        },
        name="C",
    )
    out = _regularize_unit_id_coord_for_image_grid(da)
    np.testing.assert_array_equal(
        np.asarray(out.coords["unit_id"].values), np.arange(4)
    )


def test_regularize_unit_id_preserves_evenly_spaced_coordinates() -> None:
    u = np.arange(100, 105, dtype=np.float64)
    da = xr.DataArray(
        np.ones((5, 2)),
        dims=("unit_id", "frame"),
        coords={"unit_id": u, "frame": [0, 1]},
        name="C",
    )
    out = _regularize_unit_id_coord_for_image_grid(da)
    np.testing.assert_array_equal(np.asarray(out.coords["unit_id"].values), u)


def test_regularize_unit_id_identity_for_single_cell() -> None:
    da = xr.DataArray(
        np.ones((1, 10)),
        dims=("unit_id", "frame"),
        coords={"unit_id": np.array([999]), "frame": np.arange(10)},
    )
    out = _regularize_unit_id_coord_for_image_grid(da)
    assert int(out.coords["unit_id"].values.item()) == 999


def test_regularize_preserves_arrays_without_unit_id_coords() -> None:
    da = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))
    assert _regularize_unit_id_coord_for_image_grid(da) is da


def test_visualize_spatial_update_runs_without_datashade() -> None:
    rng = np.random.default_rng(7)
    nu, nf, hh, ww = 3, 18, 8, 9
    uids_irreg = np.array([10.0, 250.0, 900.5])
    A = xr.DataArray(
        rng.uniform(0.1, 1.0, size=(nu, hh, ww)).astype(np.float32),
        dims=("unit_id", "height", "width"),
        coords={
            "unit_id": uids_irreg,
            "height": np.arange(hh),
            "width": np.arange(ww),
        },
    ).chunk(dict(unit_id=1))
    C = xr.DataArray(
        rng.uniform(0, 1, size=(nu, nf)).astype(np.float32),
        dims=("unit_id", "frame"),
        coords={"unit_id": uids_irreg, "frame": np.arange(nf)},
    )
    layout = visualize_spatial_update(
        {(2.5,): A},
        {(2.5,): C},
        kdims=["sparse penalty"],
        norm=False,
        datashading=False,
    )
    assert isinstance(layout, hv.Layout)


def test_cnmfviewer_update_ac_use_ac_contracts_movie() -> None:
    nu, nf, hh, ww = 2, 12, 6, 6
    rng = np.random.default_rng(11)
    uids = np.array([41, 55])
    A = xr.DataArray(
        rng.uniform(0.2, 1.0, size=(nu, hh, ww)).astype(np.float32),
        dims=("unit_id", "height", "width"),
        coords={
            "unit_id": uids,
            "height": np.arange(hh),
            "width": np.arange(ww),
        },
    ).chunk(dict(unit_id=1, height=3, width=3))
    C = xr.DataArray(
        rng.random((nu, nf), dtype=np.float32),
        dims=("unit_id", "frame"),
        coords={"unit_id": uids, "frame": np.arange(nf)},
    ).chunk(dict(frame=-1))
    S = xr.DataArray(
        rng.random((nu, nf), dtype=np.float32) * 0.5,
        dims=("unit_id", "frame"),
        coords={"unit_id": uids, "frame": np.arange(nf)},
    ).chunk(dict(frame=-1))
    org = xr.DataArray(
        rng.random((nf, hh, ww), dtype=np.float32),
        dims=("frame", "height", "width"),
        coords={
            "frame": np.arange(nf),
            "height": np.arange(hh),
            "width": np.arange(ww),
        },
    ).chunk(dict(frame=4, height=hh, width=ww))

    with dask.config.set(scheduler="threads"):
        viewer = CNMFViewer(A=A, C=C, S=S, org=org, sortNN=False)
        viewer._useAC = True
        viewer.update_AC([int(uids[0]), int(uids[1])])

    assert viewer._AC.size > 0
    assert "frame" in viewer._AC.dims and "height" in viewer._AC.dims
