"""Regression tests for :func:`~minian.cnmf.compute_AtC`, :func:`~minian.utilities.open_minian`, and safe max-projection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import dask

from minian.cnmf import compute_AtC
from minian.initialization import max_proj_frame
from minian.utilities import open_minian, save_minian


def _chunked_A_C(
    *,
    uids_a: np.ndarray,
    uids_c: np.ndarray,
    C_values: np.ndarray,
    A_values: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    nf, nu_c = C_values.shape
    nu_a, nh, nw = A_values.shape
    assert uids_c.shape == (nu_c,)
    assert uids_a.shape == (nu_a,)
    assert A_values.shape[0] == nu_a
    A = (
        xr.DataArray(
            A_values,
            dims=("unit_id", "height", "width"),
            coords={
                "unit_id": uids_a,
                "height": np.arange(nh),
                "width": np.arange(nw),
            },
        )
        .chunk({"unit_id": -1, "height": -1, "width": -1})
        .astype(np.float64)
    )
    C = (
        xr.DataArray(
            C_values,
            dims=("frame", "unit_id"),
            coords={"frame": np.arange(nf), "unit_id": uids_c},
        )
        .chunk({"frame": 2, "unit_id": 2})
        .astype(np.float64)
    )
    return A, C


def _reference_einsum(
    uids_a: np.ndarray, uids_c: np.ndarray, C: np.ndarray, A: np.ndarray
) -> np.ndarray:
    c_uid_set = set(uids_c.tolist())
    shared = np.array(
        [u for u in uids_a.tolist() if u in c_uid_set], dtype=uids_a.dtype
    )
    if shared.size == 0:
        raise ValueError("empty intersection")
    ix_a = np.array([np.where(uids_a == u)[0][0] for u in shared])
    ix_c = np.array([np.where(uids_c == u)[0][0] for u in shared])
    C_s = C[:, ix_c]
    A_s = A[ix_a, :, :]
    return np.einsum("fu,uhw->fhw", C_s, A_s)


def test_compute_AtC_matches_einsum_chunked() -> None:
    rng = np.random.default_rng(42)
    nh, nw, nu, nf = 2, 2, 3, 7
    uids = np.array([100, 101, 102])
    A_np = rng.random((nu, nh, nw))
    C_np = rng.random((nf, nu))
    A, C = _chunked_A_C(
        uids_a=uids,
        uids_c=uids,
        C_values=C_np,
        A_values=A_np,
    )
    ref = np.einsum("fu,uhw->fhw", C_np, A_np)
    with dask.config.set(scheduler="threads"):
        out = compute_AtC(A, C).values
    assert_allclose(out, ref, rtol=1e-5, atol=1e-8)


def test_compute_AtC_intersects_unit_ids() -> None:
    rng = np.random.default_rng(0)
    uids_a = np.array([10, 11, 12])
    uids_c = np.array([99, 12, 11])  # scrambled; shared {11,12} in A order
    A_np = rng.random((3, 2, 2))
    C_np = rng.random((5, 3))
    A, C = _chunked_A_C(
        uids_a=uids_a,
        uids_c=uids_c,
        C_values=C_np,
        A_values=A_np,
    )
    ref = _reference_einsum(uids_a, uids_c, C_np, A_np)
    with dask.config.set(scheduler="threads"):
        got = compute_AtC(A, C).values
    assert_allclose(got, ref, rtol=1e-5, atol=1e-8)


def test_compute_AtC_raises_when_no_unit_overlap() -> None:
    rng = np.random.default_rng(1)
    A, C = _chunked_A_C(
        uids_a=np.array([1, 2]),
        uids_c=np.array([10, 20]),
        C_values=rng.random((4, 2)),
        A_values=rng.random((2, 2, 2)),
    )
    with pytest.raises(ValueError, match="no overlapping unit_id"):
        with dask.config.set(scheduler="threads"):
            compute_AtC(A, C)


def test_open_minian_outer_merges_disjoint_unit_id_coords(tmp_path) -> None:
    dpath = tmp_path / "minian_root"
    dpath.mkdir()
    root = str(dpath)
    foo = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape(2, 3),
        dims=("unit_id", "frame"),
        coords={"unit_id": np.array([10, 20]), "frame": [0, 1, 2]},
        name="foo",
    )
    save_minian(foo, root, overwrite=True)
    bar = xr.DataArray(
        np.ones((3, 3), dtype=np.float64),
        dims=("unit_id", "frame"),
        coords={"unit_id": np.array([20, 30, 40]), "frame": [0, 1, 2]},
        name="bar",
    )
    save_minian(bar, root, overwrite=True)

    merged = open_minian(root)
    assert isinstance(merged, xr.Dataset)
    ds = merged
    assert ds.sizes["unit_id"] == 4
    assert "foo" in ds.data_vars and "bar" in ds.data_vars
    for u in (30, 40):
        assert np.isnan(ds["foo"].sel(unit_id=int(u)).values).all()
    assert np.isnan(ds["bar"].sel(unit_id=10).values).all()


def test_max_proj_frame_all_nan_pixel_stays_nan_matches_finite_elsewhere() -> None:
    # (0,0) is NaN at every frame → safe max should leave NaN, not -inf/numeric garbage.
    data = np.array(
        [
            [[np.nan, 2.0], [3.0, 4.0]],
            [[np.nan, np.nan], [1.5, 0.5]],
        ],
        dtype=np.float64,
    )
    varr = xr.DataArray(
        data,
        dims=("frame", "height", "width"),
        coords={
            "frame": [0, 1],
            "height": [0, 1],
            "width": [0, 1],
        },
    )
    mp = max_proj_frame(varr, slice(None))
    assert np.isnan(mp.sel(height=0, width=0).values)
    assert float(mp.sel(height=0, width=1).values) == pytest.approx(2.0)
    assert float(mp.sel(height=1, width=0).values) == pytest.approx(3.0)
    assert float(mp.sel(height=1, width=1).values) == pytest.approx(4.0)


def test_max_proj_frame_matches_plain_max_when_no_nans() -> None:
    rng = np.random.default_rng(3)
    data = rng.random((10, 4, 5))
    varr = xr.DataArray(
        data,
        dims=("frame", "height", "width"),
        coords={
            "frame": np.arange(10),
            "height": np.arange(4),
            "width": np.arange(5),
        },
    )
    mp = max_proj_frame(varr, slice(None))
    xr.testing.assert_allclose(mp, varr.max("frame"))


def test_open_minian_merge_no_xarray_join_futurewarning(tmp_path) -> None:
    """Explicit ``join`` on ``xr.merge`` avoids the xarray join-default FutureWarning."""
    import warnings

    dpath = tmp_path / "z"
    dpath.mkdir()
    root = str(dpath)
    a = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]}, name="va")
    save_minian(a, root, overwrite=True)
    b = xr.DataArray([3.0], dims=("x",), coords={"x": [2]}, name="vb")
    save_minian(b, root, overwrite=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = open_minian(root)
    join_future = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "join" in str(w.message).lower()
        and "exact" in str(w.message).lower()
    ]
    assert not join_future
