from pathlib import Path

import dask.array as darr
import numpy as np
import xarray as xr

from ..utilities import open_minian, save_minian, update_meta


def _make_var(name, fill, attrs=None) -> xr.DataArray:
    """Build a small named DataArray with optional attrs."""
    var = xr.DataArray(
        np.full((2, 3, 4), fill, dtype=float),
        dims=["frame", "height", "width"],
        coords={"frame": [0, 1], "height": [0, 1, 2], "width": [0, 1, 2, 3]},
        name=name,
    )
    if attrs:
        var.attrs.update(attrs)
    return var


def _make_dataset(dpath: Path) -> xr.DataArray:
    """Save a single-variable minian dataset at `dpath` with no metadata coords."""
    var = _make_var("test_var", fill=1.0)
    save_minian(var, str(dpath), overwrite=True)
    return var


def test_update_meta_assigns_coords_from_hierarchy(tmp_path):
    # Layout: <tmp>/animalA/session1/minian/test_var.zarr
    mn_path = tmp_path / "animalA" / "session1" / "minian"
    var = _make_dataset(mn_path)

    # Saved without metadata, so the coords are absent.
    before = open_minian(str(mn_path))
    assert "session" not in before.coords
    assert "animal" not in before.coords

    update_meta(str(tmp_path), meta_dict={"session": -1, "animal": -2})

    after = open_minian(str(mn_path))
    assert str(after.coords["session"].values) == "session1"
    assert str(after.coords["animal"].values) == "animalA"
    # The underlying data must be untouched by the metadata update.
    np.testing.assert_array_equal(after["test_var"].values, var.values)


def test_update_meta_is_lazy(tmp_path):
    """
    update_meta does not overwrite or copy existing arrays.
    similar to, but separable from testing for value equality.
    """
    mn_path = tmp_path / "animalA" / "session1" / "minian"
    meta_dict = {"session": -1, "animal": -2}
    _make_dataset(mn_path)

    non_updated_paths = [
        p for p in mn_path.rglob("*") if not p.is_dir() and p.parent.name not in meta_dict
    ]
    assert len(non_updated_paths) > 0

    before = {p: p.stat().st_mtime_ns for p in non_updated_paths}
    update_meta(str(mn_path), meta_dict=meta_dict)
    after = {p: p.stat().st_mtime_ns for p in non_updated_paths}

    assert all(a == b for a, b in zip(before.values(), after.values()))


def test_update_meta_only_matches_pattern(tmp_path):
    # A sibling directory that should be ignored by the default pattern.
    _make_dataset(tmp_path / "animalA" / "session1" / "minian")
    _make_dataset(tmp_path / "animalA" / "session1" / "not_minian")

    update_meta(str(tmp_path), meta_dict={"session": -1})

    matched = open_minian(str(tmp_path / "animalA" / "session1" / "minian"))
    skipped = open_minian(str(tmp_path / "animalA" / "session1" / "not_minian"))
    assert str(matched.coords["session"].values) == "session1"
    assert "session" not in skipped.coords


def test_update_meta_matches_save_minian_on_multivar_dataset(tmp_path):
    # A realistic minian dataset bundles several variables in one `minian` dir.
    meta_dict = {"session": -1, "animal": -2}
    variables = [
        _make_var("A", fill=1.0, attrs={"unit": "au", "doc": "footprints"}),
        _make_var("C", fill=2.0),
        _make_var("S", fill=3.0),
    ]

    # Path 1: save without metadata, then stamp it in with update_meta.
    updated_root = tmp_path / "updated"
    updated_mn = updated_root / "animalX" / "session9" / "minian"
    for var in variables:
        save_minian(var, str(updated_mn), overwrite=True)
    update_meta(str(updated_root), meta_dict=meta_dict)

    # Path 2: save the same data with meta_dict up front.
    direct_root = tmp_path / "direct"
    direct_mn = direct_root / "animalX" / "session9" / "minian"
    for var in variables:
        save_minian(var, str(direct_mn), meta_dict=meta_dict, overwrite=True)

    updated = open_minian(str(updated_mn))
    direct = open_minian(str(direct_mn))

    # (b) every variable survives and the dir reopens / merges cleanly.
    assert set(updated.data_vars) == {"A", "C", "S"}
    assert str(updated.coords["session"].values) == "session9"
    assert str(updated.coords["animal"].values) == "animalX"

    # (c) updating after the fact is equivalent to saving with meta_dict,
    # including data, coords, and the preserved variable attrs.
    xr.testing.assert_identical(updated, direct)
    assert updated["A"].attrs["unit"] == "au"
    assert updated["A"].attrs["doc"] == "footprints"


def test_save_minian_discards_stale_chunk_encoding(tmp_path):
    """A stale ``encoding['chunks']`` must not be honored when saving.

    xarray stamps ``encoding['chunks']`` when it reads a zarr store and leaves it
    in place when the array is later rechunked in memory. Saving such an array
    then raises "would overlap multiple Dask chunks" on xarray >=2024. save_minian
    drops the tag, so the round-trip is byte-identical and the on-disk grid
    follows the live layout.
    """
    mn = tmp_path / "minian"
    data = np.arange(3 * 8, dtype="float64").reshape(3, 8)
    var = xr.DataArray(
        darr.from_array(data, chunks=(3, 2)),  # live frame chunks: 2, 2, 2, 2
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(3), "frame": np.arange(8)},
        name="C",
    )
    # Stale tag from an earlier save: a 4-frame grid no longer tiling (3, 2).
    var.encoding["chunks"] = (3, 4)

    saved = save_minian(var, str(mn), overwrite=True)
    np.testing.assert_array_equal(saved.compute().values, data)

    roundtrip = open_minian(str(mn))["C"]
    np.testing.assert_array_equal(roundtrip.values, data)
    # On-disk grid follows the live (3, 2) layout, not the stale (3, 4).
    assert roundtrip.encoding.get("chunks") == (3, 2)
