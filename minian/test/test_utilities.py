import numpy as np
import xarray as xr

from ..utilities import open_minian, save_minian, update_meta


def _make_dataset(dpath):
    """Save a small minian dataset at `dpath` with no metadata coords."""
    var = xr.DataArray(
        np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
        dims=["frame", "height", "width"],
        coords={"frame": [0, 1], "height": [0, 1, 2], "width": [0, 1, 2, 3]},
        name="test_var",
    )
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


def test_update_meta_only_matches_pattern(tmp_path):
    # A sibling directory that should be ignored by the default pattern.
    _make_dataset(tmp_path / "animalA" / "session1" / "minian")
    _make_dataset(tmp_path / "animalA" / "session1" / "not_minian")

    update_meta(str(tmp_path), meta_dict={"session": -1})

    matched = open_minian(str(tmp_path / "animalA" / "session1" / "minian"))
    skipped = open_minian(str(tmp_path / "animalA" / "session1" / "not_minian"))
    assert str(matched.coords["session"].values) == "session1"
    assert "session" not in skipped.coords
