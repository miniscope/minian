"""Unit tests for the ``minian.simulation`` runtime substrate (migration Step 4).

Covers the empty, correctly-shaped ``Scene`` built by ``Scene.zeros`` /
``Scene.ones`` (shape, dims, coords, fill, dtype, mutability), the RNG handling,
and the ``Cell`` / ``GroundTruthBuilder`` defaults. Step bodies that *populate*
the scene arrive in Step 5; this file only exercises the substrate.
"""

import numpy as np
import xarray as xr

from minian.simulation import (
    Acquisition,
    Cell,
    GroundTruthBuilder,
    ImageSensor,
    Scene,
)
from minian.simulation.scene import MOVIE_DIMS


def _tiny_acquisition(**kw):
    """32×24 sensor, 25 s at 20 fps → a small, fast scene."""
    kw.setdefault("fps", 20.0)
    kw.setdefault("duration_s", 25.0)
    kw.setdefault("image_sensor", ImageSensor(n_px_height=32, n_px_width=24))
    return Acquisition(**kw)


# --- Scene.zeros / Scene.ones shape & content ------------------------------


def test_zeros_shape_dims_and_coords():
    acq = _tiny_acquisition()
    scene = Scene.zeros(acq)
    assert isinstance(scene.movie, xr.DataArray)
    assert scene.movie.dims == MOVIE_DIMS
    assert scene.movie.shape == (acq.n_frames, 32, 24)  # (500, 32, 24)
    np.testing.assert_array_equal(scene.movie["frame"].values, np.arange(acq.n_frames))
    np.testing.assert_array_equal(scene.movie["height"].values, np.arange(32))
    np.testing.assert_array_equal(scene.movie["width"].values, np.arange(24))


def test_zeros_is_all_zero_and_ones_is_all_one():
    acq = _tiny_acquisition()
    assert (Scene.zeros(acq).movie.values == 0.0).all()
    assert (Scene.ones(acq).movie.values == 1.0).all()


def test_working_movie_is_float64():
    # Honest-radiometry accumulation happens in float64; the downcast to
    # Output.store_dtype is a finalize() concern (Step 6), not here.
    assert Scene.zeros(_tiny_acquisition()).movie.dtype == np.float64


def test_movie_is_mutable():
    # Unlike a frozen Spec, a Scene is mutated in place by steps.
    scene = Scene.zeros(_tiny_acquisition())
    scene.movie.loc[{"frame": 0, "height": 1, "width": 2}] = 5.0
    assert scene.movie.isel(frame=0, height=1, width=2) == 5.0


# --- RNG handling ----------------------------------------------------------


def test_rng_defaults_to_a_generator():
    assert isinstance(Scene.zeros(_tiny_acquisition()).rng, np.random.Generator)


def test_provided_rng_is_stored_as_is():
    rng = np.random.default_rng(123)
    assert Scene.ones(_tiny_acquisition(), rng=rng).rng is rng


# --- fresh-scene defaults --------------------------------------------------


def test_fresh_scene_has_empty_collections():
    scene = Scene.zeros(_tiny_acquisition())
    assert scene.cells == []
    assert scene.snapshots == {}
    assert isinstance(scene.truth, GroundTruthBuilder)


def test_ground_truth_builder_starts_all_none():
    gt = GroundTruthBuilder()
    assert (
        gt.shifts
        is gt.vignette
        is gt.leakage
        is gt.bleaching
        is gt.neuropil_temporal
        is gt.neuropil_spatial
        is None
    )


# --- Cell record -----------------------------------------------------------


def test_cell_defaults_are_unpopulated():
    cell = Cell(center_um=(50.0, 10.0, 12.0), snr=4.0)
    assert cell.center_um == (50.0, 10.0, 12.0)
    assert cell.snr == 4.0
    assert (
        cell.footprint_planted
        is cell.footprint_observed
        is cell.trace
        is cell.spikes
        is cell.in_focus
        is cell.optical_brightness
        is cell.detectable
        is None
    )
