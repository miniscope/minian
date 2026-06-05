"""Unit tests for the simulate() orchestrator (migration Step 6b).

Covers the compose loop end-to-end: a recording from a full ``Spec``, seed
determinism, automatic motion-margin sizing (so motion specs need no hand-sized
canvas), ``until=`` partial builds, and the per-stage snapshot keys.
"""

import numpy as np
import pytest

from minian.simulation import (
    Acquisition,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Render,
    Sensor,
    Spec,
    Vignette,
    simulate,
)


def _acq(n_px=64, fps=20.0, duration_s=1.0, bit_depth=8):
    """64×64 sensor at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    return Acquisition(
        fps=fps,
        duration_s=duration_s,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth
        ),
    )


def _minimal_spec(**output_kw):
    acq = _acq()
    return Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceNeurons(density_per_mm2=2500.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Sensor(photons_per_unit=100.0),
        ],
        output=Output(**output_kw),
    )


def _full_spec(**output_kw):
    acq = _acq()
    return Spec(
        acquisition=acq,
        seed=11,
        steps=[
            PlaceNeurons(density_per_mm2=2500.0, soma_radius_um=4.0, depth_range_um=(0.0, 100.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Neuropil(n_components=2),
            Bleaching(final_fraction=0.7),
            BrainMotion(walk_step_um=0.3, max_shift_um=2.0),  # ≤ 5% of the 64 µm FOV
            Vignette(falloff=0.6),
            Leakage(profile="gaussian", level=0.1),
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(**output_kw),
    )


def test_simulate_runs_minimal_spec_end_to_end():
    rec = simulate(_minimal_spec())
    acq = rec.spec.acquisition
    assert rec.observed.shape == (acq.n_frames, 64, 64)
    assert rec.observed.dtype == np.float32
    np.testing.assert_array_equal(rec.observed, np.round(rec.observed))  # counts
    assert 0.0 <= rec.observed.min() and rec.observed.max() <= 255.0
    assert rec.ground_truth.n_units > 0


def test_simulate_is_deterministic_given_seed():
    a = simulate(_minimal_spec())
    b = simulate(_minimal_spec())
    np.testing.assert_array_equal(a.observed, b.observed)
    assert a.ground_truth.n_units == b.ground_truth.n_units
    np.testing.assert_array_equal(a.ground_truth.A_observed, b.ground_truth.A_observed)


def test_simulate_auto_sizes_motion_margin():
    # A motion spec runs with no hand-sized canvas: simulate() pads it, and the
    # observed movie comes back cropped to the sensor FOV with shifts recorded.
    rec = simulate(_full_spec())
    acq = rec.spec.acquisition
    assert rec.observed.shape == (acq.n_frames, 64, 64)
    assert rec.ground_truth.shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(rec.ground_truth.shifts[0], [0.0, 0.0])


def test_simulate_until_stops_after_named_stage():
    rec = simulate(_full_spec(save_intermediates=True), until="cells_only")
    # only render (and the cell steps before it) ran -> just the cells_only snapshot
    assert set(rec.snapshots) == {"cells_only"}
    # sensor never ran, so observed is the raw render movie (not integer counts)
    assert not np.allclose(rec.observed, np.round(rec.observed))


def test_simulate_save_intermediates_records_movie_stage_names():
    rec = simulate(_full_spec(save_intermediates=True))
    assert set(rec.snapshots) == {
        "cells_only", "neuropil", "bleaching", "brain_motion", "vignette", "leakage", "sensor",
    }
    # cell-domain steps are not snapshotted (they don't touch the movie)
    assert "place_neurons" not in rec.snapshots
    assert "optics" not in rec.snapshots
    # the sensor-stage snapshot is the observed movie
    np.testing.assert_array_equal(rec.stage("sensor").values, rec.observed)


def test_simulate_without_intermediates_keeps_no_snapshots():
    rec = simulate(_minimal_spec())
    assert rec.snapshots == {}


def test_simulate_rejects_unknown_until_stage():
    with pytest.raises(ValueError, match="until="):
        simulate(_minimal_spec(), until="not_a_stage")
