"""Unit tests for the typed output + finalize() transform (migration Step 6a).

Covers ``finalize(scene, spec)`` turning an exhausted ``Scene`` into a
``Recording``/``GroundTruth``: array shapes and dtypes, the planted-vs-observed
footprint split, FOV cropping of margin cells under motion, the per-effect
ground-truth fields (present vs ``None``), the detectability rule, and
``detectable_subset()``. The ``simulate()`` orchestrator is Step 6b.
"""

import numpy as np
import pytest

from minian.simulation import (
    Acquisition,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    GroundTruth,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    PlaceNeurons,
    Render,
    Scene,
    Sensor,
    Spec,
    Vignette,
    finalize,
)
from minian.simulation.scene import Cell


def _acq(n_px=20, fps=20.0, duration_s=1.0, bit_depth=8, **kw):
    """Small scene at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    kw.setdefault("optics", Optics(magnification=8.0))
    kw.setdefault(
        "image_sensor",
        ImageSensor(n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth),
    )
    return Acquisition(fps=fps, duration_s=duration_s, **kw)


def _run(acq, steps, seed=0, margin_px=0):
    """Run a step list against a fresh (optionally margined) scene."""
    rng = np.random.default_rng(seed)
    scene = Scene.zeros(acq, margin_px=margin_px)
    for sspec in steps:
        sspec.build(acq, rng)(scene)
    return scene


def _dot(shape, iy, ix, value=1.0):
    fp = np.zeros(shape)
    fp[iy, ix] = value
    return fp


# --- shapes / types --------------------------------------------------------


def test_finalize_produces_typed_recording():
    acq = _acq(n_px=24, duration_s=1.0)
    steps = [
        PlaceNeurons(density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Sensor(photons_per_unit=100.0),
    ]
    rec = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps))
    gt = rec.ground_truth
    n = gt.n_units
    assert n > 0
    assert gt.A_planted.shape == (n, 24, 24)
    assert gt.A_observed.shape == (n, 24, 24)
    assert gt.C.shape == (n, acq.n_frames)
    assert gt.S.shape == (n, acq.n_frames)
    assert gt.centers_um.shape == (n, 3)
    assert gt.in_focus.shape == (n,) and gt.detectable.shape == (n,)
    # observed is the integer-count movie downcast to the store dtype.
    assert rec.observed.shape == (acq.n_frames, 24, 24)
    assert rec.observed.dtype == np.float32
    np.testing.assert_array_equal(rec.observed, np.round(rec.observed))


def test_observed_footprint_differs_from_planted_under_optics():
    acq = _acq(n_px=40, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=0.0)
    steps = [
        PlaceNeurons(density_per_mm3=62500.0, soma_radius_um=4.0, depth_range_um=(80.0, 120.0)),
        CellActivity(active_rate_hz=5.0),
        CellOptics(),
        Render(),
        Sensor(),
    ]
    gt = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps)).ground_truth
    # Deep cells: scatter + defocus make the observed footprint dimmer/broader.
    assert not np.allclose(gt.A_planted, gt.A_observed)
    assert gt.A_observed.sum() < gt.A_planted.sum()


# --- per-effect fields -----------------------------------------------------


def test_per_effect_fields_are_none_when_steps_absent():
    acq = _acq()
    steps = [
        PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        Render(),
        Sensor(),
    ]
    gt = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps)).ground_truth
    assert gt.shifts is None
    assert gt.vignette is None
    assert gt.leakage is None
    assert gt.bleaching is None
    assert gt.neuropil_spatial is None and gt.neuropil_temporal is None


def test_per_effect_fields_present_for_full_pipeline():
    acq = _acq(n_px=32, duration_s=1.0)
    max_shift_um = 3.0
    margin = int(np.ceil(acq.um_to_px(max_shift_um))) + 1
    steps = [
        PlaceNeurons(density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 100.0)),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Neuropil(n_components=2),
        Bleaching(final_fraction=0.7),
        BrainMotion(walk_step_um=0.3, max_shift_um=max_shift_um),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(),
    ]
    gt = finalize(
        _run(acq, steps, margin_px=margin), Spec(acquisition=acq, steps=steps)
    ).ground_truth
    assert gt.shifts.shape == (acq.n_frames, 2)
    assert gt.vignette.shape == (32, 32)  # FOV-sized
    assert gt.leakage.shape == (32, 32)
    assert gt.bleaching.shape == (acq.n_frames,)
    assert gt.neuropil_temporal.shape == (2, acq.n_frames)
    assert gt.neuropil_spatial.shape == (2, 32, 32)  # cropped to FOV grid


# --- FOV cropping of margin cells ------------------------------------------


def test_finalize_drops_pure_margin_cells():
    acq = _acq(n_px=20)
    margin = 6
    canvas = 20 + 2 * margin
    nf = acq.n_frames
    scene = Scene.zeros(acq, margin_px=margin)
    # one cell inside the FOV, one entirely in the top margin (rows < margin).
    scene.cells += [
        Cell(
            center_um=(0.0, margin + 10.0, margin + 10.0),
            footprint_planted=_dot((canvas, canvas), margin + 10, margin + 10),
            trace=np.ones(nf),
        ),
        Cell(
            center_um=(0.0, 2.0, 10.0),
            footprint_planted=_dot((canvas, canvas), 2, 10),
            trace=np.ones(nf),
        ),
    ]
    minimal = [PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)), Render()]
    gt = finalize(scene, Spec(acquisition=acq, steps=minimal)).ground_truth
    assert gt.n_units == 1  # the margin-only cell was dropped
    assert gt.A_planted.shape == (1, 20, 20)
    # surviving cell's center is in FOV coordinates (canvas 16 -> FOV 10).
    np.testing.assert_allclose(gt.centers_um[0], [0.0, 10.0, 10.0])


# --- detectability ---------------------------------------------------------


def _detect_spec(acq):
    return Spec(
        acquisition=acq,
        steps=[
            PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
            CellActivity(tau_decay_s=0.4),
            Render(),
            Sensor(photons_per_unit=100.0),
        ],
    )


def test_detectable_reflects_brightness_and_focus():
    acq = _acq(n_px=16)
    nf = acq.n_frames
    trace = np.ones(nf)
    trace[nf // 2] = 5.0  # baseline 1, transient peak 5
    scene = Scene.zeros(acq)
    scene.cells += [
        # bright, in focus -> well above the sensor floor
        Cell(center_um=(0.0, 8.0, 8.0), footprint_planted=_dot((16, 16), 8, 8),
             trace=trace, in_focus=True, optical_brightness=1.0),
        # in focus but optically ~dark -> below the floor
        Cell(center_um=(0.0, 4.0, 4.0), footprint_planted=_dot((16, 16), 4, 4),
             trace=trace, in_focus=True, optical_brightness=1e-3),
        # bright but out of focus -> gated out by in_focus
        Cell(center_um=(0.0, 12.0, 12.0), footprint_planted=_dot((16, 16), 12, 12),
             trace=trace, in_focus=False, optical_brightness=1.0),
    ]
    gt = finalize(scene, _detect_spec(acq)).ground_truth
    assert gt.detectable.tolist() == [True, False, False]
    # detectable is always a subset of in_focus.
    assert not (gt.detectable & ~gt.in_focus).any()


def test_detectable_subset_keeps_only_detectable_units():
    acq = _acq(n_px=16)
    nf = acq.n_frames
    trace = np.ones(nf)
    trace[nf // 2] = 5.0
    scene = Scene.zeros(acq)
    scene.cells += [
        Cell(center_um=(0.0, 8.0, 8.0), footprint_planted=_dot((16, 16), 8, 8),
             trace=trace, in_focus=True, optical_brightness=1.0),
        Cell(center_um=(0.0, 4.0, 4.0), footprint_planted=_dot((16, 16), 4, 4),
             trace=trace, in_focus=True, optical_brightness=1e-3),
    ]
    gt = finalize(scene, _detect_spec(acq)).ground_truth
    sub = gt.detectable_subset()
    assert sub.n_units == 1
    assert sub.detectable.all()
    assert isinstance(sub, GroundTruth)


# --- snapshots / stage -----------------------------------------------------


def test_stage_raises_for_absent_snapshot():
    acq = _acq()
    steps = [PlaceNeurons(depth_range_um=(0.0, 0.0)), Render(), Sensor()]
    rec = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps))
    assert rec.snapshots == {}  # save_intermediates defaulted off
    with pytest.raises(KeyError, match="save_intermediates"):
        rec.stage("observed")
