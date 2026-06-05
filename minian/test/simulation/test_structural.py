"""Whole-pipeline structural tests (migration Step 10a).

These assert that a full ``simulate()`` recording is *physically coherent* — the
foundation the per-stage recovery demos (10b motion, 10c CNMF) stand on. Unlike
``test_steps.py`` (which exercises each step's ``build()`` in isolation), these
run the composed pipeline end to end and check emergent physics that a broken
chain would violate:

* detectability falls as cells go deeper (defocus + scatter),
* a strong vignette preferentially kills rim cells,
* photobleaching dims later frames,
* the static sensor-frame fields are invariant to brain motion.

They use small, fixed-seed specs and decisive (not finely-calibrated) margins —
this is a capability demonstration, not the threshold-calibrated replacement
suite (that lands in a later PR).
"""

import numpy as np

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
    PlaceNeurons,
    Render,
    Sensor,
    Spec,
    Vignette,
    simulate,
    sweep,
)


def _acq(n_px=96, focal_depth_in_tissue_um=0.0, depth_of_field_um=40.0, duration_s=1.0):
    """A ~96 µm FOV at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    return Acquisition(
        fps=20.0,
        duration_s=duration_s,
        focal_depth_in_tissue_um=focal_depth_in_tissue_um,
        optics=Optics(
            magnification=8.0, na=0.45, depth_of_field_um=depth_of_field_um,
        ),
        image_sensor=ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=8
        ),
    )


def test_detectability_falls_with_depth():
    # Focal plane fixed at the surface (z=0): as the cell band descends past the
    # depth of field, cells defocus out (geometric in_focus -> 0) and scatter dims
    # them, so both the in-focus and the detectable fraction fall with depth. The
    # in-focus fraction is the rock-solid geometric signal; detectability tracks it.
    base = Spec(
        acquisition=_acq(focal_depth_in_tissue_um=0.0, depth_of_field_um=40.0, duration_s=3.0),
        seed=10,
        steps=[
            PlaceNeurons(density_per_mm2=6000.0, soma_radius_um=4.0, depth_range_um=(0.0, 10.0)),
            CellActivity(active_rate_hz=8.0, tau_decay_s=0.4),  # ~all cells fire
            CellOptics(),
            Render(),
            # photons ~ 200 / NA² (NA 0.45): the NA²-collection factor moved into
            # cell_optics dims the signal, and photons_per_unit (the exposure scale
            # that absorbs collection's absolute constant) compensates so the
            # detectability regime is unchanged.
            Sensor(photons_per_unit=1000.0),
        ],
    )
    bands = [(0.0, 10.0), (40.0, 50.0), (80.0, 90.0), (120.0, 130.0)]
    in_focus, detectable = [], []
    for spec in sweep(base, {"steps.place_neurons.depth_range_um": bands}):
        gt = simulate(spec).ground_truth
        in_focus.append(gt.in_focus.mean())
        detectable.append(gt.detectable.sum() / gt.n_units)
    # geometric depth-of-field effect: fully in focus at the surface, gone when deep
    assert in_focus[0] > 0.9
    assert in_focus[-1] < 0.1
    assert all(later <= earlier + 0.05 for earlier, later in zip(in_focus, in_focus[1:]))
    # detectability tracks the focus down: a meaningful shallow fraction, ~none deep
    assert detectable[0] > 0.2
    assert detectable[-1] < 0.05
    assert detectable[0] > detectable[-1]


def test_strong_vignette_concentrates_detection_centrally():
    # A steep illumination falloff dims rim cells below the noise floor, so among
    # in-focus cells the detectable ones cluster toward the FOV center.
    spec = Spec(
        acquisition=_acq(focal_depth_in_tissue_um=5.0, depth_of_field_um=40.0),
        seed=11,
        steps=[
            PlaceNeurons(density_per_mm2=6000.0, soma_radius_um=4.0, depth_range_um=(0.0, 10.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Vignette(falloff=0.3, exponent=2.0),  # edge at 30% brightness
            Sensor(photons_per_unit=110.0),
        ],
    )
    rec = simulate(spec)
    gt = rec.ground_truth
    acq = rec.spec.acquisition
    yx = gt.centers_um[:, 1:] / acq.pixel_size_um
    center = np.array([acq.image_sensor.n_px_height, acq.image_sensor.n_px_width]) / 2.0
    r = np.linalg.norm(yx - center, axis=1)
    inner = r < np.median(r)
    assert gt.detectable[inner].mean() > gt.detectable[~inner].mean()


def test_bleaching_dims_later_frames():
    spec = Spec(
        acquisition=_acq(focal_depth_in_tissue_um=5.0, duration_s=2.0),
        seed=12,
        steps=[
            PlaceNeurons(density_per_mm2=4000.0, soma_radius_um=4.0, depth_range_um=(0.0, 10.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Neuropil(n_components=2, amplitude=0.4),  # a background floor to dim
            Bleaching(model="mono_exp", final_fraction=0.5),
            Sensor(photons_per_unit=140.0),
        ],
    )
    mov = simulate(spec).observed
    q = mov.shape[0] // 4
    assert mov[-q:].mean() < mov[:q].mean()


def test_static_fields_are_invariant_to_motion():
    # Vignette and leakage are sensor-frame (static); their ground-truth fields
    # must be identical whether or not the brain moves, even though the recorded
    # shift trajectory differs. Same seed, differing only in motion magnitude.
    def _spec(max_shift_um, walk_step_um):
        return Spec(
            acquisition=_acq(focal_depth_in_tissue_um=5.0),
            seed=13,
            steps=[
                PlaceNeurons(density_per_mm2=4000.0, soma_radius_um=4.0, depth_range_um=(0.0, 10.0)),
                CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
                CellOptics(),
                Render(),
                BrainMotion(walk_step_um=walk_step_um, max_shift_um=max_shift_um),
                Vignette(falloff=0.4, exponent=2.0),
                Leakage(profile="gaussian", level=0.12, sigma_um=80.0),
                Sensor(photons_per_unit=130.0),
            ],
        )

    still = simulate(_spec(max_shift_um=0.5, walk_step_um=0.05)).ground_truth
    moving = simulate(_spec(max_shift_um=5.0, walk_step_um=0.6)).ground_truth

    np.testing.assert_array_equal(still.vignette, moving.vignette)
    np.testing.assert_array_equal(still.leakage, moving.leakage)
    # the recordings really do differ in how much the brain moved
    assert np.abs(moving.shifts).max() > np.abs(still.shifts).max()
