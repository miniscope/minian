"""Unit tests for the minimal runnable step chain (migration Step 5a).

Covers the four executable steps that turn a blank ``Scene`` into a digitized
recording — ``place_somata`` → ``cell_activity`` → ``render`` → ``sensor`` — both
in isolation (each step run against a hand-built scene, the primary test
substrate) and as the end-to-end chain. The ``optics`` step and the
planted/observed footprint split are migration Step 5b and are out of scope
here; ``render`` therefore composites the planted footprint.
"""

import numpy as np
import pytest

from minian.simulation import (
    Acquisition,
    CellActivity,
    CellOptics,
    ImageSensor,
    Optics,
    PlaceSomata,
    Render,
    Scene,
    Sensor,
    SNRDistribution,
)
from minian.simulation.steps import (
    CellActivityStep,
    CellOpticsStep,
    PlaceSomataStep,
    RenderStep,
    SensorStep,
    calcium_kernel,
    degrade_footprint,
    resolve_focal_plane,
    soma_footprint,
)
from minian.simulation.scene import Cell


def _acq(n_px=50, fps=20.0, duration_s=2.0, bit_depth=8, **kw):
    """A small scene with a clean 1.0 µm/px scale (pitch 8 µm / magnification 8).

    50 px → a 50 µm FOV (area 2.5e-3 mm²), so an (unphysically high, on purpose)
    density makes the cell count a clean integer for assertions.
    """
    kw.setdefault("optics", Optics(magnification=8.0))
    kw.setdefault(
        "image_sensor",
        ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth
        ),
    )
    return Acquisition(fps=fps, duration_s=duration_s, **kw)


# --- place_somata ----------------------------------------------------------


def test_place_somata_count_from_density_and_fov():
    # 50 µm FOV → 2.5e-3 mm²; density 2000/mm² → exactly 5 cells.
    acq = _acq()
    step = PlaceSomata(
        density_per_mm2=2000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(0))
    scene = Scene.zeros(acq)
    step(scene)
    assert len(scene.cells) == 5


def test_place_somata_centers_in_bounds():
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    step = PlaceSomata(density_per_mm2=2000.0, depth_range_um=(10.0, 80.0)).build(
        acq, np.random.default_rng(1)
    )
    scene = Scene.zeros(acq)
    step(scene)
    for cell in scene.cells:
        z, y, x = cell.center_um
        assert 10.0 <= z <= 80.0
        assert 0.0 <= y <= fov_h
        assert 0.0 <= x <= fov_w


def test_place_somata_footprint_is_peak_normalized():
    acq = _acq()
    step = PlaceSomata(
        density_per_mm2=2000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(2))
    scene = Scene.zeros(acq)
    step(scene)
    for cell in scene.cells:
        fp = cell.footprint_planted
        assert fp.shape == (50, 50)
        assert fp.max() == pytest.approx(1.0)
        assert fp.min() >= 0.0
        assert (fp > 0).sum() > 0


def test_irregularity_zero_is_a_clean_binary_disk():
    fp = soma_footprint(
        (40, 40),
        (20.0, 20.0),
        radius_px=8.0,
        irregularity=0.0,
        rng=np.random.default_rng(3),
    )
    # A clean disk is binary (0 outside, 1 inside) and roughly the disk's area.
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() == pytest.approx(np.pi * 8.0**2, rel=0.1)


def test_snr_uniform_within_range():
    acq = _acq()
    step = PlaceSomata(
        density_per_mm2=2000.0,
        depth_range_um=(0.0, 0.0),
        snr=SNRDistribution(distribution="uniform", low=2.0, high=6.0),
    ).build(acq, np.random.default_rng(4))
    scene = Scene.zeros(acq)
    step(scene)
    snrs = np.array([c.snr for c in scene.cells])
    assert ((snrs >= 2.0) & (snrs <= 6.0)).all()


def test_min_distance_is_respected():
    acq = _acq()
    step = PlaceSomata(
        density_per_mm2=2000.0, depth_range_um=(0.0, 0.0), min_distance_um=8.0
    ).build(acq, np.random.default_rng(5))
    scene = Scene.zeros(acq)
    step(scene)
    centers = np.array([c.center_um for c in scene.cells])
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.linalg.norm(centers[i] - centers[j]) >= 8.0


def test_place_somata_is_reproducible():
    acq = _acq()
    spec = PlaceSomata(density_per_mm2=2000.0, depth_range_um=(0.0, 50.0))
    scenes = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        spec.build(acq, np.random.default_rng(7))(scene)
        scenes.append([c.center_um for c in scene.cells])
    assert scenes[0] == scenes[1]


def test_neurite_stubs_not_implemented():
    acq = _acq()
    step = PlaceSomata(density_per_mm2=2000.0, n_neurite_stubs=1).build(
        acq, np.random.default_rng(8)
    )
    with pytest.raises(NotImplementedError, match="neurite"):
        step(Scene.zeros(acq))


# --- cell_activity ---------------------------------------------------------


def test_calcium_kernel_shape():
    k = calcium_kernel(tau_rise_s=0.05, tau_decay_s=0.5, fps=20.0)
    assert k.max() == pytest.approx(1.0)
    assert (k >= 0).all()
    assert k[0] == pytest.approx(0.0, abs=1e-9)  # k(0) = 1 - 1 = 0
    assert k[-1] < k.max()  # has decayed by the end of the window


def test_calcium_kernel_requires_rise_faster_than_decay():
    with pytest.raises(ValueError, match="tau_rise_s"):
        calcium_kernel(tau_rise_s=0.5, tau_decay_s=0.5, fps=20.0)


def test_cell_activity_sets_trace_and_spikes():
    acq = _acq(duration_s=5.0)
    scene = Scene.zeros(acq)
    PlaceSomata(density_per_mm2=2000.0, depth_range_um=(0.0, 0.0)).build(
        acq, np.random.default_rng(9)
    )(scene)
    CellActivity(active_rate_hz=5.0, tau_decay_s=0.4).build(
        acq, np.random.default_rng(9)
    )(scene)
    for cell in scene.cells:
        assert cell.trace.shape == (acq.n_frames,)
        assert cell.spikes.shape == (acq.n_frames,)
        assert (cell.spikes >= 0).all()
        np.testing.assert_array_equal(
            cell.spikes, np.round(cell.spikes)
        )  # integer counts


def test_noise_free_trace_never_dips_below_baseline():
    # With trace_noise=0 the trace is f0 + (nonneg amplitudes) ⊛ (nonneg kernel),
    # so it can never fall below f0.
    acq = _acq(duration_s=5.0)
    scene = Scene.zeros(acq)
    scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0), snr=4.0))
    CellActivity(f0=1.0, trace_noise=0.0, active_rate_hz=5.0).build(
        acq, np.random.default_rng(10)
    )(scene)
    assert scene.cells[0].trace.min() >= 1.0 - 1e-9


def test_cell_activity_is_reproducible():
    acq = _acq(duration_s=5.0)
    traces = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0), snr=4.0))
        CellActivity().build(acq, np.random.default_rng(11))(scene)
        traces.append(scene.cells[0].trace)
    np.testing.assert_array_equal(traces[0], traces[1])


# --- render ----------------------------------------------------------------


def test_render_is_the_footprint_trace_outer_sum():
    acq = _acq(n_px=8, duration_s=0.15)  # 3 frames
    scene = Scene.zeros(acq)
    fp1 = np.zeros((8, 8))
    fp1[2, 3] = 1.0
    fp2 = np.zeros((8, 8))
    fp2[5, 6] = 1.0
    tr1 = np.array([1.0, 2.0, 3.0])
    tr2 = np.array([4.0, 5.0, 6.0])
    scene.cells += [
        Cell(center_um=(0.0, 0.0, 0.0), snr=1.0, footprint_planted=fp1, trace=tr1),
        Cell(center_um=(0.0, 0.0, 0.0), snr=1.0, footprint_planted=fp2, trace=tr2),
    ]
    RenderStep(Render(), acq, np.random.default_rng(0))(scene)
    np.testing.assert_allclose(scene.movie.values[:, 2, 3], tr1)
    np.testing.assert_allclose(scene.movie.values[:, 5, 6], tr2)
    # Pixels with no cell stay zero.
    assert scene.movie.values[:, 0, 0].sum() == 0.0


def test_render_prefers_observed_footprint_when_present():
    acq = _acq(n_px=8, duration_s=0.05)  # 1 frame
    scene = Scene.zeros(acq)
    planted = np.zeros((8, 8))
    planted[1, 1] = 1.0
    observed = np.zeros((8, 8))
    observed[4, 4] = 1.0
    scene.cells.append(
        Cell(
            center_um=(0.0, 0.0, 0.0),
            snr=1.0,
            footprint_planted=planted,
            footprint_observed=observed,
            trace=np.array([2.0]),
        )
    )
    RenderStep(Render(), acq, np.random.default_rng(0))(scene)
    assert scene.movie.values[0, 4, 4] == pytest.approx(2.0)  # observed used
    assert scene.movie.values[0, 1, 1] == 0.0  # planted ignored


def test_render_empty_scene_leaves_movie_untouched():
    acq = _acq(n_px=8, duration_s=0.15)
    scene = Scene.zeros(acq)
    RenderStep(Render(), acq, np.random.default_rng(0))(scene)
    assert (scene.movie.values == 0.0).all()


# --- sensor ----------------------------------------------------------------


def test_sensor_counts_are_integer_and_within_adc_range():
    acq = _acq(n_px=16, duration_s=0.5, bit_depth=8)
    scene = Scene.zeros(acq)
    scene.movie.values[:] = 1.5  # uniform positive intensity
    SensorStep(Sensor(photons_per_unit=100.0), acq, np.random.default_rng(0))(scene)
    counts = scene.movie.values
    np.testing.assert_array_equal(counts, np.round(counts))  # integer-valued
    assert counts.min() >= 0.0
    assert counts.max() <= 255.0  # 2^8 - 1


def test_sensor_mean_counts_increase_with_exposure():
    acq = _acq(n_px=16, duration_s=0.5, bit_depth=12)  # headroom so we don't saturate
    means = []
    for ppu in (20.0, 80.0):
        scene = Scene.zeros(acq)
        scene.movie.values[:] = 1.0
        SensorStep(Sensor(photons_per_unit=ppu), acq, np.random.default_rng(0))(scene)
        means.append(scene.movie.values.mean())
    assert means[1] > means[0]


def test_sensor_is_reproducible():
    acq = _acq(n_px=16, duration_s=0.5)
    outs = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        scene.movie.values[:] = 1.0
        SensorStep(Sensor(), acq, np.random.default_rng(3))(scene)
        outs.append(scene.movie.values.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


# --- optics (5b) -----------------------------------------------------------


def _cell_with_footprint(acq, z, radius_um=4.0):
    """A single centered cell carrying a clean planted disk at depth ``z``."""
    h, w = acq.image_sensor.n_px_height, acq.image_sensor.n_px_width
    fp = soma_footprint(
        (h, w), (h / 2, w / 2), acq.um_to_px(radius_um), 0.0, np.random.default_rng(0)
    )
    y_um, x_um = (h / 2) * acq.pixel_size_um, (w / 2) * acq.pixel_size_um
    return Cell(center_um=(z, y_um, x_um), snr=4.0, footprint_planted=fp)


def test_degrade_footprint_blur_conserves_sum_and_drops_peak():
    fp = soma_footprint(
        (64, 64),
        (32.0, 32.0),
        radius_px=6.0,
        irregularity=0.0,
        rng=np.random.default_rng(0),
    )
    out = degrade_footprint(fp, sigma_px=2.0, attenuation=1.0)
    assert out.sum() == pytest.approx(
        fp.sum(), rel=1e-3
    )  # convolution conserves integral
    assert out.max() < fp.max()  # ...but spreads light, so the peak drops


def test_degrade_footprint_attenuation_scales_integral():
    fp = soma_footprint(
        (64, 64),
        (32.0, 32.0),
        radius_px=6.0,
        irregularity=0.0,
        rng=np.random.default_rng(0),
    )
    full = degrade_footprint(fp, sigma_px=2.0, attenuation=1.0)
    half = degrade_footprint(fp, sigma_px=2.0, attenuation=0.5)
    assert half.sum() == pytest.approx(0.5 * full.sum())


def test_resolve_focal_plane_auto_is_median_and_numeric_passes_through():
    acq = _acq()
    cells = [_cell_with_footprint(acq, z) for z in (0.0, 50.0, 100.0, 150.0, 200.0)]
    assert resolve_focal_plane(cells, acq.optics) == 100.0
    assert resolve_focal_plane([], acq.optics) == 0.0  # empty scene -> surface
    numeric = _acq(optics=Optics(magnification=8.0, focal_plane_um=42.0))
    assert resolve_focal_plane(cells, numeric.optics) == 42.0


def test_optics_in_focus_surface_cell_is_barely_degraded():
    # z=0, focal auto -> 0: no scatter, no defocus, atten=1 -> only diffraction.
    acq = _acq(n_px=64)
    scene = Scene.zeros(acq)
    scene.cells.append(_cell_with_footprint(acq, z=0.0))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    cell = scene.cells[0]
    assert cell.in_focus is True
    assert cell.optical_brightness == pytest.approx(1.0)
    assert cell.footprint_observed.sum() == pytest.approx(
        cell.footprint_planted.sum(), rel=1e-2
    )
    assert cell.detectable is None  # deferred to finalize (Step 6)


def test_optics_deeper_cell_is_broader_and_dimmer():
    # Both in focus (focal == z, defocus 0), so the difference is pure depth:
    # scatter broadens the footprint and attenuation removes light.
    def run(z):
        acq = _acq(n_px=80, optics=Optics(magnification=8.0, focal_plane_um=z))
        scene = Scene.zeros(acq)
        scene.cells.append(_cell_with_footprint(acq, z=z))
        CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
        return scene.cells[0]

    shallow, deep = run(10.0), run(180.0)
    assert deep.optical_brightness < shallow.optical_brightness  # attenuation
    assert deep.footprint_observed.sum() < shallow.footprint_observed.sum()
    assert (
        deep.footprint_observed.max() < shallow.footprint_observed.max()
    )  # broader + dimmer


def test_optics_defocus_conserves_observed_integral():
    # Fixed depth, sweep the focal plane: defocus broadens the footprint but
    # (being a convolution) conserves its integral; attenuation(z) is fixed.
    z, sums = 50.0, []
    for focal in (48.0, 50.0, 52.0):
        acq = _acq(n_px=80, optics=Optics(magnification=8.0, focal_plane_um=focal))
        scene = Scene.zeros(acq)
        scene.cells.append(_cell_with_footprint(acq, z=z, radius_um=3.0))
        CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
        sums.append(scene.cells[0].footprint_observed.sum())
    assert sums == pytest.approx([sums[0]] * 3, rel=1e-2)


def test_optics_in_focus_flag_respects_depth_of_field():
    acq = _acq(
        n_px=64,
        optics=Optics(magnification=8.0, focal_plane_um=80.0, depth_of_field_um=15.0),
    )
    scene = Scene.zeros(acq)
    for z in (70.0, 80.0, 96.0):  # within DOF, at plane, just outside DOF
        scene.cells.append(_cell_with_footprint(acq, z=z))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    assert [c.in_focus for c in scene.cells] == [True, True, False]


def test_optics_makes_render_use_the_degraded_footprint():
    # Render a deep cell from its planted footprint, then again after optics:
    # the optically degraded render is dimmer (blurred + attenuated).
    acq = _acq(n_px=40, duration_s=1.0)
    rng = np.random.default_rng(1)
    scene = Scene.zeros(acq)
    PlaceSomata(
        density_per_mm2=2000.0, soma_radius_um=4.0, depth_range_um=(120.0, 120.0)
    ).build(acq, rng)(scene)
    CellActivity(active_rate_hz=5.0).build(acq, rng)(scene)

    RenderStep(Render(), acq, rng)(scene)  # observed still None -> uses planted
    planted_peak = scene.movie.values.max()

    scene.movie.values[:] = 0.0
    CellOpticsStep(CellOptics(), acq, rng)(scene)
    RenderStep(Render(), acq, rng)(scene)  # now uses footprint_observed
    observed_peak = scene.movie.values.max()

    assert all(c.footprint_observed is not None for c in scene.cells)
    assert observed_peak < planted_peak


def test_optics_chain_with_sensor_runs_end_to_end():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)
    rng = np.random.default_rng(99)
    scene = Scene.zeros(acq)
    steps = [
        PlaceSomata(
            density_per_mm2=3000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)
    assert all(c.footprint_observed is not None for c in scene.cells)
    movie = scene.movie.values
    np.testing.assert_array_equal(movie, np.round(movie))
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.max() > 0.0 and movie.var() > 0.0


# --- the minimal chain -----------------------------------------------------


def test_minimal_chain_place_activity_render_sensor():
    acq = _acq(n_px=40, duration_s=2.0, bit_depth=8)
    rng = np.random.default_rng(2026)
    scene = Scene.zeros(acq)
    steps = [
        PlaceSomata(
            density_per_mm2=3000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Render(),
        Sensor(photons_per_unit=100.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    assert len(scene.cells) > 0
    movie = scene.movie.values
    assert movie.shape == (acq.n_frames, 40, 40)
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.max() > 0.0  # cells produced signal
    assert movie.var() > 0.0  # spatial/temporal structure, not a flat field
