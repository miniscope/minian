"""Unit tests for the executable step chain (migration Steps 5a–5c).

Covers the steps that turn a blank ``Scene`` into a digitized recording — the
minimal chain ``place_neurons`` → ``cell_activity`` → ``render`` → ``sensor``
(5a), the ``optics`` degradation and planted/observed split (5b), and the field
effects ``neuropil`` / ``bleaching`` / ``vignette`` / ``leakage`` plus the
``vasculature`` no-op placeholder (5c). Each step is exercised in isolation
against a hand-built scene (the primary test substrate) as well as in the
end-to-end chain. ``brain_motion`` (5d) is out of scope here.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

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
    Scene,
    Sensor,
    Vasculature,
    Vignette,
)
from minian.simulation.steps import (
    BleachingStep,
    BrainMotionStep,
    CellActivityStep,
    CellOpticsStep,
    LeakageStep,
    NeuropilStep,
    PlaceNeuronsStep,
    RenderStep,
    SensorStep,
    VasculatureStep,
    VignetteStep,
    bleaching_curve,
    bounded_random_walk,
    calcium_kernel,
    degrade_footprint,
    kernel_timing,
    spike_activity_params,
    neuron_footprint,
    ou_process,
    resolve_focal_plane,
    sample_neurons,
    shift_and_crop,
    tau_from_kernel_timing,
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


# --- place_neurons ----------------------------------------------------------


def test_place_neurons_count_from_density_and_fov():
    # 50 µm FOV → 2.5e-3 mm²; depth (0,0) floors to one soma diameter (2·5 = 10 µm
    # → 1e-2 mm), so volume = 2.5e-5 mm³ and 200000/mm³ → exactly 5 cells.
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=200000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(0))
    scene = Scene.zeros(acq)
    step(scene)
    assert len(scene.cells) == 5


def test_place_neurons_centers_in_bounds():
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    step = PlaceNeurons(density_per_mm3=28571.0, depth_range_um=(10.0, 80.0)).build(
        acq, np.random.default_rng(1)
    )
    scene = Scene.zeros(acq)
    step(scene)
    for cell in scene.cells:
        z, y, x = cell.center_um
        assert 10.0 <= z <= 80.0
        assert 0.0 <= y <= fov_h
        assert 0.0 <= x <= fov_w


def test_place_neurons_footprint_is_peak_normalized():
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=200000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
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
    fp = neuron_footprint(
        (40, 40),
        (20.0, 20.0),
        radius_px=8.0,
        irregularity=0.0,
        rng=np.random.default_rng(3),
    )
    # A clean disk is binary (0 outside, 1 inside) and roughly the disk's area.
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() == pytest.approx(np.pi * 8.0**2, rel=0.1)


def _placed_scene(acq, seed=4):
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)).build(
        acq, np.random.default_rng(seed)
    )(scene)
    return scene


def test_brightness_gain_is_per_cell_and_scales_the_whole_trace():
    # cell_activity draws a per-cell expression gain (mean 1, spread brightness_cv)
    # and scales each cell's *whole* trace by it -- baseline included -- so a bright
    # cell is brighter everywhere.
    acq = _acq()
    scene = _placed_scene(acq)
    spec = CellActivity(brightness_cv=0.5)
    spec.build(acq, np.random.default_rng(4))(scene)
    amps = np.array([c.amplitude for c in scene.cells])
    assert (amps > 0).all() and amps.std() > 0  # a real cell-to-cell spread
    # the convolved calcium is non-negative, so the trace never dips below the
    # cell's scaled baseline gain*f0 -- the gain is applied to the whole trace.
    for cell in scene.cells:
        assert cell.trace.min() >= cell.amplitude * spec.f0 - 1e-6


def test_brightness_cv_zero_makes_every_cell_equally_bright():
    acq = _acq()
    scene = _placed_scene(acq)
    CellActivity(brightness_cv=0.0).build(acq, np.random.default_rng(4))(scene)
    amps = np.array([c.amplitude for c in scene.cells])
    np.testing.assert_array_equal(amps, np.ones_like(amps))


def test_min_distance_is_respected():
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=142857.0, depth_range_um=(0.0, 0.0), min_distance_um=8.0
    ).build(acq, np.random.default_rng(5))
    scene = Scene.zeros(acq)
    step(scene)
    centers = np.array([c.center_um for c in scene.cells])
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.linalg.norm(centers[i] - centers[j]) >= 8.0


def test_place_neurons_is_reproducible():
    acq = _acq()
    spec = PlaceNeurons(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0))
    scenes = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        spec.build(acq, np.random.default_rng(7))(scene)
        scenes.append([c.center_um for c in scene.cells])
    assert scenes[0] == scenes[1]


def test_sample_neurons_matches_the_full_step():
    # The extracted distribution sampler must reproduce, draw-for-draw, the
    # centers the fused PlaceNeuronsStep would have placed — same spec, same FOV,
    # same seeded rng. (This is the contract the notebook relies on: visualizing
    # sample_neurons() shows the *real* placement, not an approximation.)
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    spec = PlaceNeurons(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0))
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(7))

    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(7))(scene)
    step_centers = [c.center_um for c in scene.cells]

    assert centers == step_centers


def test_sample_neurons_count_from_density_and_fov():
    # No footprints stamped, but the volumetric count rule still gives a clean 5:
    # 2.5e-3 mm² × floor(2·7 = 14 µm) = 3.5e-5 mm³, × 142857/mm³ = 5.
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    spec = PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0))
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(0))
    assert len(centers) == 5


def test_cytosolic_morphology_adds_dendrites_beyond_soma():
    # Same seed, same soma: cytosolic lights strictly more pixels (the dendrites)
    # and reaches farther from the center than the soma-only variant, while the
    # peak stays at the soma (1.0) and the dendrites are graded dimmer.
    shape, center, radius = (80, 80), (40.0, 40.0), 6.0
    soma = neuron_footprint(
        shape, center, radius, irregularity=0.0, rng=np.random.default_rng(0)
    )
    cyto = neuron_footprint(
        shape,
        center,
        radius,
        irregularity=0.0,
        rng=np.random.default_rng(0),
        morphology="cytosolic",
        n_dendrites=4,
        dendrite_length_px=18.0,
        dendrite_width_px=2.5,
    )
    assert cyto.max() == pytest.approx(1.0)  # still peak-normalized at the soma
    assert (cyto > 0).sum() > (soma > 0).sum()  # dendrites light extra pixels
    yy, xx = np.indices(shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    assert rr[cyto > 0].max() > rr[soma > 0].max() + 2.0  # reach beyond the soma
    assert 0.0 < cyto[cyto > 0].min() < 1.0  # dendrites graded dimmer than soma


def test_soma_morphology_is_unchanged_by_dendrite_params():
    # The default "soma" variant ignores the dendrite params and matches a
    # soma-only footprint bit-for-bit (cytosolic only *adds* after the soma, so
    # the soma's RNG draw is never perturbed).
    shape, center, radius = (60, 60), (30.0, 30.0), 5.0
    a = neuron_footprint(
        shape, center, radius, irregularity=0.3, rng=np.random.default_rng(11)
    )
    b = neuron_footprint(
        shape,
        center,
        radius,
        irregularity=0.3,
        rng=np.random.default_rng(11),
        morphology="soma",
        n_dendrites=4,
        dendrite_length_px=18.0,
        dendrite_width_px=2.5,
    )
    np.testing.assert_array_equal(a, b)


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


@pytest.mark.parametrize("tr,td", [(0.05, 0.5), (0.02, 0.3), (0.1, 0.8), (0.08, 0.4)])
def test_kernel_timing_round_trips_with_tau(tr, td):
    # (tau_rise, tau_decay) -> (t_peak, fwhm) -> (tau_rise, tau_decay) is identity.
    t_peak, fwhm = kernel_timing(tr, td)
    tr2, td2 = tau_from_kernel_timing(t_peak, fwhm)
    assert tr2 == pytest.approx(tr, rel=1e-3)
    assert td2 == pytest.approx(td, rel=1e-3)


def test_kernel_timing_matches_a_finely_sampled_kernel():
    # The analytic peak time and FWHM must match what a densely sampled kernel shows.
    tr, td, fps = 0.05, 0.5, 2000.0
    t_peak, fwhm = kernel_timing(tr, td)
    k = calcium_kernel(tr, td, fps)
    t = np.arange(len(k)) / fps
    assert t[k.argmax()] == pytest.approx(t_peak, abs=1.0 / fps)
    above = t[k >= 0.5]  # kernel is peak-normalized to 1, so half max = 0.5
    assert (above.max() - above.min()) == pytest.approx(fwhm, abs=2.0 / fps)


def test_tau_from_kernel_timing_clamps_impossible_ratio():
    # t_peak/fwhm above the kernel's achievable range clamps to the alpha-function
    # limit (tau_rise -> tau_decay) instead of failing.
    tr, td = tau_from_kernel_timing(t_peak_s=0.3, fwhm_s=0.35)
    assert tr == pytest.approx(td, rel=1e-3)


def test_spike_activity_params_hit_calab_levels():
    # activity 0/0.5/1 reproduce CaLab's sparse/moderate/dense SPIKE_ACTIVITY_LEVELS.
    assert spike_activity_params(0.0) == pytest.approx((0.002, 0.4, 90.0, 0.3))
    assert spike_activity_params(0.5) == pytest.approx((0.005, 0.3, 150.0, 0.6))
    assert spike_activity_params(1.0) == pytest.approx((0.01, 0.2, 210.0, 1.5))


def test_spike_activity_params_are_monotonic_in_density():
    # Denser activity: bursts start more often (p_q2a up), last longer (p_a2q down),
    # fire harder (active_rate up), and the background rate rises.
    sparse, dense = spike_activity_params(0.2), spike_activity_params(0.8)
    assert dense[0] > sparse[0]   # p_quiescent_to_active
    assert dense[1] < sparse[1]   # p_active_to_quiescent
    assert dense[2] > sparse[2]   # active_rate_hz
    assert dense[3] > sparse[3]   # quiescent_rate_hz
    assert spike_activity_params(2.0) == spike_activity_params(1.0)  # clamps


def test_cell_activity_sets_trace_and_spikes():
    acq = _acq(duration_s=5.0)
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)).build(
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
    scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0)))
    # brightness_cv=0 keeps the per-cell gain at 1, so the baseline stays exactly f0
    # and this isolates the kernel/amplitude non-negativity invariant.
    CellActivity(f0=1.0, trace_noise=0.0, active_rate_hz=5.0, brightness_cv=0.0).build(
        acq, np.random.default_rng(10)
    )(scene)
    assert scene.cells[0].trace.min() >= 1.0 - 1e-9


def test_cell_activity_is_reproducible():
    acq = _acq(duration_s=5.0)
    traces = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0)))
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
        Cell(center_um=(0.0, 0.0, 0.0), footprint_planted=fp1, trace=tr1),
        Cell(center_um=(0.0, 0.0, 0.0), footprint_planted=fp2, trace=tr2),
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
    fp = neuron_footprint(
        (h, w), (h / 2, w / 2), acq.um_to_px(radius_um), 0.0, np.random.default_rng(0)
    )
    y_um, x_um = (h / 2) * acq.pixel_size_um, (w / 2) * acq.pixel_size_um
    return Cell(center_um=(z, y_um, x_um), footprint_planted=fp)


def test_degrade_footprint_blur_conserves_sum_and_drops_peak():
    fp = neuron_footprint(
        (64, 64),
        (32.0, 32.0),
        radius_px=6.0,
        irregularity=0.0,
        rng=np.random.default_rng(0),
    )
    out = degrade_footprint(fp, sigma_px=2.0, gain=1.0)
    assert out.sum() == pytest.approx(
        fp.sum(), rel=1e-3
    )  # convolution conserves integral
    assert out.max() < fp.max()  # ...but spreads light, so the peak drops


def test_degrade_footprint_gain_scales_integral():
    fp = neuron_footprint(
        (64, 64),
        (32.0, 32.0),
        radius_px=6.0,
        irregularity=0.0,
        rng=np.random.default_rng(0),
    )
    full = degrade_footprint(fp, sigma_px=2.0, gain=1.0)
    half = degrade_footprint(fp, sigma_px=2.0, gain=0.5)
    assert half.sum() == pytest.approx(0.5 * full.sum())


def test_resolve_focal_plane_auto_is_median_and_numeric_passes_through():
    acq = _acq()  # focal_depth_in_tissue_um defaults to "auto"
    cells = [_cell_with_footprint(acq, z) for z in (0.0, 50.0, 100.0, 150.0, 200.0)]
    assert resolve_focal_plane(cells, acq.focal_depth_in_tissue_um) == 100.0
    assert resolve_focal_plane([], acq.focal_depth_in_tissue_um) == 0.0  # empty -> surface
    numeric = _acq(focal_depth_in_tissue_um=42.0)
    assert resolve_focal_plane(cells, numeric.focal_depth_in_tissue_um) == 42.0


def test_resolve_focal_plane_auto_accounts_for_field_curvature():
    # All cells at the same depth but spread laterally. Defocus is linear in the
    # effective depth z + shift(r), so the min-total-defocus focus is the MEDIAN
    # effective depth. With curvature, off-axis cells read deeper, so auto sits
    # deeper than the plain median z -- and falls back to median z without optics.
    acq = _acq(n_px=200, optics=Optics(magnification=8.0, field_curvature_radius_um=600.0))
    px = acq.pixel_size_um
    axis = (100 * px, 100 * px)  # canvas center (n_px=200) in µm
    radii = (0.0, 20.0, 40.0, 60.0, 80.0)
    cells = [Cell(center_um=(100.0, axis[0], axis[1] + r)) for r in radii]
    expected = float(np.median([100.0 + acq.optics.focal_curvature_shift_um(r) for r in radii]))

    assert resolve_focal_plane(cells, "auto") == 100.0  # no optics -> plain median z
    curv = resolve_focal_plane(cells, "auto", acq.optics, axis)
    assert curv == pytest.approx(expected)
    assert curv > 100.0  # curvature pulls the focus deeper to recover off-axis cells
    flat = Optics(magnification=8.0)  # field_curvature_radius_um=None
    assert resolve_focal_plane(cells, "auto", flat, axis) == 100.0  # flat field -> median z


def test_optics_in_focus_surface_cell_is_barely_degraded():
    # z=0, focal auto -> 0: no scatter, no defocus, atten=1 -> only diffraction
    # blur and the flat NA² collection efficiency dim the footprint.
    acq = _acq(n_px=64)
    collection = acq.optics.collection_efficiency
    scene = Scene.zeros(acq)
    scene.cells.append(_cell_with_footprint(acq, z=0.0))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    cell = scene.cells[0]
    assert cell.in_focus is True
    assert cell.optical_brightness == pytest.approx(collection)
    # the sum-normalized PSF conserves the integral, so the only change is the
    # NA² collection loss: observed integral == planted integral × collection.
    assert cell.footprint_observed.sum() == pytest.approx(
        cell.footprint_planted.sum() * collection, rel=1e-2
    )
    assert cell.detectable is None  # deferred to finalize (Step 6)


def test_optics_deeper_cell_is_broader_and_dimmer():
    # Both in focus (focal == z, defocus 0), so the difference is pure depth:
    # scatter broadens the footprint and attenuation removes light.
    def run(z):
        acq = _acq(n_px=80, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=z)
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
        acq = _acq(n_px=80, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=focal)
        scene = Scene.zeros(acq)
        scene.cells.append(_cell_with_footprint(acq, z=z, radius_um=3.0))
        CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
        sums.append(scene.cells[0].footprint_observed.sum())
    assert sums == pytest.approx([sums[0]] * 3, rel=1e-2)


def test_optics_in_focus_flag_respects_depth_of_field():
    acq = _acq(
        n_px=64,
        optics=Optics(magnification=8.0, depth_of_field_um=15.0),
        focal_depth_in_tissue_um=80.0,
    )
    scene = Scene.zeros(acq)
    for z in (70.0, 80.0, 96.0):  # within DOF, at plane, just outside DOF
        scene.cells.append(_cell_with_footprint(acq, z=z))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    assert [c.in_focus for c in scene.cells] == [True, True, False]


def test_focal_curvature_shift_um():
    flat = Optics(magnification=8.0)  # field_curvature_radius_um defaults to None
    assert flat.focal_curvature_shift_um(250.0) == 0.0  # flat field: no shift
    curved = Optics(magnification=8.0, field_curvature_radius_um=2500.0)
    assert curved.focal_curvature_shift_um(0.0) == 0.0  # on-axis: no shift
    s250 = curved.focal_curvature_shift_um(250.0)
    assert s250 == pytest.approx(250.0**2 / (2 * 2500.0), rel=1e-2)  # ~ r²/2R ≈ 12.5
    assert curved.focal_curvature_shift_um(500.0) > s250  # grows with field radius
    with pytest.raises(ValidationError, match="field_curvature_radius_um"):
        Optics(field_curvature_radius_um=0.0)  # must be > 0 or None


def test_resolved_depth_of_field_um():
    # a numeric value is used as-is
    assert Optics(na=0.3, depth_of_field_um=12.0).resolved_depth_of_field_um == 12.0
    # "auto" (the default) derives the DOF from NA: n·λ/NA², falling as 1/NA²
    o30 = Optics(na=0.30, emission_nm=525.0)  # depth_of_field_um defaults to "auto"
    assert o30.resolved_depth_of_field_um == pytest.approx(1.33 * 0.525 / 0.30**2, rel=1e-6)
    assert Optics(na=0.45).resolved_depth_of_field_um < o30.resolved_depth_of_field_um
    with pytest.raises(ValidationError, match="depth_of_field_um"):
        Optics(depth_of_field_um=0.0)  # must be > 0 or "auto"


def test_field_curvature_blurs_off_axis_cells():
    # Two cells at the same depth (the central focal plane): on-axis vs near the
    # FOV corner. With curvature the corner cell focuses shallower, so it falls
    # out of focus and its peak drops, while the on-axis cell stays sharp.
    npx = 200
    acq = _acq(
        n_px=npx,
        optics=Optics(
            magnification=8.0,
            depth_of_field_um=5.0,
            field_curvature_radius_um=600.0,
        ),
        focal_depth_in_tissue_um=100.0,
    )
    px = acq.pixel_size_um
    z = 100.0

    def cell_at(y_um, x_um):
        fp = neuron_footprint(
            (npx, npx), (y_um / px, x_um / px), acq.um_to_px(4.0), 0.0,
            np.random.default_rng(0),
        )
        return Cell(center_um=(z, y_um, x_um), footprint_planted=fp)

    center = cell_at(npx * px / 2, npx * px / 2)  # on axis (r = 0)
    corner = cell_at(10.0, 10.0)                  # near a corner (large r)
    scene = Scene.zeros(acq)
    scene.cells += [center, corner]
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    assert center.in_focus is True
    assert corner.in_focus is False  # off-axis sagitta pushes it past the DOF
    assert corner.footprint_observed.max() < center.footprint_observed.max()


def test_optics_makes_render_use_the_degraded_footprint():
    # Render a deep cell from its planted footprint, then again after optics:
    # the optically degraded render is dimmer (blurred + attenuated).
    acq = _acq(n_px=40, duration_s=1.0)
    rng = np.random.default_rng(1)
    scene = Scene.zeros(acq)
    PlaceNeurons(
        density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(120.0, 120.0)
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
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
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
        PlaceNeurons(
            density_per_mm3=375000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
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


# --- neuropil (5c) ---------------------------------------------------------


def test_ou_process_is_stationary_and_correlated():
    # Mean ~0, unit variance, and a high lag-1 correlation for a slow tau.
    slow = ou_process(20000, tau_frames=50.0, rng=np.random.default_rng(0))
    assert abs(slow.mean()) < 0.1
    assert slow.std() == pytest.approx(1.0, rel=0.1)
    slow_ac = np.corrcoef(slow[1:], slow[:-1])[0, 1]
    assert slow_ac > 0.9  # a ≈ exp(-1/50) ≈ 0.98
    # A fast tau decorrelates frame-to-frame.
    fast = ou_process(20000, tau_frames=0.2, rng=np.random.default_rng(1))
    assert np.corrcoef(fast[1:], fast[:-1])[0, 1] < slow_ac


def test_neuropil_adds_nonnegative_background():
    acq = _acq(n_px=40, duration_s=1.0)
    scene = Scene.zeros(acq)
    NeuropilStep(Neuropil(amplitude=0.5), acq, np.random.default_rng(0))(scene)
    movie = scene.movie.values
    assert movie.min() >= 0.0  # additive light, never negative
    assert movie.mean() > 0.0  # background was actually added


def test_neuropil_records_smooth_spatial_and_temporal_ground_truth():
    acq = _acq(n_px=40, duration_s=2.0)
    scene = Scene.zeros(acq)
    NeuropilStep(
        Neuropil(n_components=4, spatial_sigma_um=10.0, temporal_tau_s=10.0),
        acq,
        np.random.default_rng(1),
    )(scene)
    spatial = scene.truth.neuropil_spatial
    temporal = scene.truth.neuropil_temporal
    assert spatial.shape == (4, 40, 40)
    assert temporal.shape == (4, acq.n_frames)
    # Spatial fields are non-negative, peak-normalized, and smooth (adjacent-pixel
    # variation well below the field's overall spread — unlike white noise).
    field = spatial[0]
    assert field.min() >= 0.0 and field.max() == pytest.approx(1.0)
    assert np.abs(np.diff(field, axis=0)).mean() < field.std()
    # Temporal envelopes are strictly positive (the lognormal guarantee).
    assert (temporal > 0).all()


def test_neuropil_is_reproducible():
    acq = _acq(n_px=32, duration_s=1.0)
    outs = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        Neuropil().build(acq, np.random.default_rng(3))(scene)
        outs.append(scene.movie.values.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


# --- bleaching (5c) --------------------------------------------------------


def test_bleaching_curve_endpoints_and_monotonic():
    curve = bleaching_curve("mono_exp", final_fraction=0.5, n_frames=100)
    assert curve.shape == (100,)
    assert curve[0] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(0.5)
    assert (np.diff(curve) <= 0).all()  # monotonically decaying


def test_bleaching_scales_every_pixel_by_the_curve():
    acq = _acq(n_px=8, duration_s=2.0)
    scene = Scene.ones(acq)
    BleachingStep(Bleaching(final_fraction=0.6), acq, np.random.default_rng(0))(scene)
    curve = bleaching_curve("mono_exp", 0.6, acq.n_frames)
    # On an all-ones movie the result is exactly the curve, broadcast over pixels.
    np.testing.assert_allclose(scene.movie.values[:, 3, 5], curve)
    np.testing.assert_allclose(scene.truth.bleaching, curve)


def test_bleaching_bi_exp_rejected_at_construction():
    with pytest.raises(ValidationError, match="bi_exp"):
        Bleaching(model="bi_exp")


# --- vignette (5c) ---------------------------------------------------------


def test_vignette_is_radial_and_time_invariant():
    acq = _acq(n_px=51, duration_s=1.0)  # odd -> a clean center pixel at (25, 25)
    scene = Scene.ones(acq)
    VignetteStep(
        Vignette(falloff=0.5, exponent=2.0), acq, np.random.default_rng(0)
    )(scene)
    field = scene.truth.vignette
    assert field.shape == (51, 51)
    assert field[25, 25] == pytest.approx(1.0)  # bright center
    assert field[0, 0] == pytest.approx(0.5)  # farthest corner == falloff
    # On an all-ones movie the movie equals the field, and is identical per frame.
    np.testing.assert_allclose(scene.movie.values[0], field)
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


def test_vignette_center_offset_moves_the_bright_spot():
    acq = _acq(n_px=51, duration_s=0.1)  # 1 µm/px, so +12 µm == +12 px in y
    scene = Scene.ones(acq)
    VignetteStep(
        Vignette(falloff=0.4, center_offset_um=(12.0, 0.0)),
        acq,
        np.random.default_rng(0),
    )(scene)
    peak_row, peak_col = np.unravel_index(scene.truth.vignette.argmax(), (51, 51))
    assert peak_row > 25  # shifted down from the FOV center row
    assert peak_col == pytest.approx(25, abs=1)  # unshifted in x


# --- leakage (5c) ----------------------------------------------------------


def test_leakage_uniform_adds_level_everywhere():
    acq = _acq(n_px=16, duration_s=1.0)
    scene = Scene.zeros(acq)
    LeakageStep(
        Leakage(profile="uniform", level=0.2), acq, np.random.default_rng(0)
    )(scene)
    np.testing.assert_allclose(scene.movie.values, 0.2)
    np.testing.assert_allclose(scene.truth.leakage, 0.2)
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


def test_leakage_gaussian_peaks_at_center():
    acq = _acq(n_px=51, duration_s=0.1)
    scene = Scene.zeros(acq)
    LeakageStep(
        Leakage(profile="gaussian", level=0.3, sigma_um=10.0),
        acq,
        np.random.default_rng(0),
    )(scene)
    field = scene.truth.leakage
    assert field[25, 25] == pytest.approx(0.3)  # central glow == level
    assert field[0, 0] < field[25, 25]  # dimmer at the corner
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


# --- vasculature placeholder (5c) ------------------------------------------


def test_vasculature_is_an_honest_noop():
    acq = _acq(n_px=16, duration_s=0.5)
    scene = Scene.ones(acq)
    Vasculature().build(acq, np.random.default_rng(0))(scene)
    assert (scene.movie.values == 1.0).all()  # scene untouched
    assert scene.truth.neuropil_spatial is None  # no ground-truth contribution


# --- scene-grid plumbing (5d-1 refactor) -----------------------------------


def _oversized_scene(acq, canvas):
    """A scene whose movie canvas is larger than the sensor — the shape a motion
    margin (5d-2) will produce. Built by hand here, before Scene grows margin
    support, to prove the steps read their grid from the scene, not the sensor."""
    movie = xr.DataArray(
        np.zeros((acq.n_frames, canvas, canvas)),
        dims=("frame", "height", "width"),
    )
    return Scene(acq=acq, rng=np.random.default_rng(0), movie=movie)


def test_steps_fill_the_scene_canvas_not_the_sensor_dims():
    # Sensor is 20×20 but the canvas is 30×30; place_neurons and neuropil must
    # honor the canvas (so off-FOV tissue exists to move in under motion).
    acq = _acq(n_px=20, duration_s=1.0)
    scene = _oversized_scene(acq, canvas=30)

    PlaceNeurons(
        density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(0))(scene)
    assert scene.cells, "expected cells placed across the larger canvas"
    assert all(c.footprint_planted.shape == (30, 30) for c in scene.cells)

    NeuropilStep(Neuropil(), acq, np.random.default_rng(0))(scene)
    assert scene.truth.neuropil_spatial.shape[1:] == (30, 30)


# --- the field chain -------------------------------------------------------


def test_field_chain_runs_end_to_end_and_records_ground_truth():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)
    rng = np.random.default_rng(7)
    scene = Scene.zeros(acq)
    steps = [
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Neuropil(amplitude=0.3),
        Bleaching(final_fraction=0.7),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    movie = scene.movie.values
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.var() > 0.0
    # Every field step left its ground-truth contribution.
    assert scene.truth.neuropil_spatial is not None
    assert scene.truth.bleaching is not None
    assert scene.truth.vignette is not None
    assert scene.truth.leakage is not None


# --- brain_motion (5d) -----------------------------------------------------


def test_bounded_random_walk_starts_at_zero_and_stays_bounded():
    walk = bounded_random_walk(500, step_px=1.0, max_px=4.0, rng=np.random.default_rng(0))
    assert walk.shape == (500, 2)
    np.testing.assert_array_equal(walk[0], [0.0, 0.0])  # frame 0 is the reference
    mags = np.hypot(walk[:, 0], walk[:, 1])
    assert mags.max() <= 4.0 + 1e-9  # never leaves the radius-max_px disk


def test_shift_and_crop_recenters_a_padded_frame():
    # A 4 px frame padded to 8 px, shifted back by the margin, crops to the FOV.
    canvas = np.zeros((1, 8, 8))
    canvas[0, 5, 5] = 1.0
    out = shift_and_crop(canvas, np.array([[-2.0, -2.0]]), fov_shape=(4, 4))
    assert out.shape == (1, 4, 4)
    assert out[0, 1, 1] == pytest.approx(1.0)  # (5,5) - margin(2) - shift(2) -> (1,1)


def test_explicit_trajectory_moves_content_by_the_given_shift():
    acq = _acq(n_px=20, duration_s=0.1)  # 2 frames, 1 µm/px
    margin = 6
    scene = Scene.zeros(acq, margin_px=margin)
    c = (20 + 2 * margin) // 2  # canvas center == FOV center
    scene.movie.values[:, c, c] = 9.0
    BrainMotion(trajectory_um=[(0.0, 0.0), (3.0, 2.0)]).build(
        acq, np.random.default_rng(0)
    )(scene)
    fov0, fov1 = scene.movie.values[0], scene.movie.values[1]
    fc = 10  # FOV center (20×20)
    assert fov0[fc, fc] == pytest.approx(9.0)  # frame 0 unshifted
    assert fov1[fc + 3, fc + 2] == pytest.approx(9.0)  # frame 1 moved down 3, right 2
    assert fov1[fc, fc] < 1e-6


def test_motion_brings_offscreen_tissue_into_view():
    # A bright spot in the top margin (outside the FOV) is cropped away at rest
    # and brought into view by a downward shift — real tissue, not a fill.
    acq = _acq(n_px=20, duration_s=0.1)  # 2 frames
    margin = 6
    scene = Scene.zeros(acq, margin_px=margin)
    scene.movie.values[:, margin - 3, margin + 10] = 5.0  # 3 px above the FOV top
    BrainMotion(trajectory_um=[(0.0, 0.0), (4.0, 0.0)]).build(
        acq, np.random.default_rng(0)
    )(scene)
    assert scene.movie.values[0].max() < 1e-6  # off-FOV at rest
    assert scene.movie.values[1].max() == pytest.approx(5.0)  # shifted into view


def test_brain_motion_records_shifts_in_pixels():
    acq = _acq(n_px=16, duration_s=0.5)  # 10 frames
    scene = Scene.zeros(acq, margin_px=6)
    BrainMotion(walk_step_um=0.5, max_shift_um=4.0).build(
        acq, np.random.default_rng(2)
    )(scene)
    shifts = scene.truth.shifts
    assert shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(shifts[0], [0.0, 0.0])
    assert np.hypot(shifts[:, 0], shifts[:, 1]).max() <= 4.0 + 1e-9  # px, bounded


def test_static_field_is_invariant_under_motion():
    # vignette is fixed to the sensor: the field it writes is byte-identical
    # whether or not the brain moved first (the reference-frame invariant).
    acq = _acq(n_px=24, duration_s=0.5)
    s0 = Scene.ones(acq)
    VignetteStep(Vignette(falloff=0.5), acq, np.random.default_rng(0))(s0)

    s1 = Scene.ones(acq, margin_px=4)
    BrainMotion(walk_step_um=0.5, max_shift_um=3.0).build(
        acq, np.random.default_rng(1)
    )(s1)
    VignetteStep(Vignette(falloff=0.5), acq, np.random.default_rng(2))(s1)

    assert s1.truth.vignette.shape == (24, 24)  # cropped to the sensor FOV
    np.testing.assert_array_equal(s0.truth.vignette, s1.truth.vignette)


def test_brain_motion_rejects_wrong_length_trajectory():
    acq = _acq(n_px=16, duration_s=0.5)  # 10 frames
    scene = Scene.zeros(acq, margin_px=4)
    with pytest.raises(ValueError, match="trajectory_um"):
        BrainMotion(trajectory_um=[(0.0, 0.0), (1.0, 1.0)]).build(
            acq, np.random.default_rng(0)
        )(scene)


def test_brain_motion_rejects_insufficient_margin():
    acq = _acq(n_px=16, duration_s=0.1)  # 2 frames
    scene = Scene.zeros(acq, margin_px=1)  # only 1 px of tissue margin
    with pytest.raises(ValueError, match="margin"):
        BrainMotion(trajectory_um=[(0.0, 0.0), (5.0, 0.0)]).build(
            acq, np.random.default_rng(0)
        )(scene)  # 5 px shift overruns the 1 px margin


def test_full_pipeline_with_motion_runs_end_to_end():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)  # 1 µm/px
    rng = np.random.default_rng(11)
    max_shift_um = 4.0
    margin = int(np.ceil(acq.um_to_px(max_shift_um))) + 1
    scene = Scene.zeros(acq, margin_px=margin)
    steps = [
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Neuropil(amplitude=0.3),
        Bleaching(final_fraction=0.7),
        BrainMotion(walk_step_um=0.4, max_shift_um=max_shift_um),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    movie = scene.movie.values
    assert movie.shape == (acq.n_frames, 40, 40)  # cropped back to the sensor FOV
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.var() > 0.0
    assert scene.truth.shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(scene.truth.shifts[0], [0.0, 0.0])
