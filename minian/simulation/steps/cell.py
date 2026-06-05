"""Cell-domain steps: place neurons, then give them calcium activity.

These are the first two steps of the forward pipeline — pure biology, before any
optics or sensor effect:

* :class:`PlaceNeuronsStep` positions neurons in a 3-D µm volume and stamps
  a sharp, pre-optics footprint for each. The *distribution* half (sampling
  centers only, no footprints) is factored into :func:`sample_neurons` so it can
  be reused cheaply for teaching/visualization at full FOV.
* :class:`CellActivityStep` gives every soma a calcium trace built from a
  2-state Markov spike model convolved with a double-exponential indicator
  kernel, plus a per-cell brightness gain that scales the trace.

Both only fill per-cell records on the scene (``scene.cells``); nothing is drawn
into the movie until ``render`` (:mod:`minian.simulation.steps.tissue`). The
optical degradation that turns the *planted* (sharp) footprint into the
*observed* (blurred, attenuated) one is the next step, ``optics`` (migration
Step 5b); until it lands, ``render`` composites the planted footprint directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from minian.simulation.scene import Cell, Scene
from minian.simulation.steps.base import Step

if TYPE_CHECKING:
    from minian.simulation.spec import PlaceNeurons

# Guards the noise normalization for a degenerate (flat) low-pass field; far
# below any physically meaningful intensity.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# place_neurons
# ---------------------------------------------------------------------------


# Proximal-dendrite rendering constants (cytosolic morphology only). Dendrites
# are graded dimmer than the soma and taper to a thread, so blur, defocus, and
# the sensor noise floor erase them first — the "we lose thin features fast"
# lesson falls out of the physics for free.
_DENDRITE_BASE_INTENSITY = 0.6  # planted weight where a dendrite leaves the soma
_DENDRITE_TIP_INTENSITY = 0.15  # ...tapering to this at the distal tip
_DENDRITE_TIP_WIDTH_PX = 0.75  # minimum stamp radius, keeps the thread continuous
_DENDRITE_WANDER_RAD = 0.15  # per-step heading random walk, radians (gently wavy, not curly)
_DENDRITE_ANGLE_JITTER_RAD = 0.4  # jitter on the evenly spaced launch angles


def neuron_footprint(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    radius_px: float,
    irregularity: float,
    rng: np.random.Generator,
    *,
    morphology: str = "soma",
    n_dendrites: int = 0,
    dendrite_length_px: float = 0.0,
    dendrite_width_px: float = 0.0,
) -> np.ndarray:
    """A sharp, peak-normalized neuron footprint — the *planted* spatial weight A.

    Models the cell's true (pre-optics) fluorophore support, with **no optical
    blur** — diffraction/defocus/scatter are applied later by the ``optics`` step.
    It is peak-normalized (``max == 1`` at the soma) so a cell's brightness is
    carried entirely by its calcium trace, not baked into the footprint.

    Two GCaMP targeting variants are supported via ``morphology``:

    * ``"soma"`` — soma-targeted GCaMP (e.g. SomaGCaMP / riboGCaMP): a single
      filled, possibly lumpy disk, the soma body only.
    * ``"cytosolic"`` — standard cytosolic GCaMP (GCaMP6/7/8…): the same soma
      disk plus ``n_dendrites`` tapering proximal dendrites. The dendrites are
      *graded* (dimmer than the soma) and *thin*, so they are exactly what
      diffraction, defocus, scatter, and the sensor noise floor erase first — a
      faithful demonstration of how quickly fine neurites become unresolvable.

    The soma is **identical** in both variants; ``"cytosolic"`` only *adds*
    dendrites after the soma is drawn, so ``"soma"`` (the default) reproduces the
    soma-only footprint bit-for-bit.

    ``irregularity`` ∈ [0, 1] warps the soma boundary: at ``0`` it is a clean
    disk; above ``0`` the per-pixel radius is modulated by a low-pass-filtered
    noise field (smoothed on the soma's own length scale), giving a lumpy outline
    that is more soma-like than a perfect circle while staying coarser than the
    optics will later blur away. Typical cortical somata are ~5–10 µm radius.

    Note — the shape is *physical*, the grid is *sampling*. This routine
    rasterizes a continuous µm-space shape onto whatever pixel grid the caller
    passes (via ``shape`` and the ``*_px`` arguments). The cell's true geometry is
    intrinsic and independent of pixel size; only how finely it is sampled depends
    on the sensor. In the normal pipeline the planted footprint is rasterized once
    at the sensor's own scale, which is fine because the result is then blurred by
    the (coarser) optics. But a caller that needs the *same* cell across multiple
    pixel sizes should generate it once on a fixed fine grid and resample, rather
    than re-rasterizing per grid — re-rasterizing re-draws the ``rng`` noise field
    at the new size and so changes the lumpy outline. This costs nothing in
    fidelity: a 1-photon miniscope is pixel-limited, never diffraction-limited
    (see :meth:`Optics.diffraction_sigma_um`), so a sub-pixel reference grid holds
    more detail than the optics can ever resolve.
    """
    h, w = shape
    cy, cx = center_px
    footprint = np.zeros((h, w), dtype=float)
    # The soma is local: it reaches at most radius_px·(1 + irregularity) from the
    # center (the +irregularity headroom covers the lumpy-boundary wobble). Compute
    # the hypot/threshold/noise only inside that bounding box and write it into the
    # full canvas — bit-for-bit the same as filling the whole grid wherever the soma
    # is, just far cheaper than touching every pixel for a cell that occupies a tiny
    # patch. Dendrites are stamped afterwards and self-window (see :func:`_stamp_disk`).
    reach = radius_px * (1.0 + max(irregularity, 0.0)) + 1.0
    y0 = max(int(math.floor(cy - reach)), 0)
    y1 = min(int(math.ceil(cy + reach)) + 1, h)
    x0 = max(int(math.floor(cx - reach)), 0)
    x1 = min(int(math.ceil(cx + reach)) + 1, w)
    if y0 < y1 and x0 < x1:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(yy - cy, xx - cx)
        if irregularity > 0:
            # Low-pass noise on the soma's own scale → a smoothly lumpy boundary,
            # normalized to ~[-1, 1] so `irregularity` is the fractional radius wobble.
            noise = gaussian_filter(
                rng.standard_normal((y1 - y0, x1 - x0)),
                sigma=max(radius_px / 2.0, 1.0),
            )
            noise /= max(noise.max(), -noise.min()) + _EPS  # scale to ~[-1, 1]
            r_eff = radius_px * (1.0 + irregularity * noise)
        else:
            r_eff = radius_px
        # A 0/1 membership mask is already peak-normalized (max == 1) by construction.
        footprint[y0:y1, x0:x1] = (dist <= r_eff).astype(float)
    # Cytosolic GCaMP fills the proximal dendrites too. Stamp them *after* the
    # soma so the soma's RNG draw above is untouched — "soma" stays bit-identical.
    if morphology == "cytosolic" and n_dendrites > 0 and dendrite_length_px > 0:
        _stamp_dendrites(
            footprint,
            cy,
            cx,
            radius_px,
            n_dendrites,
            dendrite_length_px,
            dendrite_width_px,
            rng,
        )
    if not footprint.any():
        # Sub-pixel soma: keep at least the nearest pixel lit so the cell is
        # never silently empty.
        iy = int(np.clip(round(cy), 0, h - 1))
        ix = int(np.clip(round(cx), 0, w - 1))
        footprint[iy, ix] = 1.0
    return footprint


def _stamp_disk(
    footprint: np.ndarray, y: float, x: float, radius_px: float, intensity: float
) -> None:
    """Paint one filled disk into ``footprint`` via ``maximum`` (overlaps never sum).

    Works only on the disk's local bounding box, so laying down a dendrite is
    cheap regardless of canvas size. ``maximum`` keeps the soma's peak at 1 where
    a dendrite overlaps it, preserving peak-normalization.
    """
    h, w = footprint.shape
    y0 = max(int(np.floor(y - radius_px)), 0)
    y1 = min(int(np.ceil(y + radius_px)) + 1, h)
    x0 = max(int(np.floor(x - radius_px)), 0)
    x1 = min(int(np.ceil(x + radius_px)) + 1, w)
    if y0 >= y1 or x0 >= x1:
        return  # disk fell entirely off the canvas
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((yy - y) ** 2 + (xx - x) ** 2 <= radius_px**2) * intensity
    sub = footprint[y0:y1, x0:x1]
    np.maximum(sub, disk, out=sub)


def _stamp_dendrites(
    footprint: np.ndarray,
    cy: float,
    cx: float,
    radius_px: float,
    n_dendrites: int,
    length_px: float,
    width_px: float,
    rng: np.random.Generator,
) -> None:
    """Grow ``n_dendrites`` tapering proximal dendrites out of the soma.

    Each dendrite launches from just inside the soma edge at a roughly evenly
    spaced (then jittered) angle and walks outward in ~1 px steps with a small
    per-step heading wobble, so it curves gently rather than spiking out straight.
    Both its width and its intensity taper from base to tip; it is laid down as a
    chain of overlapping disks (:func:`_stamp_disk`), so it stays continuous.
    """
    # Roughly even angular spread, then jittered, so dendrites don't all clump.
    base = rng.uniform(0.0, 2.0 * np.pi)
    angles = base + np.arange(n_dendrites) * (2.0 * np.pi / n_dendrites)
    angles = angles + rng.normal(0.0, _DENDRITE_ANGLE_JITTER_RAD, size=n_dendrites)
    n_steps = max(int(round(length_px)), 2)
    for theta in angles:
        # Start just inside the soma so the dendrite connects without a gap.
        y = cy + 0.8 * radius_px * np.sin(theta)
        x = cx + 0.8 * radius_px * np.cos(theta)
        heading = theta
        for i in range(n_steps):
            frac = i / (n_steps - 1)  # 0 at the soma .. 1 at the tip
            # width_px is a diameter; stamp radius is half of it, tapering to a thread
            rad = max(0.5 * width_px * (1.0 - frac), _DENDRITE_TIP_WIDTH_PX)
            intensity = _DENDRITE_BASE_INTENSITY + frac * (
                _DENDRITE_TIP_INTENSITY - _DENDRITE_BASE_INTENSITY
            )
            _stamp_disk(footprint, y, x, rad, intensity)
            heading += rng.normal(0.0, _DENDRITE_WANDER_RAD)
            y += np.sin(heading)
            x += np.cos(heading)


def sample_neurons(
    spec: PlaceNeurons,
    fov_h_um: float,
    fov_w_um: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Sample the soma *distribution* for a ``PlaceNeurons`` spec over a FOV.

    This is the half of ``place_neurons`` that decides **where cells go** — with no
    footprint stamping, so it is cheap even at a full sensor FOV (the per-cell
    :func:`neuron_footprint` paints are the expensive part). :class:`PlaceNeuronsStep`
    calls this and then stamps a footprint per returned center; teaching code (the
    anatomy notebook's placement widget) can call it directly to show the population
    layout without paying for footprints.

    Placement is **purely spatial**: brightness is no longer drawn here. Per-cell
    response gain (how bright a cell is) is biology that belongs to the calcium
    response, so it is drawn in :class:`CellActivityStep` (``brightness_cv``); SNR is
    an emergent measurement property and is not an input anywhere.

    The count is **volumetric**: ``round(density_per_mm3 · area_mm2 · thickness)``,
    where the slab thickness is ``depth_range_um`` width **floored at one soma
    diameter** (``2 · soma_radius_um``) so a thin — or strictly planar
    (``lo == hi``) — layer still yields cells rather than zero. A thicker slab
    therefore holds proportionally more cells, the physical behavior. Centers are
    drawn uniformly in ``(y, x)`` across the FOV and in ``z`` across
    ``depth_range_um``; if ``min_distance_um > 0`` they are rejection-sampled
    (Poisson-disk style) to a 3-D center-to-center minimum.

    Returns ``centers``: a list of ``(z, y, x)`` µm tuples.
    """
    area_mm2 = (fov_h_um / 1000.0) * (fov_w_um / 1000.0)
    lo, hi = spec.depth_range_um
    thickness_um = max(hi - lo, 2.0 * spec.soma_radius_um)  # floor: one soma diameter
    count = round(spec.density_per_mm3 * area_mm2 * (thickness_um / 1000.0))
    return _sample_centers(
        count, fov_h_um, fov_w_um, spec.depth_range_um, spec.min_distance_um, rng
    )


def _sample_centers(
    count: int,
    fov_h_um: float,
    fov_w_um: float,
    depth_range_um: tuple[float, float],
    min_distance_um: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    z_lo, z_hi = depth_range_um

    def draw() -> tuple[float, float, float]:
        return (
            rng.uniform(z_lo, z_hi),
            rng.uniform(0.0, fov_h_um),
            rng.uniform(0.0, fov_w_um),
        )

    if min_distance_um <= 0:
        return [draw() for _ in range(count)]

    # Poisson-disk-style rejection sampling. Capped so an over-dense request
    # ends with fewer cells rather than looping forever (an honest outcome:
    # you cannot pack more than the minimum spacing allows).
    centers: list[tuple[float, float, float]] = []
    attempts, max_attempts = 0, max(1000, 100 * count)
    while len(centers) < count and attempts < max_attempts:
        attempts += 1
        cand = draw()
        if all(math.dist(cand, c) >= min_distance_um for c in centers):
            centers.append(cand)
    return centers


def _sample_brightness(cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-cell expression/response gain: lognormal, **mean 1**, given CV.

    ``cv == 0`` makes every cell equally bright (gain 1). Otherwise the gain is
    lognormal with mean exactly 1 (so changing the spread does not change the
    population's total light budget) and a right tail, the familiar "a few bright
    cells over a dimmer majority".
    """
    if n == 0:
        return np.empty(0)
    if cv <= 0:
        return np.ones(n)
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma  # makes E[gain] == 1
    return np.exp(rng.normal(mu, sigma, size=n))


class PlaceNeuronsStep(Step):
    """Position neurons in a 3-D µm volume and stamp a planted footprint each.

    Placement (*where cells go*) is delegated to :func:`sample_neurons`; this step
    adds the part that function omits — stamping a peak-normalized planted footprint
    (:func:`neuron_footprint`, soma-only or soma + proximal dendrites per
    ``spec.morphology``) at each sampled center. Placement is purely spatial now:
    per-cell brightness is drawn later in ``cell_activity``, not here.

    The cell count is volumetric — ``round(density_per_mm3 · canvas_area_mm2 ·
    thickness)`` over the **canvas** area (the scene movie's grid, which a motion
    margin may enlarge beyond the sensor FOV) and the ``depth_range_um`` thickness
    (floored at one soma diameter); see :func:`sample_neurons`. The per-cell depth
    is consumed later by the ``optics`` step (5b) for the ``in_focus`` /
    ``detectable`` flags.
    """

    name = "place_neurons"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        acq, rng = self.acq, self.rng
        # Fill whatever canvas the scene movie defines, not the bare sensor: a
        # motion margin (Step 5d) enlarges the canvas beyond the sensor FOV so
        # that real, simulated tissue moves into view at the edges. At margin 0
        # the canvas equals the sensor FOV. Cell positions are in canvas/tissue
        # coordinates (origin = canvas top-left); the FOV crop offset is applied
        # at finalize (Step 6).
        shape = scene.movie.values.shape[1:]  # (height, width) of the canvas
        fov_h_um = shape[0] * acq.pixel_size_um
        fov_w_um = shape[1] * acq.pixel_size_um
        radius_px = acq.um_to_px(spec.soma_radius_um)
        dendrite_length_px = acq.um_to_px(spec.dendrite_length_um)
        dendrite_width_px = acq.um_to_px(spec.dendrite_width_um)

        centers = sample_neurons(spec, fov_h_um, fov_w_um, rng)
        for (z, y, x) in centers:
            footprint = neuron_footprint(
                shape,
                (acq.um_to_px(y), acq.um_to_px(x)),
                radius_px,
                spec.irregularity,
                rng,
                morphology=spec.morphology,
                n_dendrites=spec.n_dendrites,
                dendrite_length_px=dendrite_length_px,
                dendrite_width_px=dendrite_width_px,
            )
            scene.cells.append(
                Cell(center_um=(z, y, x), footprint_planted=footprint)
            )


# ---------------------------------------------------------------------------
# cell_activity
# ---------------------------------------------------------------------------


def calcium_kernel(tau_rise_s: float, tau_decay_s: float, fps: float) -> np.ndarray:
    """Double-exponential calcium-indicator kernel, sampled at the frame rate.

    ``k(t) = exp(-t/τ_decay) − exp(-t/τ_rise)`` — the canonical CaLab-style
    impulse response of a fluorescent calcium indicator: a fast rise (``τ_rise``)
    onto a slow decay (``τ_decay``). Sampled at ``1/fps`` intervals out to
    ``5·τ_decay`` (where the response has decayed to <1%) and peak-normalized, so
    convolving it with an amplitude-weighted spike train yields a ΔF trace whose
    per-spike height is the spike amplitude. Requires ``τ_rise < τ_decay`` (a
    rise slower than the decay is not a physical indicator response). Typical
    GCaMP: ``τ_rise`` ~0.05 s, ``τ_decay`` ~0.3–0.7 s.
    """
    if tau_rise_s >= tau_decay_s:
        raise ValueError(
            f"tau_rise_s ({tau_rise_s}) must be < tau_decay_s ({tau_decay_s}) "
            "for a double-exponential indicator kernel."
        )
    length = max(int(np.ceil(tau_decay_s * 5.0 * fps)), 2)
    t = np.arange(length) / fps
    k = np.exp(-t / tau_decay_s) - np.exp(-t / tau_rise_s)
    return k / k.max()


# --- kernel timing: (time-to-peak, FWHM) <-> (tau_rise, tau_decay) ----------
#
# The double-exp kernel g(t) = e^{-t/τd} - e^{-t/τr} has a useful scale property:
# write r = τr/τd and t = τd·u, and g(τd·u) = e^{-u} - e^{-u/r} depends on r alone.
# So both the time-to-peak and the FWHM are τd times a function of r only, and their
# *ratio* depends only on r. That turns the two-feature inversion into a single
# robust 1-D root-find on r, then τd = t_peak / peak_u(r) and τr = r·τd. Users think
# in observable kernel features (how fast it rises, how wide it is); the simulator
# keeps physical time constants. These two helpers bridge the two.


def _peak_u(r: float) -> float:
    """Location (in u = t/τd units) of the kernel peak, for r = τr/τd ∈ (0, 1)."""
    return r * math.log(r) / (r - 1.0)


def _shape_u(u: float, r: float) -> float:
    return math.exp(-u) - math.exp(-u / r)


def _half_max_u(r: float) -> tuple[float, float]:
    """The two u where the kernel is half its peak (u_lo < u_peak < u_hi)."""
    from scipy.optimize import brentq

    up = _peak_u(r)
    half = _shape_u(up, r) / 2.0
    u_lo = brentq(lambda u: _shape_u(u, r) - half, 1e-12, up)
    hi = up
    while _shape_u(hi, r) > half:  # grow until the decay falls below half max
        hi *= 2.0
    u_hi = brentq(lambda u: _shape_u(u, r) - half, up, hi)
    return u_lo, u_hi


def kernel_timing(tau_rise_s: float, tau_decay_s: float) -> tuple[float, float]:
    """``(time_to_peak_s, fwhm_s)`` of the double-exp kernel for given time constants.

    The forward direction of :func:`tau_from_kernel_timing`. Requires
    ``tau_rise_s < tau_decay_s``.
    """
    if tau_rise_s >= tau_decay_s:
        raise ValueError(
            f"tau_rise_s ({tau_rise_s}) must be < tau_decay_s ({tau_decay_s})."
        )
    r = tau_rise_s / tau_decay_s
    u_lo, u_hi = _half_max_u(r)
    return _peak_u(r) * tau_decay_s, (u_hi - u_lo) * tau_decay_s


# Achievable ratio range t_peak/FWHM, over r in (0, 1): -> 0 as r -> 0 (instant rise,
# FWHM -> ln2·τd), -> ~0.409 as r -> 1 (the alpha-function limit). Targets outside
# this are physically impossible for this kernel family, so we clamp into it.
_R_LO, _R_HI = 1e-4, 1.0 - 1e-6


def tau_from_kernel_timing(t_peak_s: float, fwhm_s: float) -> tuple[float, float]:
    """Invert observable kernel features to ``(tau_rise_s, tau_decay_s)``.

    ``t_peak_s`` is the time from a spike to the kernel's peak; ``fwhm_s`` is the
    kernel's full width at half maximum. Because the peak/FWHM ratio depends only on
    ``r = tau_rise/tau_decay``, this solves a single 1-D root-find for ``r`` and then
    recovers ``tau_decay = t_peak / peak_u(r)`` and ``tau_rise = r·tau_decay``. A
    ratio outside the kernel's achievable range (roughly ``0 < t_peak/fwhm < 0.41``)
    is clamped to the nearest feasible shape.
    """
    from scipy.optimize import brentq

    def ratio(r: float) -> float:
        u_lo, u_hi = _half_max_u(r)
        return _peak_u(r) / (u_hi - u_lo)

    rho = t_peak_s / fwhm_s
    lo, hi = ratio(_R_LO), ratio(_R_HI)
    if rho <= lo:
        r = _R_LO
    elif rho >= hi:
        r = _R_HI
    else:
        r = brentq(lambda rr: ratio(rr) - rho, _R_LO, _R_HI)
    tau_decay = t_peak_s / _peak_u(r)
    return r * tau_decay, tau_decay


# CaLab's SPIKE_ACTIVITY_LEVELS (sparse, moderate, dense), expressed in our units:
# (p_quiescent_to_active /frame, p_active_to_quiescent /frame, active_rate_hz,
# quiescent_rate_hz). The Hz values are CaLab's per-bin spike probs x its 300 Hz sim
# rate (0.3/0.5/0.7 -> 90/150/210; 0.001/0.002/0.005 -> 0.3/0.6/1.5). Going denser
# couples three things: bursts start more often (p_q2a up), last longer (p_a2q down),
# and fire harder (active_rate up) -- the path that keeps the look realistic.
# Attribution: CaLab web simulator (simulation-quality-presets.ts).
_ACTIVITY_SPARSE = (0.002, 0.4, 90.0, 0.3)
_ACTIVITY_MODERATE = (0.005, 0.3, 150.0, 0.6)
_ACTIVITY_DENSE = (0.01, 0.2, 210.0, 1.5)


def spike_activity_params(activity: float) -> tuple[float, float, float, float]:
    """Map a single ``activity`` ∈ [0, 1] onto CaLab's sparse→moderate→dense path.

    Returns ``(p_quiescent_to_active, p_active_to_quiescent, active_rate_hz,
    quiescent_rate_hz)`` for :class:`~minian.simulation.spec.CellActivity`. ``0`` is
    sparse, ``0.5`` is moderate (CaLab's default, the screenshot regime), ``1`` is
    dense; values interpolate piecewise-linearly through CaLab's three
    ``SPIKE_ACTIVITY_LEVELS`` and are clamped to ``[0, 1]``.

    This is the whole spike-activity control: CaLab moves all four Markov parameters
    together along this single density axis (it has no separate rate/burstiness
    knobs), and that coupling is what keeps firing realistic, dense bursts plus a
    little background, with no dead stretches, across the entire range.
    """
    a = min(max(activity, 0.0), 1.0)
    if a <= 0.5:
        t, lo, hi = a / 0.5, _ACTIVITY_SPARSE, _ACTIVITY_MODERATE
    else:
        t, lo, hi = (a - 0.5) / 0.5, _ACTIVITY_MODERATE, _ACTIVITY_DENSE
    return tuple(l + t * (h - l) for l, h in zip(lo, hi))


class CellActivityStep(Step):
    """Give each soma a calcium trace, the CaLab way: 300 Hz spikes → kernel → bin.

    Spikes are generated on a **high-resolution grid** (``spike_sim_hz``, ~300 Hz)
    rather than at the camera frame rate, then convolved with the calcium kernel at
    that fine rate and **bin-averaged down** to the imaging rate (which is what the
    camera's exposure integration physically does). Two payoffs: one spike per fine
    bin is at most ~3 ms apart, so the **refractory period** is respected and bursts
    cannot pack unphysically tight; and sub-frame spike timing survives the kernel,
    which matters for fast indicators. (Attribution: this matches the CaLab web
    simulator's spike model.)

    Per cell, a 2-state Markov gate (quiescent ↔ active) modulates the per-bin spike
    probability between ``quiescent_rate_hz`` and ``active_rate_hz`` (each ÷
    ``spike_sim_hz``). Bursting comes from a **high in-active rate** concentrated into
    short active bouts, not from a high mean rate; :func:`bursty_spike_params` maps a
    target mean rate + burstiness onto the gate. The gate itself is stepped at the
    frame rate (bouts are ~seconds, so sub-frame onset timing is irrelevant once we
    bin); the spikes it gates are the part that runs at ``spike_sim_hz``.

    Amplitude is biology and enters as a single **per-cell** expression/response gain
    (lognormal spread ``brightness_cv``, mean 1) that scales each cell's *whole* trace
    — baseline and transients together, so a bright cell is brighter everywhere. No
    measurement noise is added here: the trace is the clean ground-truth ``C``;
    shot/read noise enter at the ``sensor`` and background at ``neuropil``, so SNR is
    emergent, never set here.

    Writes ``cell.trace`` (the calcium trace ``C``), ``cell.spikes`` (the per-frame
    spike *count* train ``S`` — the fine 300 Hz train is binned away, since nothing
    downstream recovers spikes faster than the frame rate), and ``cell.amplitude``
    (the per-cell gain); these are the ideal deconvolution targets in ground truth.
    """

    name = "cell_activity"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        n_frames, fps = self.acq.n_frames, self.acq.fps
        bins = max(int(round(spec.spike_sim_hz / fps)), 1)  # fine bins per frame
        hr_fps = bins * fps  # realized high-res rate (integer multiple of fps)
        kernel = calcium_kernel(spec.tau_rise_s, spec.tau_decay_s, hr_fps)
        gains = _sample_brightness(spec.brightness_cv, len(scene.cells), self.rng)
        for cell, gain in zip(scene.cells, gains):
            fine = self._fine_spikes(spec, n_frames, fps, bins, self.rng)
            calcium = fftconvolve(fine, kernel)[: fine.size]
            # Bin-average to the imaging rate (camera exposure integration); the
            # per-cell gain scales the whole trace (baseline + transients), so a
            # bright cell emits more light everywhere -- a higher emergent SNR later.
            clean = calcium.reshape(n_frames, bins).mean(axis=1)
            trace = gain * (spec.f0 + clean)
            if spec.trace_noise > 0:
                trace = trace + self.rng.normal(0.0, spec.trace_noise, size=n_frames)
            cell.trace = trace
            cell.spikes = fine.reshape(n_frames, bins).sum(axis=1)  # per-frame counts
            cell.amplitude = float(gain)

    @staticmethod
    def _fine_spikes(
        spec, n_frames: int, fps: float, bins: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Binary spike train on the high-res grid (length ``n_frames * bins``).

        The 2-state gate is stepped once per frame (sequential, O(n_frames)); spikes
        are then drawn per fine bin as a Bernoulli at ``rate / hr_fps``. One spike
        per ~3 ms bin enforces the refractory period for free. The gate starts in its
        **stationary** state (active with prob = the long-run active fraction), so a
        recording does not always open with a quiet stretch.
        """
        state = np.empty(n_frames, dtype=bool)
        p_q2a, p_a2q = spec.p_quiescent_to_active, spec.p_active_to_quiescent
        f_stationary = p_q2a / (p_q2a + p_a2q)
        active = bool(rng.random() < f_stationary)
        for f in range(n_frames):
            if not active:
                if rng.random() < p_q2a:
                    active = True
            elif rng.random() < p_a2q:
                active = False
            state[f] = active
        hr_fps = bins * fps
        rate = np.where(np.repeat(state, bins), spec.active_rate_hz, spec.quiescent_rate_hz)
        p_spike = np.minimum(rate / hr_fps, 1.0)
        return (rng.random(n_frames * bins) < p_spike).astype(float)


# ---------------------------------------------------------------------------
# optics
# ---------------------------------------------------------------------------


def resolve_focal_plane(
    cells: list[Cell],
    focal_depth_in_tissue_um: float | str,
    optics: "Optics | None" = None,
    axis_yx: tuple[float, float] | None = None,
) -> float:
    """Resolve ``Acquisition.focal_depth_in_tissue_um`` to a concrete focal depth, µm.

    A numeric value is the focal depth below the surface as-is. ``"auto"`` puts the
    focal plane where it **minimizes the total defocus blur** over the population.
    Defocus is ``σ = NA·|z − focal_eff|`` with ``focal_eff = focal − shift(r)`` the
    field-curvature-corrected focal depth a cell at field radius ``r`` actually sees
    (off-axis cells focus shallower). That is ``NA·|(z + shift(r)) − focal|`` —
    linear in each cell's **effective depth** ``e = z + shift(r)`` — so the focal
    that minimizes ``Σ σ`` is exactly the **median of the effective depths**. This
    automatically accounts for field curvature: with an un-flattened field the
    off-axis cells read deeper, pulling the plane down so more of them come into
    focus than a naive median-of-``z`` would manage.

    Field curvature is only included when both ``optics`` (with a non-``None``
    ``field_curvature_radius_um``) and ``axis_yx`` (the optical axis in canvas µm)
    are supplied; otherwise ``"auto"`` falls back to the plain median cell depth.
    An empty scene falls back to the surface (``0.0``). This is the one place
    ``"auto"`` becomes concrete; every downstream read sees a number.
    """
    if focal_depth_in_tissue_um != "auto":
        return float(focal_depth_in_tissue_um)
    if not cells:
        return 0.0
    depths = np.array([cell.center_um[0] for cell in cells], dtype=float)
    if optics is not None and axis_yx is not None and optics.field_curvature_radius_um is not None:
        axis_y, axis_x = axis_yx
        shifts = np.array(
            [
                optics.focal_curvature_shift_um(
                    math.hypot(cell.center_um[1] - axis_y, cell.center_um[2] - axis_x)
                )
                for cell in cells
            ],
            dtype=float,
        )
        return float(np.median(depths + shifts))  # median effective depth = min total defocus
    return float(np.median(depths))


_GAUSS_TRUNCATE = 4.0  # scipy gaussian_filter default; the PSF is exactly 0 beyond


def degrade_footprint(
    planted: np.ndarray, sigma_px: float, gain: float
) -> np.ndarray:
    """Apply the optical PSF blur and the multiplicative light-loss to a footprint.

    ``observed = gain · (planted ⊛ Gaussian(sigma_px))``. The Gaussian
    convolution is the combined diffraction + defocus + scatter point-spread; it
    is sum-normalized, so it **conserves integrated intensity** — that is what
    makes *defocus* intensity-conserving (it spreads light: the peak drops but
    the integral is unchanged). ``gain`` is the flat light-loss that actually
    removes signal: scatter ``attenuation(z)`` (depth) × ``collection_efficiency``
    (``∝ NA²``, the objective's light-gathering power). Both are focal-plane
    independent, so the observed footprint's integral is too. ``mode="constant"``
    means light blurred past the FOV edge is lost — physically honest for a cell
    near the boundary.

    A footprint is local (one cell) on a canvas that may be far larger, so the
    blur is computed only within the cell's bounding box, grown by the PSF's
    truncation radius (``4·sigma_px``). Beyond that the Gaussian is exactly zero,
    so the result is **bit-identical** to filtering the whole canvas — just much
    cheaper when the cell is small relative to the frame.
    """
    rows = np.any(planted > 0, axis=1)
    if not rows.any():
        return np.zeros(planted.shape, dtype=float)  # empty footprint → nothing to blur
    cols = np.any(planted > 0, axis=0)
    y0, y1 = int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), len(cols) - int(np.argmax(cols[::-1]))
    pad = int(np.ceil(_GAUSS_TRUNCATE * sigma_px)) + 1
    y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
    y1, x1 = min(y1 + pad, planted.shape[0]), min(x1 + pad, planted.shape[1])
    observed = np.zeros(planted.shape, dtype=float)
    observed[y0:y1, x0:x1] = gain * gaussian_filter(
        planted[y0:y1, x0:x1], sigma=sigma_px, mode="constant"
    )
    return observed


class CellOpticsStep(Step):
    """Degrade each planted footprint by diffraction + defocus(|z−focal|) + scatter(z).

    Reads each cell's depth ``z`` and the physical ``Optics``/``Tissue``
    constants (via :meth:`Acquisition.cell_optics`) — there are no tunable
    fields. For every cell it:

    * writes ``footprint_observed = gain · (planted ⊛ Gaussian(σ_total))`` where
      ``gain = attenuation(z) · collection_efficiency`` — the blurred, dimmed
      footprint CNMF could actually recover;
    * sets ``in_focus`` geometrically (``|z − focal_eff| ≤`` the NA-derived depth
      of field), where ``focal_eff`` includes the field-curvature shift;
    * stores ``optical_brightness`` — the per-cell *peak* scalar from
      ``cell_optics`` (defocus drops the peak as ``σ₀²/σ_total²``; scatter
      ``attenuation(z)`` and ``collection_efficiency ∝ NA²`` dim it). Footprint
      *integral* scales with that same ``gain``, but a cell's *detectability*
      turns on its peak, which defocus also lowers — hence two distinct
      quantities. ``detectable`` itself is left for ``finalize()``
      (Step 6), where this peak combines with the illumination field and the
      sensor noise floor.

    The *central* focal plane is resolved once for the whole scene from
    ``Acquisition.focal_depth_in_tissue_um`` (``"auto"`` → the focus that minimizes
    total defocus, i.e. the median curvature-corrected effective depth; see
    :func:`resolve_focal_plane`). When ``Optics.field_curvature_radius_um`` is set,
    each cell's effective focal depth is that plane minus the field-curvature
    sagitta at its distance from the optical axis (canvas center), so off-axis cells
    focus shallower and blur out toward the edges — the sharp-center/soft-edge look
    of an un-flattened miniscope. Cells without a planted footprint are skipped.
    """

    name = "optics"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        acq = self.acq
        # Optical axis = canvas center (the sensor FOV is a centered crop of the
        # canvas). Off-axis cells focus shallower by the field-curvature sagitta,
        # so each cell sees its own focal depth (no footprint warping: the
        # curvature over one soma is negligible vs the ~mm curvature radius).
        h, w = scene.movie.values.shape[1:]
        axis_y = h * acq.pixel_size_um / 2.0
        axis_x = w * acq.pixel_size_um / 2.0
        # "auto" focus minimizes total defocus over the population; pass the optics
        # + axis so it folds in field curvature (off-axis cells read deeper).
        focal = resolve_focal_plane(
            scene.cells, acq.focal_depth_in_tissue_um, acq.optics, (axis_y, axis_x)
        )
        dof = acq.optics.resolved_depth_of_field_um
        for cell in scene.cells:
            if cell.footprint_planted is None:
                continue
            z = cell.center_um[0]
            r = math.hypot(cell.center_um[1] - axis_y, cell.center_um[2] - axis_x)
            focal_eff = focal - acq.optics.focal_curvature_shift_um(r)
            sigma_px, brightness = acq.cell_optics(z, focal_eff)
            cell.footprint_observed = degrade_footprint(
                cell.footprint_planted,
                sigma_px,
                acq.tissue.attenuation(z) * acq.optics.collection_efficiency,
            )
            cell.in_focus = abs(z - focal_eff) <= dof
            cell.optical_brightness = brightness
