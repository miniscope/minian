"""Cell-domain steps: place neurons, then give them calcium activity.

These are the first two steps of the forward pipeline — pure biology, before any
optics or sensor effect:

* :class:`PlaceNeuronsStep` positions neurons in a 3-D µm volume and stamps
  a sharp, pre-optics footprint for each. The *distribution* half (sampling
  centers + per-cell SNR, no footprints) is factored into :func:`sample_neurons`
  so it can be reused cheaply for teaching/visualization at full FOV.
* :class:`CellActivityStep` gives every soma a calcium trace built from a
  2-state Markov spike model convolved with a double-exponential indicator
  kernel.

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

from minian.simulation.scene import Cell, Scene
from minian.simulation.steps.base import Step

if TYPE_CHECKING:
    from minian.simulation.spec import PlaceNeurons, SNRDistribution

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
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - cy, xx - cx)
    if irregularity > 0:
        # Low-pass noise on the soma's own scale → a smoothly lumpy boundary,
        # normalized to ~[-1, 1] so `irregularity` is the fractional radius wobble.
        noise = gaussian_filter(
            rng.standard_normal((h, w)), sigma=max(radius_px / 2.0, 1.0)
        )
        noise /= max(noise.max(), -noise.min()) + _EPS  # scale to ~[-1, 1]
        r_eff = radius_px * (1.0 + irregularity * noise)
    else:
        r_eff = radius_px
    # A 0/1 membership mask is already peak-normalized (max == 1) by construction.
    footprint = (dist <= r_eff).astype(float)
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
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Sample the soma *distribution* for a ``PlaceNeurons`` spec over a FOV.

    This is the half of ``place_neurons`` that decides **where cells go and how
    bright they are** — with no footprint stamping, so it is cheap even at a full
    sensor FOV (the per-cell :func:`neuron_footprint` paints are the expensive
    part). :class:`PlaceNeuronsStep` calls this and then stamps a footprint per
    returned center; teaching code (the anatomy notebook's placement widget) can
    call it directly to show the population layout without paying for footprints.

    The count is derived from the areal density and the FOV area,
    ``round(density_per_mm2 · area_mm2)``. Centers are drawn uniformly in
    ``(y, x)`` across the FOV and in ``z`` across ``depth_range_um``; if
    ``min_distance_um > 0`` they are rejection-sampled (Poisson-disk style) to a
    3-D center-to-center minimum. SNRs are drawn from the configured distribution.

    Returns ``(centers, snrs)``: ``centers`` is a list of ``(z, y, x)`` µm tuples
    and ``snrs`` a 1-D array aligned to it. The draw order (centers, then SNRs)
    matches the original fused step, so seeded results are unchanged.
    """
    area_mm2 = (fov_h_um / 1000.0) * (fov_w_um / 1000.0)
    count = round(spec.density_per_mm2 * area_mm2)
    centers = _sample_centers(
        count, fov_h_um, fov_w_um, spec.depth_range_um, spec.min_distance_um, rng
    )
    snrs = _sample_snr(spec.snr, len(centers), rng)
    return centers, snrs


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


def _sample_snr(snr_spec: SNRDistribution, n: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.empty(0)
    if snr_spec.distribution == "uniform":
        return rng.uniform(snr_spec.low, snr_spec.high, size=n)
    # Lognormal: treat (low, high) as ~±2σ anchors in log space, so the bulk
    # of the distribution falls within the configured range.
    mu = (np.log(snr_spec.low) + np.log(snr_spec.high)) / 2.0
    sigma = (np.log(snr_spec.high) - np.log(snr_spec.low)) / 4.0
    return np.exp(rng.normal(mu, sigma, size=n))


class PlaceNeuronsStep(Step):
    """Position neurons in a 3-D µm volume and stamp a planted footprint each.

    Placement and brightness (*where cells go, how bright*) are delegated to
    :func:`sample_neurons`; this step adds the part that function omits — stamping
    a peak-normalized planted footprint (:func:`neuron_footprint`, soma-only or
    soma + proximal dendrites per ``spec.morphology``) at each sampled center.

    The cell count is derived from the areal density and the **canvas** area
    (``round(density_per_mm2 · canvas_area_mm2)``) — the scene movie's grid,
    which a motion margin may enlarge beyond the sensor FOV. The SNR and depth
    are stored now and consumed later by the ``optics`` step (5b) for the
    ``in_focus`` / ``detectable`` flags.
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

        centers, snrs = sample_neurons(spec, fov_h_um, fov_w_um, rng)
        for (z, y, x), snr in zip(centers, snrs):
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
                Cell(center_um=(z, y, x), snr=float(snr), footprint_planted=footprint)
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


class CellActivityStep(Step):
    """Give each soma a calcium trace: Markov gate → Poisson spikes → kernel.

    Per cell, a 2-state Markov chain (quiescent ↔ active, per-frame transition
    probabilities) gates a Poisson spike count whose rate switches between
    ``quiescent_rate_hz`` and ``active_rate_hz``. Spikes get lognormal amplitudes
    (coefficient of variation ``spike_amp_cv``, mean 1) and convolve with the
    double-exponential :func:`calcium_kernel` to form the noise-free trace,
    offset by the baseline ``f0``. Writes ``cell.trace`` (the calcium trace
    ``C``) and ``cell.spikes`` (the spike-count train ``S``); these are the ideal
    deconvolution targets in ground truth.
    """

    name = "cell_activity"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        n_frames, fps = self.acq.n_frames, self.acq.fps
        kernel = calcium_kernel(spec.tau_rise_s, spec.tau_decay_s, fps)
        for cell in scene.cells:
            spikes = self._spike_train(spec, n_frames, fps, self.rng)
            weighted = self._apply_amplitudes(spikes, spec.spike_amp_cv, self.rng)
            trace = spec.f0 + np.convolve(weighted, kernel)[:n_frames]
            if spec.trace_noise > 0:
                trace = trace + self.rng.normal(0.0, spec.trace_noise, size=n_frames)
            cell.trace = trace
            cell.spikes = spikes

    @staticmethod
    def _spike_train(
        spec, n_frames: int, fps: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Per-frame spike counts from the 2-state Markov rate model.

        Sequential by construction (the state at frame ``f`` depends on ``f-1``),
        so this is an explicit O(n_frames) loop — clear over clever, and cheap at
        the recording sizes the simulator targets.
        """
        spikes = np.zeros(n_frames)
        rates = (spec.quiescent_rate_hz, spec.active_rate_hz)
        state = 0  # 0 = quiescent, 1 = active
        for f in range(n_frames):
            spikes[f] = rng.poisson(rates[state] / fps)
            if state == 0 and rng.random() < spec.p_quiescent_to_active:
                state = 1
            elif state == 1 and rng.random() < spec.p_active_to_quiescent:
                state = 0
        return spikes

    @staticmethod
    def _apply_amplitudes(
        spikes: np.ndarray, cv: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Weight each spike by a lognormal amplitude (mean 1, given CV).

        Returns a per-frame summed amplitude. ``cv == 0`` is the noise-free case
        (every spike has unit amplitude), so the result is just the spike counts.
        """
        if cv <= 0:
            return spikes.astype(float)
        sigma = np.sqrt(np.log(1.0 + cv * cv))
        mu = -0.5 * sigma * sigma  # makes E[amplitude] == 1
        weighted = np.zeros_like(spikes, dtype=float)
        for f, c in enumerate(spikes):
            n = int(c)
            if n > 0:
                weighted[f] = np.exp(rng.normal(mu, sigma, size=n)).sum()
        return weighted


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
