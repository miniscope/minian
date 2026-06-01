"""Cell-domain steps: place somata, then give them calcium activity.

These are the first two steps of the forward pipeline — pure biology, before any
optics or sensor effect:

* :class:`PlaceSomataStep` positions neuron somata in a 3-D µm volume and stamps
  a sharp, pre-optics footprint for each.
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
    from minian.simulation.spec import Optics

# Guards the noise normalization for a degenerate (flat) low-pass field; far
# below any physically meaningful intensity.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# place_somata
# ---------------------------------------------------------------------------


def soma_footprint(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    radius_px: float,
    irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A sharp, peak-normalized soma mask — a (possibly lumpy) filled disk.

    Models the soma as a filled disk of radius ``radius_px`` centered at
    ``center_px`` (both in pixels). This is the *planted* footprint: the true
    spatial extent of the fluorophore-filled cytoplasm, with **no optical blur**
    — diffraction/defocus/scatter are applied later by the ``optics`` step. It
    is peak-normalized (``max == 1``) so a cell's brightness is carried entirely
    by its calcium trace, not baked into the footprint.

    ``irregularity`` ∈ [0, 1] warps the boundary: at ``0`` it is a clean disk;
    above ``0`` the per-pixel radius is modulated by a low-pass-filtered noise
    field (smoothed on the soma's own length scale), giving a lumpy outline that
    is more soma-like than a perfect circle while staying coarser than the
    optics will later blur away. Typical cortical somata are ~5–10 µm radius.
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
    if not footprint.any():
        # Sub-pixel soma: keep at least the nearest pixel lit so the cell is
        # never silently empty.
        iy = int(np.clip(round(cy), 0, h - 1))
        ix = int(np.clip(round(cx), 0, w - 1))
        footprint[iy, ix] = 1.0
    return footprint


class PlaceSomataStep(Step):
    """Position somata in a 3-D µm volume and stamp a planted footprint each.

    The cell count is derived from the areal density and the **canvas** area
    (``round(density_per_mm2 · canvas_area_mm2)``) — the scene movie's grid,
    which a motion margin may enlarge beyond the sensor FOV. Centers are drawn
    uniformly in ``(y, x)`` across the canvas and in ``z`` across
    ``depth_range_um``; if
    ``min_distance_um > 0`` they are rejection-sampled (Poisson-disk style) to a
    3-D center-to-center minimum. Each cell gets a peak-normalized planted
    footprint (:func:`soma_footprint`) and an SNR drawn from the configured
    distribution. The SNR and depth are stored now and consumed later by the
    ``optics`` step (5b) for the ``in_focus`` / ``detectable`` flags.
    """

    name = "place_somata"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        spec = self.spec  # n_neurite_stubs > 0 is rejected at spec construction (v1)
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
        area_mm2 = (fov_h_um / 1000.0) * (fov_w_um / 1000.0)
        count = round(spec.density_per_mm2 * area_mm2)
        radius_px = acq.um_to_px(spec.soma_radius_um)

        centers = self._sample_centers(
            count, fov_h_um, fov_w_um, spec.depth_range_um, spec.min_distance_um, rng
        )
        snrs = self._sample_snr(spec.snr, len(centers), rng)
        for (z, y, x), snr in zip(centers, snrs):
            footprint = soma_footprint(
                shape,
                (acq.um_to_px(y), acq.um_to_px(x)),
                radius_px,
                spec.irregularity,
                rng,
            )
            scene.cells.append(
                Cell(center_um=(z, y, x), snr=float(snr), footprint_planted=footprint)
            )

    @staticmethod
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

    @staticmethod
    def _sample_snr(snr_spec, n: int, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty(0)
        if snr_spec.distribution == "uniform":
            return rng.uniform(snr_spec.low, snr_spec.high, size=n)
        # Lognormal: treat (low, high) as ~±2σ anchors in log space, so the bulk
        # of the distribution falls within the configured range.
        mu = (np.log(snr_spec.low) + np.log(snr_spec.high)) / 2.0
        sigma = (np.log(snr_spec.high) - np.log(snr_spec.low)) / 4.0
        return np.exp(rng.normal(mu, sigma, size=n))


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


def resolve_focal_plane(cells: list[Cell], optics: Optics) -> float:
    """Resolve ``Optics.focal_plane_um`` to a concrete depth, µm.

    A numeric focal plane is used as-is. ``"auto"`` resolves to the **median
    realized cell depth**, so the focal plane sits in the middle of the placed
    population (the most cells in focus). An empty scene falls back to the
    surface (``0.0``). This is the one place ``"auto"`` becomes concrete; every
    downstream read sees a number.
    """
    focal = optics.focal_plane_um
    if focal != "auto":
        return float(focal)
    if not cells:
        return 0.0
    return float(np.median([cell.center_um[0] for cell in cells]))


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
    """
    return gain * gaussian_filter(planted, sigma=sigma_px, mode="constant")


class CellOpticsStep(Step):
    """Degrade each planted footprint by diffraction + defocus(|z−focal|) + scatter(z).

    Reads each cell's depth ``z`` and the physical ``Optics``/``Tissue``
    constants (via :meth:`Acquisition.cell_optics`) — there are no tunable
    fields. For every cell it:

    * writes ``footprint_observed = gain · (planted ⊛ Gaussian(σ_total))`` where
      ``gain = attenuation(z) · collection_efficiency`` — the blurred, dimmed
      footprint CNMF could actually recover;
    * sets ``in_focus`` geometrically (``|z − focal| ≤ depth_of_field_um``);
    * stores ``optical_brightness`` — the per-cell *peak* scalar from
      ``cell_optics`` (defocus drops the peak as ``σ₀²/σ_total²``; scatter
      ``attenuation(z)`` and ``collection_efficiency ∝ NA²`` dim it). Footprint
      *integral* scales with that same ``gain``, but a cell's *detectability*
      turns on its peak, which defocus also lowers — hence two distinct
      quantities. ``detectable`` itself is left for ``finalize()``
      (Step 6), where this peak combines with the illumination field and the
      sensor noise floor.

    The focal plane is resolved once for the whole scene (``"auto"`` → median
    cell depth). Cells without a planted footprint are skipped.
    """

    name = "optics"
    domain = "cell"

    def __call__(self, scene: Scene) -> None:
        acq = self.acq
        focal = resolve_focal_plane(scene.cells, acq.optics)
        dof = acq.optics.depth_of_field_um
        for cell in scene.cells:
            if cell.footprint_planted is None:
                continue
            z = cell.center_um[0]
            sigma_px, brightness = acq.cell_optics(z, focal)
            cell.footprint_observed = degrade_footprint(
                cell.footprint_planted,
                sigma_px,
                acq.tissue.attenuation(z) * acq.optics.collection_efficiency,
            )
            cell.in_focus = abs(z - focal) <= dof
            cell.optical_brightness = brightness
