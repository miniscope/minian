"""Tissue-domain steps: composite the cells, then add brain-bound field effects.

The tissue domain is the brain-frame stack — everything that moves rigidly with
the tissue under :mod:`~minian.simulation` motion (Step 5d), as opposed to the
static optics/sensor fields (:mod:`minian.simulation.steps.sensor`). It opens
with ``render`` (the cell→image boundary) and then layers the diffuse/global
effects that ride on top of the cells:

* :class:`RenderStep` (``cells_only``) — composite ``Σ footprint·trace`` into the
  movie; the first step to write ``scene.movie``.
* :class:`NeuropilStep` (``neuropil``) — additive diffuse background, a smooth
  spatial field modulated by a slow temporal envelope.
* :class:`BleachingStep` (``bleaching``) — global multiplicative fluorophore
  decay over the recording.
* :class:`VasculatureStep` (``vasculature``) — honest no-op placeholder; the
  absorbing-vessel model is deferred to v1.1.

All run before the motion boundary, so a later ``brain_motion`` step translates
the cells *and* these fields together (they are part of the brain frame), unlike
the static vignette/leakage applied after motion.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from minian.simulation.scene import Scene
from minian.simulation.steps.base import Step

# Guards a divide-by-peak for a degenerate (flat) smooth field; far below any
# physically meaningful intensity.
_EPS = 1e-12

# Temporal fluctuation depth of the neuropil envelope, as the log-space sigma of
# its mean-1 lognormal modulation (see :func:`neuropil_envelope`). A fixed v1
# constant — slow background *drifts* by tens of percent rather than blinking;
# per-component variability could become a spec field later.
_NEUROPIL_FLUCT_LOG_STD = 0.4


class RenderStep(Step):
    """Composite ``Σ_i footprint_i · trace_i`` additively into the movie.

    Each cell contributes its footprint scaled, frame by frame, by its calcium
    trace. The *observed* (optically degraded) footprint is used when present;
    until the ``optics`` step (5b) populates it, the *planted* (sharp) footprint
    is used — so the minimal chain renders sharp cells, and gains optical
    realism for free once optics lands, with no change here. Cells missing a
    footprint or a trace are skipped (e.g. before ``cell_activity`` has run), and
    an empty scene leaves the movie untouched. The composite is **additive** so
    later tissue effects (neuropil, etc.) accumulate onto the same movie.
    """

    name = "cells_only"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        footprints, traces = [], []
        for cell in scene.cells:
            footprint = (
                cell.footprint_observed
                if cell.footprint_observed is not None
                else cell.footprint_planted
            )
            if footprint is None or cell.trace is None:
                continue
            footprints.append(footprint)
            traces.append(cell.trace)
        if not footprints:
            return
        A = np.stack(footprints)  # (unit, height, width)
        C = np.stack(traces)  # (unit, frame)
        contrib = np.tensordot(C, A, axes=([0], [0]))  # (frame, height, width)
        scene.movie.values[:] += contrib


# ---------------------------------------------------------------------------
# neuropil
# ---------------------------------------------------------------------------


def smooth_spatial_field(
    shape: tuple[int, int], sigma_px: float, rng: np.random.Generator
) -> np.ndarray:
    """A smooth, non-negative spatial field, peak-normalized to ``[0, 1]``.

    Low-pass-filtered white noise: ``gaussian_filter`` of a standard-normal field
    at ``sigma_px`` produces a blob with structure on the ``sigma_px`` length
    scale. Shifted to be non-negative (background light cannot be negative) and
    divided by its peak so ``amplitude`` carries the absolute level. A degenerate
    flat field falls back to all-ones.
    """
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma_px, mode="nearest")
    field -= field.min()
    peak = float(field.max())
    if peak <= _EPS:
        return np.ones(shape)
    return field / peak


def ou_process(n: int, tau_frames: float, rng: np.random.Generator) -> np.ndarray:
    """A stationary Ornstein–Uhlenbeck sequence: mean 0, unit variance, length ``n``.

    Discrete OU at a one-frame step: ``x[t] = a·x[t-1] + √(1-a²)·ε`` with
    ``a = exp(-1/τ_frames)`` the per-frame correlation. Larger ``τ_frames`` ⇒
    ``a → 1`` ⇒ slower drift. The noise term ``√(1-a²)`` fixes the stationary
    variance at 1, and the first sample is drawn from that stationary
    distribution, so the whole sequence is mean-0/unit-variance with correlation
    time ``τ_frames``. Sequential by construction (``x[t]`` depends on ``x[t-1]``)
    — an explicit loop, cheap at the recording lengths the simulator targets.
    """
    if n <= 0:
        return np.zeros(0)
    a = math.exp(-1.0 / tau_frames) if tau_frames > 0 else 0.0
    noise_scale = math.sqrt(1.0 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = a * x[t - 1] + noise_scale * rng.standard_normal()
    return x


def neuropil_envelope(
    n_frames: int, tau_frames: float, rng: np.random.Generator
) -> np.ndarray:
    """A slow, strictly positive temporal envelope with mean 1.

    Exponentiates a unit OU process into a lognormal modulation
    ``exp(σ·OU − σ²/2)`` (``σ = _NEUROPIL_FLUCT_LOG_STD``): always positive — so
    the additive background it scales stays non-negative without clipping — and
    mean exactly 1, so the neuropil's overall level is set by ``amplitude`` alone,
    not by the temporal fluctuation. The ``−σ²/2`` offset is the lognormal
    mean-1 correction, the same trick used for spike amplitudes in
    :mod:`minian.simulation.steps.cell`.
    """
    s = _NEUROPIL_FLUCT_LOG_STD
    return np.exp(s * ou_process(n_frames, tau_frames, rng) - 0.5 * s * s)


class NeuropilStep(Step):
    """Additive diffuse background: ``amplitude · meanₖ(Sₖ(y,x) · Tₖ(t))``.

    Sums ``n_components`` independent diffuse sources, each a smooth spatial
    field :func:`smooth_spatial_field` (``[0, 1]``, structure on
    ``spatial_sigma_um``) modulated by a slow positive temporal envelope
    :func:`neuropil_envelope` (mean 1, correlation time ``temporal_tau_s``). The
    components are averaged and scaled by ``amplitude`` (the background level
    relative to the ``f0 = 1`` cell baseline), so the contribution is
    non-negative by construction and added onto the movie. This is the modeled
    diffuse mesh only — out-of-focus neurons are a *separate* background that
    emerges for free from ``place_neurons`` + ``optics``.

    Records the spatial fields ``(component, height, width)`` and temporal
    envelopes ``(component, frame)`` to ground truth, so a background-removal
    stage can be scored against the true diffuse structure.
    """

    name = "neuropil"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        spec, acq, rng = self.spec, self.acq, self.rng
        # Grid from the scene canvas (which a motion margin may enlarge beyond
        # the sensor) so the diffuse background covers the same tissue the cells
        # do and moves with it under motion.
        n_frames, h, w = scene.movie.values.shape
        sigma_px = acq.um_to_px(spec.spatial_sigma_um)
        tau_frames = acq.s_to_frame(spec.temporal_tau_s)

        spatial = np.stack(
            [smooth_spatial_field((h, w), sigma_px, rng) for _ in range(spec.n_components)]
        )
        temporal = np.stack(
            [neuropil_envelope(n_frames, tau_frames, rng) for _ in range(spec.n_components)]
        )
        # mean over components of the (frame, h, w) outer products, then scale.
        contrib = np.tensordot(temporal, spatial, axes=([0], [0])) / spec.n_components
        scene.movie.values[:] += spec.amplitude * contrib
        scene.truth.neuropil_spatial = spatial
        scene.truth.neuropil_temporal = temporal


# ---------------------------------------------------------------------------
# bleaching
# ---------------------------------------------------------------------------


def bleaching_curve(model: str, final_fraction: float, n_frames: int) -> np.ndarray:
    """Global per-frame brightness curve, starting at 1 and ending at ``final_fraction``.

    ``mono_exp`` is a single exponential pinned to both endpoints:
    ``b[f] = final_fraction^(f/(n_frames−1))`` — ``b[0] = 1``, ``b[-1] =
    final_fraction``, monotonically decreasing. (``bi_exp`` is rejected at spec
    construction in v1: ``final_fraction`` alone does not determine a two-
    component curve.) A single-frame recording is the trivial ``[1.0]``.
    """
    if model != "mono_exp":
        raise ValueError(f"bleaching model {model!r} is not implemented (mono_exp only).")
    if n_frames <= 1:
        return np.ones(max(n_frames, 1))
    f = np.arange(n_frames)
    return final_fraction ** (f / (n_frames - 1))


class BleachingStep(Step):
    """Global multiplicative fluorophore decay over the recording.

    Multiplies every pixel by a per-frame :func:`bleaching_curve` decaying from 1
    to ``final_fraction`` — the gradual loss of fluorophore brightness under
    sustained excitation. Being a tissue-domain step it runs *before* the static
    sensor fields, so it scales the fluorescing tissue only and never the additive
    sensor ``leakage`` (excitation light hitting the detector does not bleach).
    Records the decay curve ``(frame,)`` to ground truth.
    """

    name = "bleaching"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        curve = bleaching_curve(
            self.spec.model, self.spec.final_fraction, self.acq.n_frames
        )
        scene.movie.values[:] *= curve[:, None, None]
        scene.truth.bleaching = curve


# ---------------------------------------------------------------------------
# vasculature (placeholder)
# ---------------------------------------------------------------------------


class VasculatureStep(Step):
    """Honest no-op placeholder — the absorbing-vessel model is deferred to v1.1.

    The dark, pulsating vasculature mask (a multiplicative absorber driven by slow
    dilation + cardiac motion) is registered in the v1 catalog so the spec surface
    is stable, but its body is not implemented yet. It leaves the scene untouched
    rather than raising, so a spec that lists ``vasculature`` runs end-to-end with
    the effect simply absent (the ground-truth slot stays ``None``).
    """

    name = "vasculature"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        return  # no-op: deferred to v1.1
