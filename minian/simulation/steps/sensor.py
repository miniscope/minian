"""Sensor-domain steps: the static optical fields, then digitization.

The sensor domain is the *static* frame — effects fixed to the optics and
detector that do **not** move with the brain (unlike the tissue-frame steps in
:mod:`minian.simulation.steps.tissue`). They are applied after the motion
boundary:

* :class:`VignetteStep` (``vignette``) — multiplicative radial illumination
  falloff (lumped excitation × collection efficiency).
* :class:`LeakageStep` (``leakage``) — additive static baseline, the "glow"
  minian's glow-removal subtracts.
* :class:`SensorStep` (``sensor``) — the only step that turns honest radiometric
  intensity into integer ADC counts (shot + read noise, gain, quantization).

The two field steps each record their static ``(height, width)`` field to ground
truth. The vignette field in particular is load-bearing downstream: ``finalize()``
(Step 6) reads it at each cell's lateral position to finish the per-cell
``detectable`` flag, since excitation falloff dims edge cells.
"""

from __future__ import annotations

import numpy as np

from minian.simulation.scene import Scene
from minian.simulation.steps.base import Step


def radius_grid(shape: tuple[int, int], center_px: tuple[float, float]) -> np.ndarray:
    """Per-pixel Euclidean distance (in pixels) from ``center_px``, shape ``shape``.

    The shared geometry behind both static fields — the vignette falloff and the
    gaussian leakage glow are each a radial function of this distance.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.hypot(yy - center_px[0], xx - center_px[1])


class VignetteStep(Step):
    """Multiplicative radial illumination falloff (static, lumped excitation × collection).

    Multiplies every frame by the same field
    ``V(r) = 1 − (1 − falloff)·(r / r_max)^exponent``: ``1`` at the bright center,
    dropping to ``falloff`` at the farthest corner, with ``exponent`` shaping the
    rolloff. The bright center sits at the FOV center plus ``center_offset_um``
    (converted to pixels). The field is time-invariant — the same for every
    frame — and recorded ``(height, width)`` to ground truth, where it is the
    illumination field a normalization stage estimates and that ``finalize()``
    reads for per-cell detectability.
    """

    name = "vignette"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        spec, acq = self.spec, self.acq
        # Grid from the scene movie. As a sensor-frame step this runs after the
        # motion crop, so the movie is already the sensor FOV — the static field
        # is fixed to the detector and must not extend into the motion margin.
        h, w = scene.movie.values.shape[1:]
        cy = (h - 1) / 2.0 + acq.um_to_px(spec.center_offset_um[0])
        cx = (w - 1) / 2.0 + acq.um_to_px(spec.center_offset_um[1])
        r = radius_grid((h, w), (cy, cx))
        # Normalize by the distance to the farthest corner so V == falloff there.
        r_max = max(
            np.hypot(cy - yc, cx - xc) for yc in (0, h - 1) for xc in (0, w - 1)
        )
        if r_max <= 0:  # 1×1 FOV: no falloff to apply.
            field = np.ones((h, w))
        else:
            field = 1.0 - (1.0 - spec.falloff) * (r / r_max) ** spec.exponent
        scene.movie.values[:] *= field
        scene.truth.vignette = field


class LeakageStep(Step):
    """Additive static baseline — the "glow" minian's glow-removal subtracts.

    Adds the same field to every frame: ``uniform`` is a flat ``level``
    everywhere; ``gaussian`` is a central glow ``level·exp(−r²/2σ²)`` peaking at
    the FOV center, with ``sigma_um`` defaulting to a quarter of the smaller FOV
    dimension when unset. Time-invariant and recorded ``(height, width)`` to
    ground truth. Being a sensor-frame step it is applied after motion and is not
    scaled by ``bleaching`` (stray excitation light reaching the detector neither
    moves with the brain nor bleaches).
    """

    name = "leakage"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        spec, acq = self.spec, self.acq
        # Grid from the scene movie (the sensor FOV post-crop); see VignetteStep.
        h, w = scene.movie.values.shape[1:]
        if spec.profile == "uniform":
            field = np.full((h, w), spec.level)
        else:  # gaussian central glow
            sigma_um = spec.sigma_um if spec.sigma_um is not None else 0.25 * min(acq.fov_um)
            sigma_px = acq.um_to_px(sigma_um)
            r = radius_grid((h, w), ((h - 1) / 2.0, (w - 1) / 2.0))
            field = spec.level * np.exp(-(r**2) / (2.0 * sigma_px**2))
        scene.movie.values[:] += field
        scene.truth.leakage = field


class SensorStep(Step):
    """Intensity → expected photons → digitized counts (the only count-producing step).

    Multiplies the working movie by ``photons_per_unit`` to get the per-pixel
    expected photon count (clipped at 0 — negative intensity from optional trace
    noise is unphysical light), then runs the image sensor's forward model to add
    shot + read noise and quantize to clipped integer counts. The result is
    written back into ``scene.movie`` as integer-valued counts in the float
    working container; the downcast to ``Output.store_dtype`` is a ``finalize()``
    concern (migration Step 6).
    """

    name = "sensor"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        photons = np.clip(scene.movie.values * self.spec.photons_per_unit, 0.0, None)
        counts = self.acq.image_sensor.photons_to_counts(photons, self.rng)
        scene.movie.values[:] = counts
