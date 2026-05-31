"""Sensor-domain step: convert scene intensity to digitized camera counts.

``sensor`` is the last step of the forward pipeline and the only one that turns
honest radiometric intensity into the integer ADC counts a real recording is
made of. It scales the scene's fluorescence intensity to an expected photon
count (the ``photons_per_unit`` exposure scale — a scene/illumination property,
not sensor hardware), then defers to the detector's forward model
(:meth:`~minian.simulation.spec.ImageSensor.photons_to_counts`) for shot noise,
read noise, gain, quantization, and clipping.
"""

from __future__ import annotations

import numpy as np

from minian.simulation.scene import Scene
from minian.simulation.steps.base import Step


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
