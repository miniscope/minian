"""Small synthetic recordings with exact ground truth, for unit-testing pipeline
stages against a known answer rather than the heavy end-to-end notebook.

`minisim <https://github.com/miniscope/minisim>`_ builds a recording forward from
its physical components (biology, optics, motion, sensor) and ships the ground
truth that generated it, so a stage's output can be scored directly - estimated
motion against the applied shifts, recovered footprints against the planted ones -
instead of eyeballed. These helpers keep the recording small and fast: a 256x256,
120-frame movie simulates in ~1-2 s, and :func:`minisim.simulate_cached` memoizes
by spec hash so reruns are instant.

This module is the reusable basis for stage tests beyond motion correction:
:func:`simulate_recording` builds the recording (optionally with extra effect
steps), and :func:`as_movie` adapts its ``observed`` array into the dask-backed
``(frame, height, width)`` :class:`xarray.DataArray` that minian's functions
consume.
"""

from __future__ import annotations

from collections.abc import Sequence

import dask.array as darr
import numpy as np
import xarray as xr
from minisim import build_spec, simulate_cached
from minisim.presets import Region, Scope, ca1
from minisim.recording import Recording
from minisim.spec import AnyStep, ImageSensor, Optics, Spec

# A small, well-textured scope: 256x256 px at the Miniscope V4 magnification, so
# ~1.66 um/px over a ~424 um field of view. Small enough that estimate_motion runs
# in a fraction of a second, large enough that the CA1 preset packs ~80-90
# detectable cells into the FOV - rich texture for phase-correlation registration.
SIM_SCOPE = Scope(
    optics=Optics(magnification=2.9),
    image_sensor=ImageSensor(n_px_height=256, n_px_width=256, pixel_pitch_um=4.8),
    focal_depth_in_tissue_um="auto",
)

# Object-space size of one pixel (um), so a test can prescribe a motion trajectory
# in pixels and convert it to the um a BrainMotion spec expects.
PX_SIZE_UM = SIM_SCOPE.pixel_size_um


def make_spec(
    extra_steps: Sequence[AnyStep] = (),
    *,
    region: Region | None = None,
    duration_s: float = 6.0,
    fps: float = 20.0,
    seed: int = 1,
) -> Spec:
    """Build a validated :class:`~minisim.Spec` on :data:`SIM_SCOPE`.

    ``extra_steps`` appends effect steps (e.g. ``BrainMotion``) to the minimal
    forward chain; ``region`` defaults to :func:`minisim.presets.ca1`.
    """
    return build_spec(
        SIM_SCOPE,
        region or ca1(),
        duration_s=duration_s,
        fps=fps,
        seed=seed,
        extra_steps=extra_steps,
    )


def simulate_recording(extra_steps: Sequence[AnyStep] = (), **kwargs: object) -> Recording:
    """Simulate (or load from cache) a small recording with ground truth.

    Thin wrapper over :func:`minisim.simulate_cached` and :func:`make_spec`; keyword
    arguments are forwarded to :func:`make_spec` (``region``, ``duration_s``,
    ``fps``, ``seed``).
    """
    return simulate_cached(make_spec(extra_steps, **kwargs))


def as_movie(recording: Recording, *, chunk_nfm: int = 20) -> xr.DataArray:
    """Adapt a recording's ``observed`` counts into a minian-shaped movie.

    Returns a dask-backed ``(frame, height, width)`` :class:`xarray.DataArray`
    chunked every ``chunk_nfm`` frames - the layout :func:`minian.motion_correction.estimate_motion`
    and the rest of the pipeline expect.
    """
    obs = recording.observed
    n_frames, height, width = obs.shape
    return xr.DataArray(
        darr.from_array(obs, chunks=(chunk_nfm, -1, -1)),
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(n_frames),
            "height": np.arange(height),
            "width": np.arange(width),
        },
        name="movie",
    )
