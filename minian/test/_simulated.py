"""Synthetic recordings with exact ground truth, for unit-testing pipeline stages
against a known answer rather than the heavy end-to-end notebook.

Builds on minisim's CI-oriented factory,
:func:`minisim.testing.make_recording`, which produces a small, fast,
deterministic recording (well-separated bright somata run through the minimal
forward chain) together with the ground truth that generated it - so a stage's
output can be scored directly instead of eyeballed. minian adds only two things: a
shared geometry/sampling (so a test can convert a pixel-space motion trajectory to
the microns the factory expects), and :func:`as_movie`, which adapts a recording's
``observed`` counts into the dask-backed ``(frame, height, width)``
:class:`xarray.DataArray` the pipeline consumes.

This is the reusable basis for stage tests beyond motion correction.
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr
from minisim.recording import Recording
from minisim.spec import AnyStep
from minisim.testing import make_recording

# Shared geometry/sampling for stage tests. PIXEL_SIZE_UM is exposed so a test can
# convert a pixel-space trajectory into the microns make_recording expects, and
# DURATION_S * FPS fixes the 120-frame length the recursive motion estimator runs on.
N_PX = 256
PIXEL_SIZE_UM = 1.66
FPS = 20.0
DURATION_S = 6.0


def simulated_recording(
    extra_steps: Sequence[AnyStep] = (), *, n_cells: int = 20, seed: int = 1
) -> Recording:
    """A small, fast recording with ground truth at the shared test geometry.

    Thin wrapper over :func:`minisim.testing.make_recording`: a 256x256, 120-frame
    movie of ``n_cells`` well-separated somata. ``extra_steps`` appends effect steps
    (e.g. ``BrainMotion``) to the minimal forward chain.
    """
    return make_recording(
        n_px=N_PX,
        pixel_size_um=PIXEL_SIZE_UM,
        duration_s=DURATION_S,
        fps=FPS,
        n_cells=n_cells,
        seed=seed,
        extra_steps=extra_steps,
    )


def as_movie(recording: Recording, *, chunk_nfm: int = 20) -> xr.DataArray:
    """Adapt a recording into the dask-backed movie minian's pipeline consumes.

    :attr:`~minisim.Recording.observed_movie` is the labeled
    ``(frame, height, width)`` DataArray; chunking it every ``chunk_nfm`` frames
    yields the dask layout :func:`minian.motion_correction.estimate_motion` and the
    rest of the pipeline expect.
    """
    return recording.observed_movie.chunk({"frame": chunk_nfm})
