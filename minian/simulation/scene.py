"""Mutable runtime state for the ``minian.simulation`` pipeline.

Where :mod:`minian.simulation.spec` is the *immutable* description of a
recording, this module is its *mutable* counterpart — the working state the
executable steps read and write as they run. The split mirrors the rest of the
design: a ``Spec`` is a frozen pydantic tree you can serialize and hash; a
``Scene`` is a plain dataclass the steps mutate in place.

A step (migration Step 5) is a small callable ``step(scene) -> None`` that
either fills the per-cell records in :attr:`Scene.cells`, composites into
:attr:`Scene.movie`, or records its ground-truth contribution on
:attr:`Scene.truth`. Tests construct a ``Scene`` directly — usually with
:meth:`Scene.zeros` or :meth:`Scene.ones` — and run a single step against it;
that, not the full ``simulate()`` loop, is the primary unit-test substrate.

This file (migration Step 4) provides the empty, correctly-shaped substrate and
its constructors only. The step bodies that populate it land in Step 5; the
``finalize()`` that turns an exhausted ``Scene`` into a ``Recording`` (and its
numpydantic ``GroundTruth``) lands with ``simulate()`` in Step 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from minian.simulation.spec import Acquisition

# Movie axis order, matching the minian analysis pipeline (see
# ``minian.utilities`` / ``minian.test.toy_data``) so a simulated recording is a
# drop-in for a real one.
MOVIE_DIMS = ("frame", "height", "width")


@dataclass
class Cell:
    """One simulated neuron — the per-cell record steps fill in, pre-render.

    The fields mirror the per-cell structural columns of the eventual
    ``GroundTruth`` output (spec §8) one-for-one, so ``finalize()`` (Step 6) can
    stack them with no remapping. Each is populated by the step that owns it and
    is ``None`` until then:

    * ``center_um`` / ``snr`` — set by ``place_somata`` (the cell exists once it
      has a location and a signal level).
    * ``footprint_planted`` — the sharp, peak-normalized soma mask, also from
      ``place_somata``; the ideal CNMF target.
    * ``trace`` / ``spikes`` — the noise-free calcium trace ``C`` and spike
      train ``S`` from ``cell_activity``.
    * ``footprint_observed`` / ``in_focus`` / ``optical_brightness`` — the
      optically degraded footprint, the geometric in-focus flag, and the
      depth-driven peak-brightness scalar, all from the ``optics`` step (5b).
    * ``detectable`` is *not* an optics-only property and so is **not** set by
      the optics step: it is a whole-pipeline flag (optical brightness ×
      illumination falloff, judged against the sensor noise floor) assembled in
      ``finalize()`` (Step 6), per spec §8.

    ``center_um`` is ``(z, y, x)`` in µm (depth first); pixel coordinates are a
    conversion away via ``acq.um_to_px`` and are not stored, to avoid drift.
    """

    center_um: tuple[float, float, float]
    snr: float
    footprint_planted: np.ndarray | None = None
    footprint_observed: np.ndarray | None = None
    trace: np.ndarray | None = None
    spikes: np.ndarray | None = None
    in_focus: bool | None = None
    optical_brightness: float | None = None
    detectable: bool | None = None


@dataclass
class GroundTruthBuilder:
    """Per-effect ground-truth side channel — each non-cell step writes its own.

    The per-*cell* truth lives on :attr:`Scene.cells`; this accumulator holds the
    per-*effect* fields that have no natural per-cell home (spec §8). Each is
    ``None`` until the step that produces it runs, so a ``None`` value is the
    honest signal that the effect is absent from this recording:

    * ``shifts`` — rigid (dy, dx) per frame, from ``brain_motion``.
    * ``vignette`` / ``leakage`` — the static (height, width) optical fields.
    * ``bleaching`` — the global per-frame decay curve.
    * ``neuropil_temporal`` / ``neuropil_spatial`` — the diffuse-background
      components.

    Step 4 defines the slots; the steps that fill them arrive in Step 5, and
    ``finalize()`` reads them into the frozen ``GroundTruth`` in Step 6.
    """

    shifts: np.ndarray | None = None
    vignette: np.ndarray | None = None
    leakage: np.ndarray | None = None
    bleaching: np.ndarray | None = None
    neuropil_temporal: np.ndarray | None = None
    neuropil_spatial: np.ndarray | None = None


@dataclass
class Scene:
    """The mutable working state a pipeline of steps mutates in place.

    Holds the acquisition (which owns every µm↔px / s↔frame conversion), the RNG
    every stochastic step draws from, the movie being built, the per-cell
    records, the per-effect ground-truth side channel, and the optional
    per-stage snapshots. Unlike a ``Spec`` it is *not* frozen — mutation is the
    point.

    The working ``movie`` is held in **float64**: effects accumulate additively
    and multiplicatively across ~10 steps in honest radiometric units, and only
    the final ``sensor`` step quantizes to integer counts. The downcast to
    ``Output.store_dtype`` happens in ``finalize()`` (Step 6), not here.
    """

    acq: Acquisition
    rng: np.random.Generator
    movie: xr.DataArray
    cells: list[Cell] = field(default_factory=list)
    truth: GroundTruthBuilder = field(default_factory=GroundTruthBuilder)
    snapshots: dict[str, xr.DataArray] = field(default_factory=dict)

    @classmethod
    def zeros(cls, acq: Acquisition, rng: np.random.Generator | None = None) -> Scene:
        """A blank scene whose movie is all zeros — the base for additive builds."""
        return cls._blank(acq, 0.0, rng)

    @classmethod
    def ones(cls, acq: Acquisition, rng: np.random.Generator | None = None) -> Scene:
        """A scene whose movie is all ones — the substrate for multiplicative-field tests.

        A ``vignette`` or ``leakage`` step applied to an all-ones movie yields the
        bare field, which is exactly what a single-step test inspects.
        """
        return cls._blank(acq, 1.0, rng)

    @classmethod
    def _blank(cls, acq: Acquisition, fill: float, rng: np.random.Generator | None) -> Scene:
        if rng is None:
            rng = np.random.default_rng()
        n_frames = acq.n_frames
        height = acq.image_sensor.n_px_height
        width = acq.image_sensor.n_px_width
        movie = xr.DataArray(
            np.full((n_frames, height, width), fill, dtype=np.float64),
            dims=list(MOVIE_DIMS),
            coords={
                "frame": np.arange(n_frames),
                "height": np.arange(height),
                "width": np.arange(width),
            },
            name="movie",
        )
        return cls(acq=acq, rng=rng, movie=movie)
