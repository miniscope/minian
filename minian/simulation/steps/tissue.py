"""Tissue-domain step: composite the per-cell footprints and traces into a movie.

``render`` is the boundary between the per-cell world (footprints + traces held
on ``scene.cells``) and the image world (a movie). It is the first step to write
into ``scene.movie``; everything before it only fills per-cell records.

Its stage name is ``"cells_only"`` (the snapshot label, distinct from the spec
``kind`` — see ``simulation-spec.md`` §7): at this point the movie is exactly the
sum of cells, with no background, motion, or sensor effect yet.
"""

from __future__ import annotations

import numpy as np

from minian.simulation.scene import Scene
from minian.simulation.steps.base import Step


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
