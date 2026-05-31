"""Executable pipeline steps for ``minian.simulation``.

Each step is the runtime counterpart of a ``StepSpec`` (see
:mod:`minian.simulation.spec`): a small callable that mutates a ``Scene`` in
place, returned by the spec's ``build()`` method. They are organized by pipeline
domain — ``cell`` → ``tissue`` → ``motion`` → ``sensor`` — mirroring the forward
order biology → optics → motion → sensor.

Migration Step 5a lands the minimal runnable chain (``place_somata`` →
``cell_activity`` → ``render`` → ``sensor``); the remaining steps arrive in
5b–5d. The :class:`Step` base and the physics helpers (:func:`calcium_kernel`,
:func:`soma_footprint`) are exposed here for direct unit testing and teaching.
"""

from minian.simulation.steps.base import Step
from minian.simulation.steps.cell import (
    CellActivityStep,
    CellOpticsStep,
    PlaceSomataStep,
    calcium_kernel,
    degrade_footprint,
    resolve_focal_plane,
    soma_footprint,
)
from minian.simulation.steps.sensor import SensorStep
from minian.simulation.steps.tissue import RenderStep

__all__ = [
    "CellActivityStep",
    "CellOpticsStep",
    "PlaceSomataStep",
    "RenderStep",
    "SensorStep",
    "Step",
    "calcium_kernel",
    "degrade_footprint",
    "resolve_focal_plane",
    "soma_footprint",
]
