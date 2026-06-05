"""Executable pipeline steps for ``minian.simulation``.

Each step is the runtime counterpart of a ``StepSpec`` (see
:mod:`minian.simulation.spec`): a small callable that mutates a ``Scene`` in
place, returned by the spec's ``build()`` method. They are organized by pipeline
domain — ``cell`` → ``tissue`` → ``motion`` → ``sensor`` — mirroring the forward
order biology → optics → motion → sensor.

Migration Step 5a lands the minimal runnable chain (``place_neurons`` →
``cell_activity`` → ``render`` → ``sensor``); optics is 5b, the field effects
(``neuropil``/``bleaching``/``vignette``/``leakage``, plus the ``vasculature``
no-op placeholder) are 5c, and ``brain_motion`` is 5d — the full forward
pipeline. The :class:`Step` base and the physics helpers
(:func:`calcium_kernel`, :func:`neuron_footprint`, :func:`bleaching_curve`,
:func:`ou_process`, :func:`bounded_random_walk`, …) are exposed here for direct
unit testing and teaching.
"""

from minian.simulation.steps.base import Step
from minian.simulation.steps.cell import (
    CellActivityStep,
    CellOpticsStep,
    PlaceNeuronsStep,
    calcium_kernel,
    degrade_footprint,
    kernel_timing,
    neuron_footprint,
    resolve_focal_plane,
    sample_neurons,
    spike_activity_params,
    tau_from_kernel_timing,
)
from minian.simulation.steps.motion import (
    BrainMotionStep,
    bounded_random_walk,
    shift_and_crop,
)
from minian.simulation.steps.sensor import (
    LeakageStep,
    SensorStep,
    VignetteStep,
    radius_grid,
)
from minian.simulation.steps.tissue import (
    BleachingStep,
    NeuropilStep,
    RenderStep,
    VasculatureStep,
    bleaching_curve,
    neuropil_envelope,
    ou_process,
    smooth_spatial_field,
)

__all__ = [
    "BleachingStep",
    "BrainMotionStep",
    "CellActivityStep",
    "CellOpticsStep",
    "LeakageStep",
    "NeuropilStep",
    "PlaceNeuronsStep",
    "RenderStep",
    "SensorStep",
    "Step",
    "VasculatureStep",
    "VignetteStep",
    "bleaching_curve",
    "bounded_random_walk",
    "calcium_kernel",
    "degrade_footprint",
    "kernel_timing",
    "neuron_footprint",
    "neuropil_envelope",
    "ou_process",
    "radius_grid",
    "resolve_focal_plane",
    "sample_neurons",
    "shift_and_crop",
    "spike_activity_params",
    "tau_from_kernel_timing",
    "smooth_spatial_field",
]
