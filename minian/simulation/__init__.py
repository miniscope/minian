"""Physically-driven synthetic 1-photon miniscope data — generator + teaching tool.

The simulator builds a recording forward from its physical components (biology →
optics → motion → sensor), the inverse of the minian analysis pipeline. v1
public surface so far is the typed ``Spec`` (this module's ``spec`` submodule);
the executable engine, metrics, and presets arrive in later migration steps.

See ``proposals/simulation-plan.md`` and ``proposals/simulation-spec.md``.
"""

from minian.simulation.recording import GroundTruth, Recording, finalize
from minian.simulation.scene import Cell, GroundTruthBuilder, Scene
from minian.simulation.spec import (
    Acquisition,
    AnyStep,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    Output,
    PlaceSomata,
    Render,
    Sensor,
    SNRDistribution,
    Spec,
    SpecWarning,
    StepSpec,
    Tissue,
    Vasculature,
    Vignette,
)

__all__ = [
    "Acquisition",
    "AnyStep",
    "Bleaching",
    "BrainMotion",
    "Cell",
    "CellActivity",
    "CellOptics",
    "GroundTruth",
    "GroundTruthBuilder",
    "ImageSensor",
    "Leakage",
    "Neuropil",
    "Optics",
    "Output",
    "PlaceSomata",
    "Recording",
    "Render",
    "SNRDistribution",
    "Scene",
    "Sensor",
    "Spec",
    "SpecWarning",
    "StepSpec",
    "Tissue",
    "Vasculature",
    "Vignette",
    "finalize",
]
