"""Physically-driven synthetic 1-photon miniscope data — generator + teaching tool.

The simulator builds a recording forward from its physical components (biology →
optics → motion → sensor), the inverse of the minian analysis pipeline. v1
public surface so far is the typed ``Spec`` (this module's ``spec`` submodule);
the executable engine, metrics, and presets arrive in later migration steps.

See ``proposals/simulation-plan.md`` and ``proposals/simulation-spec.md``.
"""

from minian.simulation.cache import cache_dir, cache_path, simulate_cached
from minian.simulation.metrics import (
    Match,
    SpikeScore,
    field_pearson,
    hungarian_match,
    shift_rmse,
    spike_precision_recall,
    trace_pearson,
)
from minian.simulation.recording import GroundTruth, Recording, finalize
from minian.simulation.scene import Cell, GroundTruthBuilder, Scene
from minian.simulation.simulate import simulate
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
from minian.simulation.sweep import SweptSpec, sweep

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
    "Match",
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
    "SpikeScore",
    "StepSpec",
    "SweptSpec",
    "Tissue",
    "Vasculature",
    "Vignette",
    "cache_dir",
    "cache_path",
    "field_pearson",
    "finalize",
    "hungarian_match",
    "shift_rmse",
    "simulate",
    "simulate_cached",
    "spike_precision_recall",
    "sweep",
    "trace_pearson",
]
