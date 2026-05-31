"""The executable-step contract for the ``minian.simulation`` pipeline.

Where a ``StepSpec`` (:mod:`minian.simulation.spec`) is the *immutable,
serializable description* of a step, a ``Step`` is its *executable counterpart*:
a small callable that mutates a :class:`~minian.simulation.scene.Scene` in place.
``StepSpec.build()`` turns the former into the latter, binding it to the
``Acquisition`` (which owns every µm↔px / s↔frame conversion) and the RNG it
draws from.

A step does exactly one thing to the scene — fill per-cell records, composite
into the movie, or record a ground-truth contribution — so a single step can be
constructed and run against a hand-built ``Scene`` in a unit test. That, not the
full ``simulate()`` loop (migration Step 6), is the primary test substrate.

Each step captures the RNG handed to ``build()`` and draws from *that* generator
(not ``scene.rng``), so a step is reproducible from its construction regardless
of which scene it is later run against — matching the ``build(acq, rng)``
signature the orchestrator and the unit tests both use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    import numpy as np

    from minian.simulation.scene import Scene
    from minian.simulation.spec import Acquisition, StepSpec


class Step:
    """Base class for an executable pipeline step.

    Subclasses set the two class attributes and implement :meth:`__call__`:

    * ``name`` — the snapshot/stage name (usually the spec ``kind``; ``render``
      is the exception, whose stage is ``"cells_only"``).
    * ``domain`` — the pipeline domain (``cell`` → ``tissue`` → ``motion`` →
      ``sensor``), the same ordering axis the spec validator checks.

    The constructor stores the originating spec, the acquisition, and the RNG;
    subclasses read parameters off ``self.spec`` and physical conversions off
    ``self.acq``.
    """

    name: ClassVar[str]
    domain: ClassVar[Literal["cell", "tissue", "motion", "sensor"]]

    def __init__(
        self, spec: StepSpec, acq: Acquisition, rng: np.random.Generator
    ) -> None:
        self.spec = spec
        self.acq = acq
        self.rng = rng

    def __call__(self, scene: Scene) -> None:
        """Mutate ``scene`` in place. Implemented by each concrete step."""
        raise NotImplementedError(f"{type(self).__name__} does not implement __call__.")
