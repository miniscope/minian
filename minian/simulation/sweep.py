"""Parameter sweeps — a thin Cartesian-product generator over ``Spec`` overrides.

:func:`sweep` takes a base ``Spec`` and a dict of *dotted-path → list-of-values*
axes and yields one validated spec per point in their Cartesian product. It is the
single primitive shared by two consumers (see ``simulation-plan.md`` §"Parameter
sweeps"): ``@pytest.mark.parametrize`` correctness grids, and a benchmark harness
that collects metric values into a tidy DataFrame keyed on the physical axes. The
core :func:`~minian.simulation.simulate.simulate` never depends on this module.

Because the physical interface makes the axes physically meaningful (depth µm,
density cells/mm², NA, exposure), a sweep traces a scientifically interpretable
surface — e.g. "recall vs depth at NA 0.45". Each yielded spec carries an
``axes`` dict (the chosen value per path) for those benchmark rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

from pydantic import BaseModel, Field

from minian.simulation.spec import Spec, StepSpec


class SweptSpec(Spec):
    """A ``Spec`` tagged with the sweep-axis values that produced it.

    A genuine ``Spec`` subclass, so it drops into ``simulate()`` and ``Recording``
    unchanged. ``axes`` is excluded from serialization, so it never reaches
    ``model_dump_json`` and therefore leaves :meth:`Spec.cache_key` identical to
    the equivalent plain spec — sweeping does not perturb cache dedup, and the tag
    simply vanishes when a recording is persisted.
    """

    axes: dict = Field(
        default_factory=dict,
        exclude=True,
        description="Chosen value per swept dotted-path, for tidy benchmark rows.",
    )


def sweep(base: Spec, axes: dict[str, list]) -> Iterator[SweptSpec]:
    """Yield one validated ``SweptSpec`` per point in the Cartesian product of ``axes``.

    Parameters
    ----------
    base
        The spec every combination starts from; never mutated.
    axes
        Maps a dotted override path to the list of values to sweep it over. Path
        forms:

        * ``"acquisition.optics.na"`` — walk nested models;
        * ``"steps.<kind>.<field>"`` — address the step with that (unique) ``kind``,
          e.g. ``"steps.place_neurons.density_per_mm2"``;
        * ``"seed"`` — a top-level field.

    Yields
    ------
    SweptSpec
        The base with this combination of overrides applied and **all cross-field
        validators re-run** — an axis value that yields an invalid combination
        (``na=-1``, a soma larger than the FOV, …) raises here. An empty ``axes``
        yields the base once with ``axes={}``.

    Raises
    ------
    ValueError
        For a path naming an unknown field, an unknown step kind, or one that
        descends into a non-model (scalar) field.
    """
    paths = list(axes)
    for combo in product(*(axes[p] for p in paths)):
        chosen = dict(zip(paths, combo))
        overridden = base
        for path, value in chosen.items():
            overridden = _set_path(overridden, path.split("."), path, value)
        # Re-validate from the canonical dump: rebuilds the discriminated `steps`
        # union and re-runs every §11 validator, so a bad axis fails fast here.
        yield SweptSpec.model_validate({**overridden.model_dump(), "axes": chosen})


def _set_path(model: BaseModel, parts: list[str], full_path: str, value) -> BaseModel:
    """Return a copy of ``model`` with the dotted ``parts`` set to ``value`` (immutably).

    Each segment is checked against the model's fields *before* the copy: pydantic's
    ``model_copy(update=...)`` skips validation and silently accepts unknown keys, so
    an unchecked typo would no-op rather than raise.
    """
    head, *rest = parts
    if head == "steps":
        if not rest:
            raise ValueError(f"steps path {full_path!r} must be steps.<kind>.<field>.")
        kind, *field_parts = rest
        idx = _step_index(model, kind, full_path)
        new_steps = list(model.steps)
        new_steps[idx] = _set_path(new_steps[idx], field_parts, full_path, value)
        return model.model_copy(update={"steps": new_steps})

    if head not in model.model_fields:
        raise ValueError(f"unknown field {head!r} in sweep path {full_path!r}.")
    if not rest:
        return model.model_copy(update={head: value})

    child = getattr(model, head)
    if not isinstance(child, BaseModel):
        raise ValueError(f"sweep path {full_path!r} descends into non-model field {head!r}.")
    return model.model_copy(update={head: _set_path(child, rest, full_path, value)})


def _step_index(model: BaseModel, kind: str, full_path: str) -> int:
    """Index in ``model.steps`` of the step whose ``kind`` matches (kinds are unique)."""
    steps: list[StepSpec] = model.steps
    for i, step in enumerate(steps):
        if step.kind == kind:
            return i
    available = sorted(s.kind for s in steps)
    raise ValueError(
        f"no step of kind {kind!r} in sweep path {full_path!r}; spec has {available}."
    )
