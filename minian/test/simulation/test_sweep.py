"""Unit tests for the parameter-sweep generator (migration Step 9).

Covers the Cartesian product and axis bookkeeping, the three dotted-path forms
(nested model, step-by-kind, top-level), immutability of the base spec, fail-fast
on bad paths, cache_key parity (axes never perturb the hash), and that a yielded
spec is a genuine, runnable Spec.
"""

import numpy as np
import pytest

from minian.simulation import (
    Acquisition,
    CellActivity,
    CellOptics,
    ImageSensor,
    Optics,
    PlaceNeurons,
    Render,
    Sensor,
    Spec,
    SweptSpec,
    simulate,
    sweep,
)


def _base():
    """A short, canonical-order spec at a clean 1.0 µm/px scale (no SpecWarnings)."""
    return Spec(
        acquisition=Acquisition(
            fps=20.0,
            duration_s=1.0,
            optics=Optics(magnification=8.0, na=0.45),
            image_sensor=ImageSensor(n_px_height=64, n_px_width=64, pixel_pitch_um=8.0, bit_depth=8),
        ),
        seed=7,
        steps=[
            PlaceNeurons(density_per_mm2=2500.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Sensor(photons_per_unit=100.0),
        ],
    )


def _place(spec):
    return next(s for s in spec.steps if s.kind == "place_neurons")


def test_cartesian_product_count_and_axes():
    specs = list(sweep(_base(), {
        "acquisition.optics.na": [0.3, 0.6],
        "steps.place_neurons.density_per_mm2": [50.0, 150.0, 400.0],
    }))
    assert len(specs) == 6
    # every combination is present exactly once
    seen = {(s.acquisition.optics.na, _place(s).density_per_mm2) for s in specs}
    assert seen == {(na, d) for na in (0.3, 0.6) for d in (50.0, 150.0, 400.0)}
    # .axes mirrors the chosen values for each yielded spec
    for s in specs:
        assert s.axes["acquisition.optics.na"] == s.acquisition.optics.na
        assert s.axes["steps.place_neurons.density_per_mm2"] == _place(s).density_per_mm2


def test_nested_model_path_override():
    (s,) = list(sweep(_base(), {"acquisition.optics.na": [0.33]}))
    assert s.acquisition.optics.na == 0.33
    assert isinstance(s, Spec)


def test_step_by_kind_override_leaves_siblings_untouched():
    base = _base()
    (s,) = list(sweep(base, {"steps.place_neurons.soma_radius_um": [6.5]}))
    assert _place(s).soma_radius_um == 6.5
    # a different step is carried through unchanged
    act = next(x for x in s.steps if x.kind == "cell_activity")
    assert act.active_rate_hz == 5.0


def test_top_level_path_override():
    (s,) = list(sweep(_base(), {"seed": [99]}))
    assert s.seed == 99


def test_tuple_valued_axis():
    specs = list(sweep(_base(), {"steps.place_neurons.depth_range_um": [(0.0, 75.0), (0.0, 150.0)]}))
    assert [_place(s).depth_range_um for s in specs] == [(0.0, 75.0), (0.0, 150.0)]


def test_empty_axes_yields_base_once():
    base = _base()
    specs = list(sweep(base, {}))
    assert len(specs) == 1
    assert specs[0].axes == {}
    assert specs[0].acquisition.optics.na == base.acquisition.optics.na


def test_base_spec_is_not_mutated():
    base = _base()
    list(sweep(base, {
        "acquisition.optics.na": [0.1, 0.2],
        "steps.place_neurons.density_per_mm2": [1.0],
    }))
    assert base.acquisition.optics.na == 0.45
    assert _place(base).density_per_mm2 == 2500.0


def test_cache_key_parity_axes_excluded():
    # a swept spec hashes identically to a hand-built plain spec with the same
    # physical content -> the axes tag never perturbs cache dedup
    (s,) = list(sweep(_base(), {"acquisition.optics.na": [0.3]}))
    plain = _base().model_copy(
        update={"acquisition": _base().acquisition.model_copy(
            update={"optics": _base().acquisition.optics.model_copy(update={"na": 0.3})}
        )}
    )
    assert s.cache_key() == Spec.model_validate(plain.model_dump()).cache_key()
    assert "axes" not in s.model_dump_json()


def test_invalid_axis_value_fails_validation():
    # NA must be > 0; an out-of-range sweep value is caught when the spec re-validates
    with pytest.raises(Exception):
        list(sweep(_base(), {"acquisition.optics.na": [-1.0]}))


def test_unknown_field_path_raises():
    with pytest.raises(ValueError, match="unknown field"):
        list(sweep(_base(), {"acquisition.optics.nope": [1.0]}))


def test_unknown_step_kind_raises():
    with pytest.raises(ValueError, match="no step of kind"):
        list(sweep(_base(), {"steps.not_a_step.foo": [1.0]}))


def test_scalar_descent_raises():
    with pytest.raises(ValueError, match="non-model field"):
        list(sweep(_base(), {"seed.deeper": [1]}))


def test_malformed_steps_path_raises():
    with pytest.raises(ValueError, match="steps.<kind>.<field>"):
        list(sweep(_base(), {"steps": [1]}))


def test_yielded_spec_simulates_end_to_end():
    (s,) = list(sweep(_base(), {"steps.place_neurons.density_per_mm2": [2000.0]}))
    rec = simulate(s)
    assert rec.observed.shape == (s.acquisition.n_frames, 64, 64)
    assert rec.ground_truth.n_units > 0
    np.testing.assert_array_equal(rec.observed, np.round(rec.observed))  # sensor counts
