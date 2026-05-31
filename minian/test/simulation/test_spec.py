"""Unit tests for the ``minian.simulation`` spec surface (migration Step 2).

Covers the model contract (``extra="forbid"``, ``frozen``), the unit
conversions on ``Acquisition``/``Optics``, JSON round-tripping through the
static ``AnyStep`` discriminated union, ``cache_key`` behavior, and the §11
cross-field validators (hard fails and advisory warnings).
"""

import pytest
from pydantic import ValidationError

from minian.simulation import (
    Acquisition,
    BrainMotion,
    CellActivity,
    CellOptics,
    ImageSensor,
    Optics,
    Output,
    PlaceSomata,
    Render,
    Sensor,
    SNRDistribution,
    Spec,
    SpecWarning,
    Tissue,
)


def _minimal_steps():
    """A short, in-order, individually-valid step list for a tiny FOV."""
    return [
        PlaceSomata(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        CellOptics(),
        Render(),
        Sensor(),
    ]


def _tiny_acquisition(**kw):
    """32×32 sensor → 12 µm FOV at the default 0.375 µm pixel size."""
    kw.setdefault("fps", 20.0)
    kw.setdefault("duration_s", 25.0)
    kw.setdefault("image_sensor", ImageSensor(n_px_height=32, n_px_width=32))
    return Acquisition(**kw)


def _valid_spec(**kw):
    kw.setdefault("acquisition", _tiny_acquisition())
    kw.setdefault("steps", _minimal_steps())
    return Spec(**kw)


# --- model contract --------------------------------------------------------


def test_extra_forbidden():
    with pytest.raises(ValidationError):
        Optics(nope=1)


def test_models_are_frozen():
    opt = Optics()
    with pytest.raises(ValidationError):
        opt.na = 0.9


def test_defaults_construct():
    spec = _valid_spec()
    assert spec.seed == 42
    assert isinstance(spec.acquisition.optics, Optics)
    assert isinstance(spec.acquisition.image_sensor, ImageSensor)
    assert isinstance(spec.acquisition.tissue, Tissue)
    assert isinstance(spec.output, Output)


def test_sensor_hardware_lives_on_image_sensor():
    # Hardware moved off the Sensor step onto Acquisition.image_sensor; the step
    # keeps only the (non-hardware) exposure scale.
    assert ImageSensor(read_noise_e=3.0, quantum_efficiency=0.8, bit_depth=12).bit_depth == 12
    assert Sensor(photons_per_unit=80.0).photons_per_unit == pytest.approx(80.0)
    with pytest.raises(ValidationError):
        Sensor(read_noise_e=3.0)  # hardware no longer accepted on the step


# --- unit conversions ------------------------------------------------------


def test_pixel_size_um():
    # Pixel size is the joint optics×sensor quantity, owned by Acquisition.
    acq = Acquisition(
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(pixel_pitch_um=3.0),
    )
    assert acq.pixel_size_um == pytest.approx(0.375)


def test_n_frames_rounds():
    assert Acquisition(fps=20.0, duration_s=25.0).n_frames == 500
    assert Acquisition(fps=30.0, duration_s=1.05).n_frames == round(31.5)  # 32


def test_fov_um_derived():
    acq = _tiny_acquisition()
    assert acq.fov_um == pytest.approx((12.0, 12.0))


def test_um_to_px_and_s_to_frame():
    acq = _tiny_acquisition()
    assert acq.um_to_px(0.75) == pytest.approx(2.0)
    assert acq.s_to_frame(1.0) == pytest.approx(20.0)


# --- discriminated union + serialization -----------------------------------


def test_json_round_trip_preserves_step_types():
    spec = _valid_spec()
    restored = Spec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert isinstance(restored.steps[0], PlaceSomata)
    assert isinstance(restored.steps[-1], Sensor)


def test_union_discriminates_on_kind():
    spec = Spec.model_validate(
        {
            "acquisition": _tiny_acquisition().model_dump(),
            "steps": [{"kind": "render"}, {"kind": "sensor"}],
        }
    )
    assert isinstance(spec.steps[0], Render)
    assert isinstance(spec.steps[1], Sensor)


# --- cache key -------------------------------------------------------------


def test_cache_key_stable_and_content_sensitive():
    a = _valid_spec()
    b = _valid_spec()
    assert a.cache_key() == b.cache_key()
    assert _valid_spec(seed=43).cache_key() != a.cache_key()


# --- validators: hard fails ------------------------------------------------


def test_duplicate_kind_fails():
    with pytest.raises(ValidationError, match="unique"):
        _valid_spec(steps=[Render(), Render()])


def test_soma_larger_than_fov_fails():
    # 20 µm radius → 40 µm diameter, far larger than the 12 µm FOV.
    with pytest.raises(ValidationError, match="FOV"):
        _valid_spec(steps=[PlaceSomata(soma_radius_um=20.0)] + _minimal_steps()[1:])


def test_unresolvable_decay_fails():
    acq = _tiny_acquisition(fps=1.0)
    steps = [PlaceSomata(soma_radius_um=3.0), CellActivity(tau_decay_s=0.5), Render()]
    with pytest.raises(ValidationError, match="unresolvable"):
        Spec(acquisition=acq, steps=steps)


def test_snr_order_fails():
    with pytest.raises(ValidationError):
        SNRDistribution(low=5.0, high=2.0)


# --- validators: advisory warnings -----------------------------------------


def test_out_of_order_domains_warn():
    # sensor before render → sensor(rank3) precedes tissue(rank1)
    steps = [PlaceSomata(soma_radius_um=3.0), Sensor(), Render()]
    with pytest.warns(SpecWarning, match="natural"):
        _valid_spec(steps=steps)


def test_focal_plane_out_of_range_warns():
    acq = _tiny_acquisition(optics=Optics(focal_plane_um=500.0))
    steps = [PlaceSomata(soma_radius_um=3.0, depth_range_um=(0.0, 200.0)), Render()]
    with pytest.warns(SpecWarning, match="focal"):
        Spec(acquisition=acq, steps=steps)


def test_auto_focal_plane_does_not_warn(recwarn):
    _valid_spec()  # default optics → focal_plane_um="auto"
    assert not [w for w in recwarn.list if issubclass(w.category, SpecWarning)]


def test_large_motion_warns():
    steps = [
        PlaceSomata(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        CellOptics(),
        Render(),
        BrainMotion(max_shift_um=50.0),  # ≫ 5% of the 12 µm FOV
        Sensor(),
    ]
    with pytest.warns(SpecWarning, match="Motion"):
        _valid_spec(steps=steps)


# --- build() is not yet implemented (Step 5) -------------------------------


def test_build_not_implemented():
    with pytest.raises(NotImplementedError, match="Step 5"):
        Render().build(_tiny_acquisition(), None)
