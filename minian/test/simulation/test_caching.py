"""Unit tests for local caching (migration Step 7).

Covers the zarr persistence layer — ``Recording.save``/``load`` round-trips
(observed movie, every GroundTruth field, optional-field presence, heterogeneous
snapshots, empty ground truth, atomic overwrite, the spec-hash integrity check) —
and the thin ``simulate_cached`` wrapper (miss-writes / hit-reads, cache-dir
resolution).
"""

import numpy as np
import pytest

import minian.simulation.cache as cache_mod
from minian.simulation import (
    Acquisition,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    GroundTruth,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Recording,
    Render,
    Sensor,
    Spec,
    Vignette,
    cache_dir,
    cache_path,
    simulate,
    simulate_cached,
)


def _acq():
    """64×64 sensor at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    return Acquisition(
        fps=20.0,
        duration_s=1.0,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(n_px_height=64, n_px_width=64, pixel_pitch_um=8.0, bit_depth=8),
    )


def _minimal_spec(seed=7, **output_kw):
    """A short all-cells-then-sensor spec: no motion, every optional GT field None."""
    return Spec(
        acquisition=_acq(),
        seed=seed,
        steps=[
            PlaceNeurons(density_per_mm3=312500.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Sensor(photons_per_unit=100.0),
        ],
        output=Output(**output_kw),
    )


def _full_spec(seed=11, **output_kw):
    """Every effect present, so all optional GroundTruth fields are populated."""
    return Spec(
        acquisition=_acq(),
        seed=seed,
        steps=[
            PlaceNeurons(density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 100.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            Neuropil(n_components=2),
            Bleaching(final_fraction=0.7),
            BrainMotion(walk_step_um=0.3, max_shift_um=2.0),
            Vignette(falloff=0.6),
            Leakage(profile="gaussian", level=0.1),
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(**output_kw),
    )


def test_save_load_roundtrips_observed(tmp_path):
    rec = simulate(_minimal_spec())
    rec.save(tmp_path / "r.zarr")
    back = Recording.load(tmp_path / "r.zarr")
    np.testing.assert_array_equal(back.observed, rec.observed)
    assert back.observed.dtype == rec.observed.dtype


def test_save_load_roundtrips_required_ground_truth(tmp_path):
    rec = simulate(_minimal_spec())
    rec.save(tmp_path / "r.zarr")
    gt, back = rec.ground_truth, Recording.load(tmp_path / "r.zarr").ground_truth
    assert back.n_units == gt.n_units
    for name in ("A_planted", "A_observed", "C", "S", "centers_um", "amplitude_per_cell"):
        np.testing.assert_array_equal(getattr(back, name), getattr(gt, name))
    # bool masks survive as bool, not upcast to int/float
    for name in ("in_focus", "detectable"):
        np.testing.assert_array_equal(getattr(back, name), getattr(gt, name))
        assert getattr(back, name).dtype == np.bool_


def test_minimal_recording_keeps_optional_fields_none(tmp_path):
    rec = simulate(_minimal_spec())
    rec.save(tmp_path / "r.zarr")
    back = Recording.load(tmp_path / "r.zarr").ground_truth
    for name in ("shifts", "vignette", "leakage", "bleaching", "neuropil_temporal", "neuropil_spatial"):
        assert getattr(back, name) is None


def test_save_load_roundtrips_optional_fields_present(tmp_path):
    rec = simulate(_full_spec())
    rec.save(tmp_path / "r.zarr")
    gt, back = rec.ground_truth, Recording.load(tmp_path / "r.zarr").ground_truth
    for name in ("shifts", "vignette", "leakage", "bleaching", "neuropil_temporal", "neuropil_spatial"):
        assert getattr(back, name) is not None, name
        np.testing.assert_array_equal(getattr(back, name), getattr(gt, name))


def test_save_load_roundtrips_snapshots(tmp_path):
    rec = simulate(_full_spec(save_intermediates=True))
    rec.save(tmp_path / "r.zarr")
    back = Recording.load(tmp_path / "r.zarr")
    assert set(back.snapshots) == set(rec.snapshots)
    for name, snap in rec.snapshots.items():
        np.testing.assert_array_equal(back.snapshots[name].values, snap.values)
        assert back.snapshots[name].dims == snap.dims
    # the larger tissue-canvas stages and the cropped post-motion stages both survive
    assert back.stage("cells_only").shape[1] > back.stage("sensor").shape[1]


def test_no_snapshots_roundtrips_empty(tmp_path):
    rec = simulate(_minimal_spec())  # save_intermediates defaults False
    rec.save(tmp_path / "r.zarr")
    assert Recording.load(tmp_path / "r.zarr").snapshots == {}


def test_empty_ground_truth_roundtrips(tmp_path):
    spec = _minimal_spec()
    nf, h, w = spec.acquisition.n_frames, 64, 64
    gt = GroundTruth(
        A_planted=np.zeros((0, h, w)),
        A_observed=np.zeros((0, h, w)),
        C=np.zeros((0, nf)),
        S=np.zeros((0, nf)),
        centers_um=np.zeros((0, 3)),
        amplitude_per_cell=np.zeros((0,)),
        in_focus=np.zeros((0,), dtype=bool),
        detectable=np.zeros((0,), dtype=bool),
    )
    rec = Recording(spec=spec, observed=np.zeros((nf, h, w), dtype=np.float32), ground_truth=gt)
    rec.save(tmp_path / "r.zarr")
    back = Recording.load(tmp_path / "r.zarr")
    assert back.ground_truth.n_units == 0
    assert back.ground_truth.A_planted.shape == (0, h, w)


def test_save_overwrites_and_leaves_no_tmp(tmp_path):
    rec = simulate(_minimal_spec())
    path = tmp_path / "r.zarr"
    rec.save(path)
    rec.save(path)  # second write overwrites the first
    assert not path.with_name(path.name + ".tmp").exists()
    np.testing.assert_array_equal(Recording.load(path).observed, rec.observed)


def test_load_rejects_spec_hash_mismatch(tmp_path):
    rec = simulate(_minimal_spec(seed=7))
    path = tmp_path / "r.zarr"
    rec.save(path)
    # rewrite spec.json with a *different* valid spec -> its hash no longer matches
    # the spec_cache_key stamped in the group attrs at save time
    (path / "spec.json").write_text(_minimal_spec(seed=999).model_dump_json(indent=2))
    with pytest.raises(ValueError, match="hash mismatch"):
        Recording.load(path)


def test_simulate_cached_writes_on_miss_reads_on_hit(tmp_path, monkeypatch):
    spec = _minimal_spec()
    calls = {"n": 0}
    real = cache_mod.simulate

    def counting(s):
        calls["n"] += 1
        return real(s)

    monkeypatch.setattr(cache_mod, "simulate", counting)

    first = simulate_cached(spec, root=tmp_path)
    assert (tmp_path / f"{spec.cache_key()}.zarr").exists()
    assert calls["n"] == 1

    second = simulate_cached(spec, root=tmp_path)  # served from disk, no re-sim
    assert calls["n"] == 1
    np.testing.assert_array_equal(second.observed, first.observed)


def test_simulate_cached_distinct_specs_get_distinct_entries(tmp_path):
    a, b = _minimal_spec(seed=1), _minimal_spec(seed=2)
    simulate_cached(a, root=tmp_path)
    simulate_cached(b, root=tmp_path)
    assert (tmp_path / f"{a.cache_key()}.zarr").exists()
    assert (tmp_path / f"{b.cache_key()}.zarr").exists()
    assert a.cache_key() != b.cache_key()


def test_cache_dir_honors_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIAN_SIM_CACHE", str(tmp_path / "sim"))
    assert cache_dir() == tmp_path / "sim"
    spec = _minimal_spec()
    assert cache_path(spec) == tmp_path / "sim" / f"{spec.cache_key()}.zarr"
