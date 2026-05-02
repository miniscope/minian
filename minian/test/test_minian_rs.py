"""Tests for Rust extension ``minian.minian_rs`` (built from ``src-rust``)."""

import numpy as np
import pytest

from minian.cnmf import legacy


def _rust():
    """Skip whole module if wheel was built without the extension."""
    return pytest.importorskip("minian.minian_rs")


def test_filt_fft_f64_matches_legacy_pyfftw_reference():
    """Rust 1-D path agrees with legacy PyFFTW reference implementation."""
    rs = _rust()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128).astype(np.float64)
    freq = 0.07
    for btype in ("low", "high"):
        want = legacy.filt_fft(x.copy(), freq, btype)
        got = np.asarray(rs.filt_fft_f64(np.ascontiguousarray(x), float(freq), btype))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)


def test_filt_fft_cutoff_above_nyquist_matches_legacy():
    """When ``int(freq * T)`` exceeds ``len(rfft(x))``, match NumPy slice clipping (no panic)."""
    rs = _rust()
    rng = np.random.default_rng(42)
    t = 1001
    x = rng.standard_normal(t).astype(np.float64)
    freq = 1.2  # freq * t > len(rfft(x)) ≈ t // 2 + 1
    for btype in ("low", "high"):
        want = legacy.filt_fft(x.copy(), freq, btype)
        got = np.asarray(rs.filt_fft_f64(np.ascontiguousarray(x), float(freq), btype))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)


def test_filt_fft_vec_f64_matches_legacy_reference():
    """Rust row-major 2-D path agrees with looping legacy."""
    rs = _rust()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 96)).astype(np.float64)
    freq = 0.11
    for btype in ("low", "high"):
        want = legacy.filt_fft_vec(np.ascontiguousarray(x), freq, btype)
        got = np.asarray(
            rs.filt_fft_vec_f64(np.ascontiguousarray(x), float(freq), btype, False)
        )
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)


def test_parallel_flag_matches_serial_for_vec():
    """Rayon parallel path matches sequential path."""
    rs = _rust()
    x = np.linspace(-1.0, 1.0, 60).astype(np.float64).reshape((3, 20))
    a = np.asarray(rs.filt_fft_vec_f64(np.ascontiguousarray(x), 0.1, "low", False))
    b = np.asarray(rs.filt_fft_vec_f64(np.ascontiguousarray(x), 0.1, "low", True))
    np.testing.assert_array_equal(a, b)


def test_invalid_btype_raises():
    rs = _rust()
    x = np.ones(16, dtype=np.float64)
    with pytest.raises(Exception):
        rs.filt_fft_f64(x, 0.1, "bandpass")


def test_empty_1d_returns_empty():
    rs = _rust()
    got = rs.filt_fft_f64(np.array([], dtype=np.float64), 0.05, "low")
    assert np.asarray(got).shape == (0,)


def test_thread_allocation_matches_default_cluster_workers():
    rs = _rust()
    ta = rs.thread_allocation(2)
    assert int(ta.cluster_workers) == int(rs.default_cluster_workers(2))
    assert int(ta.logical_cpus) == int(rs.logical_parallelism())
    assert int(ta.after_reserve_cpus) == max(0, int(rs.logical_parallelism()) - 2)
    assert abs(float(ta.worker_cpu_ratio) - 2.0 / 3.0) < 1e-9


def test_thread_allocation_one_third_matches_legacy_integer_workers():
    """``ratio=1/3`` reproduces the former ``(cpus - reserve) // 3`` worker count."""
    rs = _rust()
    n = int(rs.logical_parallelism())
    for reserve in (0, 1, 2):
        after = max(0, n - reserve)
        legacy = max(1, after // 3)
        ta = rs.thread_allocation(reserve, 1.0 / 3.0)
        assert int(ta.cluster_workers) == legacy
