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
