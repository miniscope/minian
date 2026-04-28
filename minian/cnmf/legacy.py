"""PyFFTW-backed FFT band-limiting (reference / fallback path).

Used when ``minian.minian_rs`` is not installed or when you call
:func:`legacy.filt_fft` / :func:`legacy.filt_fft_vec` directly.

This matches the original Minian behavior using :mod:`pyfftw.interfaces.numpy_fft`.
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as numpy_fft


def filt_fft(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries by zero-ing bands in the fft signal (PyFFTW / numpy_fft).

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either ``"low"`` or ``"high"`` for low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    _T = len(x)
    if btype == "low":
        zero_range = slice(int(freq * _T), None)
    elif btype == "high":
        zero_range = slice(None, int(freq * _T))
    else:
        raise ValueError("btype must be 'low' or 'high'")
    xfft = numpy_fft.rfft(x)
    xfft[zero_range] = 0
    return numpy_fft.irfft(xfft, len(x))


def filt_fft_vec(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Row-wise :func:`filt_fft` along the last axis (pure PyFFTW path).
    """
    out = np.empty_like(x)
    for ix, xx in enumerate(x):
        out[ix, :] = filt_fft(xx, freq, btype)
    return out
