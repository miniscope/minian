"""Spectral and Butterworth filtering for CNMF."""

import logging

import numpy as np
import xarray as xr
from scipy.signal import butter, lfilter

from . import legacy

log = logging.getLogger(__name__)

try:
    from minian.minian_rs import filt_fft_f64 as _rust_filt_fft
    from minian.minian_rs import filt_fft_vec_f64 as _rust_filt_fft_vec

    log.info("Using Rust extension (filt_fft_f64, filt_fft_vec_f64)")
except ImportError:
    _rust_filt_fft = None
    _rust_filt_fft_vec = None
    log.info("Using Python fallback (filt_fft, filt_fft_vec)")


def smooth_sig(
    sig: xr.DataArray, freq: float, method="fft", btype="low"
) -> xr.DataArray:
    """
    Filter the input timeseries with a cut-off frequency in vecorized fashion.

    Parameters
    ----------
    sig : xr.DataArray
        The input timeseries. Should have dimension "frame".
    freq : float
        The cut-off frequency.
    method : str, optional
        Method used for filtering. Either `"fft"` or `"butter"`. If `"fft"`, the
        filtering is carried out with zero-ing fft signal. If `"butter"`, the
        fiilterings carried out with :func:`scipy.signal.butter`. By default
        "fft".
    btype : str, optional
        Either `"low"` or `"high"` specify low or high pass filtering. By
        default `"low"`.

    Returns
    -------
    sig_smth : xr.DataArray
        The filtered signal. Has same shape as input `sig`.

    Raises
    ------
    NotImplementedError
        if `method` is not "fft" or "butter"
    """
    try:
        filt_func = {"fft": filt_fft, "butter": filt_butter}[method]
    except KeyError:
        raise NotImplementedError(method)
    sig_smth = xr.apply_ufunc(
        filt_func,
        sig,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        kwargs={"btype": btype, "freq": freq},
        dask="parallelized",
        output_dtypes=[sig.dtype],
    )
    return sig_smth


def filt_fft(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries by zero-ing bands in the fft signal.

    Uses ``minian.minian_rs.filt_fft_f64`` when the extension imports; otherwise
    :func:`legacy.filt_fft` (PyFFTW).
    """
    if _rust_filt_fft is None:
        return legacy.filt_fft(x, freq, btype)

    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.asarray(_rust_filt_fft(xc, float(freq), btype), dtype=np.float64)
    return out.astype(x.dtype, copy=False)


def filt_butter(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries with Butterworth filter using
    :func:`scipy.signal.butter`.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either "low" or "high" specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    but_b, but_a = butter(2, freq * 2, btype=btype, analog=False)
    return lfilter(but_b, but_a, x)


def filt_fft_vec(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Vectorized FFT band-limiting along rows (last axis).

    Uses ``minian.minian_rs.filt_fft_vec_f64`` if the extension imports;
    otherwise :func:`legacy.filt_fft_vec` (PyFFTW row loop).
    """
    if _rust_filt_fft_vec is None:
        return legacy.filt_fft_vec(x, freq, btype)

    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.asarray(_rust_filt_fft_vec(xc, float(freq), btype, True), dtype=np.float64)
    if out.shape != x.shape:
        raise ValueError(
            "filt_fft_vec: shape mismatch {} vs {}".format(out.shape, x.shape)
        )
    return out.astype(x.dtype, copy=False)
