"""CNMF decomposition and helpers (combined module).
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as numpy_fft
import xarray as xr
from distributed import get_client
from scipy.signal import welch




def get_noise_fft(
    varr: xr.DataArray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> xr.DataArray:
    """
    Estimates noise along the "frame" dimension aggregating power spectral
    density within `noise_range`.

    This function compute a Fast Fourier transform (FFT) along the "frame"
    dimension in a vectorized fashion, and estimate noise by aggregating its
    power spectral density (PSD). Note that `noise_range` is specified relative
    to the sampling frequency, so 0.5 represents the Nyquist frequency. Three
    `noise_method` are availabe for aggregating the psd: "mean" and "median"
    will use the mean and median across all frequencies as the estimation of
    noise. "logmexp" takes the mean of the logarithmic psd, then transform it
    back with an exponential function.

    Parameters
    ----------
    varr : xr.DataArray
        Input data, should have a "frame" dimension.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.

    Returns
    -------
    sn : xr.DataArray
        Spectral density of the noise. Same shape as `varr` with the "frame"
        dimension removed.
    """
    try:
        clt = get_client()
        threads = min(clt.nthreads().values())
    except ValueError:
        threads = 1
    sn = xr.apply_ufunc(
        noise_fft,
        varr,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(
            noise_range=noise_range, noise_method=noise_method, threads=threads
        ),
        output_dtypes=[np.float64],
    )
    return sn


def noise_fft(
    px: np.ndarray, noise_range=(0.25, 0.5), noise_method="logmexp", threads=1
) -> float:
    """
    Estimates noise of the input by aggregating power spectral density within
    `noise_range`.

    The PSD is estimated using FFT.

    Parameters
    ----------
    px : np.ndarray
        Input data.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default "logmexp".
    threads : int, optional
        Number of threads to use for pyfftw. By default `1`.

    Returns
    -------
    noise : float
        The estimated noise level of input.

    See Also
    -------
    get_noise_fft
    """
    _T = len(px)
    nr = np.around(np.array(noise_range) * _T).astype(int)
    px = 1 / _T * np.abs(numpy_fft.rfft(px, threads=threads)[nr[0] : nr[1]]) ** 2
    if noise_method == "mean":
        return np.sqrt(px.mean())
    elif noise_method == "median":
        return np.sqrt(px.median())
    elif noise_method == "logmexp":
        eps = np.finfo(px.dtype).eps
        return np.sqrt(np.exp(np.log(px + eps).mean()))
    elif noise_method == "sum":
        return np.sqrt(px.sum())


def get_noise_welch(
    varr: xr.DataArray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> xr.DataArray:
    """
    Estimates noise along the "frame" dimension aggregating power spectral
    density within `noise_range`.

    The PSD is estimated using welch method as an alternative to FFT. The welch
    method assumes the noise in the signal to be a stochastic process and
    attenuates noise by windowing the original signal into segments and
    averaging over them.

    Parameters
    ----------
    varr : xr.DataArray
        Input data. Should have a "frame" dimension.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.

    Returns
    -------
    sn : xr.DataArray
        Spectral density of the noise. Same shape as `varr` with the "frame"
        dimension removed.

    See Also
    -------
    get_noise_fft : For more details on the parameters.
    """
    sn = xr.apply_ufunc(
        noise_welch,
        varr.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[varr.dtype],
    )
    return sn


def noise_welch(
    y: np.ndarray, noise_range=(0.25, 0.5), noise_method="logmexp"
) -> float:
    """
    Estimates noise of the input by aggregating power spectral density within
    `noise_range`.

    The PSD is estimated using welch method.

    Parameters
    ----------
    px : np.ndarray
        Input data.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default `"logmexp"`.
    threads : int, optional
        Number of threads to use for pyfftw. By default `1`.

    Returns
    -------
    noise : float
        The estimated noise level of input.

    See Also
    -------
    get_noise_welch
    """
    ff, Pxx = welch(y)
    mask0, mask1 = ff > noise_range[0], ff < noise_range[1]
    mask = np.logical_and(mask0, mask1)
    Pxx_ind = Pxx[mask]
    sn = {
        "mean": lambda x: np.sqrt(np.mean(x / 2)),
        "median": lambda x: np.sqrt(np.median(x / 2)),
        "logmexp": lambda x: np.sqrt(np.exp(np.mean(np.log(x / 2)))),
    }[noise_method](Pxx_ind)
    return sn
