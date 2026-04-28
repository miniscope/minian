"""Local extrema, baseline removal, sparse least-squares along rows."""

import cv2
import dask.array as darr
import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr


def local_extreme(fm: np.ndarray, k: np.ndarray, etype="max", diff=0) -> np.ndarray:
    """
    Find local extreme of a 2d array.

    Parameters
    ----------
    fm : np.ndarray
        The input 2d array.
    k : np.ndarray
        Structuring element defining the locality of the result, passed as
        `kernel` to :func:`cv2.erode` and :func:`cv2.dilate`.
    etype : str, optional
        Type of local extreme. Either `"min"` or `"max"`. By default `"max"`.
    diff : int, optional
        Threshold of difference between local extreme and its neighbours. By
        default `0`.

    Returns
    -------
    fm_ext : np.ndarray
        The returned 2d array whose non-zero elements represent the location of
        local extremes.

    Raises
    ------
    ValueError
        if `etype` is not "min" or "max"
    """
    fm_max = cv2.dilate(fm, k)
    fm_min = cv2.erode(fm, k)
    fm_diff = ((fm_max - fm_min) > diff).astype(np.uint8)
    if etype == "max":
        fm_ext = (fm == fm_max).astype(np.uint8)
    elif etype == "min":
        fm_ext = (fm == fm_min).astype(np.uint8)
    else:
        raise ValueError("Don't understand {}".format(etype))
    return cv2.bitwise_and(fm_ext, fm_diff).astype(np.uint8)


def med_baseline(a: np.ndarray, wnd: int) -> np.ndarray:
    """
    Subtract baseline from a timeseries as estimated by median-filtering the
    timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    wnd : int
        Window size of the median filter. This parameter is passed as `size` to
        :func:`scipy.ndimage.filters.median_filter`.

    Returns
    -------
    a : np.ndarray
        Timeseries with baseline subtracted.
    """
    base = median_filter(a, size=wnd)
    a -= base
    return a.clip(0, None)


@darr.as_gufunc(signature="(m,n),(m)->(n)", output_dtypes=float)
def sps_lstsq(a: csc_matrix, b: np.ndarray, **kwargs):
    out = np.zeros((b.shape[0], a.shape[1]))
    for i in range(b.shape[0]):
        out[i, :] = lsqr(a, b[i, :].squeeze(), **kwargs)[0]
    return out
