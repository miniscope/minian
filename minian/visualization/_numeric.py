"""Pure numeric / array helpers used by viewers and pipeline plots."""

import logging
from typing import Tuple

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy import linalg
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def construct_G(g: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Construct a convolving matrix from AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        Input AR coefficients.
    T : np.ndarray
        Number of time samples of the AR process.

    Returns
    -------
    G : np.ndarray
        A `T` x `T` matrix that can be used to multiply with a timeseries to
        convolve the AR process.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    cur_c, cur_r = np.zeros(T), np.zeros(T)
    cur_c[0] = 1
    cur_r[0] = 1
    cur_c[1 : len(g) + 1] = -g
    return linalg.toeplitz(cur_c, cur_r)


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize an input array to range (0, 1) using :func:`numpy.interp`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    a_norm : np.ndarray
        Normalized array.
    """
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, +1))


def normalize_along_frame_per_unit(
    array: xr.DataArray, *, output_dtype: np.dtype | None = None
) -> xr.DataArray:
    """
    Normalize each ``unit_id`` trace along ``frame`` (vectorized, dask-parallelized).

    Used by interactive viewers that need per-unit min–max scaling on calcium /
    spike traces.
    """
    dt = output_dtype if output_dtype is not None else array.dtype
    return xr.apply_ufunc(
        normalize,
        array.chunk(dict(frame=-1, unit_id="auto")),
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[dt],
    )


def norm(a: np.ndarray) -> np.ndarray:
    """
    Normalize an input array to range (0, 1) avoiding division-by-zero.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    a_norm : np.ndarray
        Normalized array. If there is only one unique value in `a` then it is
        returned unchanged.
    """
    amax = np.nanmax(a)
    amin = np.nanmin(a)
    diff = amax - amin
    if diff > 0:
        return (a - amin) / (amax - amin)
    else:
        return a


def convolve_G(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Convolve an AR process to input timeseries.

    Despite the name, only AR coefficients are needed as input. The convolving
    matrix will be computed using :func:`construct_G`.

    Parameters
    ----------
    s : np.ndarray
        The input timeseries, presumably representing spike signals.
    g : np.ndarray
        The AR coefficients.

    Returns
    -------
    c : np.ndarray
        Convolved timeseries, presumably representing calcium dynamics.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    G = construct_G(g, len(s))
    try:
        c = np.linalg.inv(G).dot(s)
    except np.linalg.LinAlgError:
        c = s.copy()
    return c


def construct_pulse_response(
    g: np.ndarray, length=500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a model pulse response corresponding to certain AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        The AR coefficients.
    length : int, optional
        Number of timepoints in output. By default `500`.

    Returns
    -------
    s : np.ndarray
        Model spike with shape `(length,)`, zero everywhere except the first
        timepoint.
    c : np.ndarray
        Model convolved calcium response, with same shape as `s`.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    s = np.zeros(length)
    s[np.arange(0, length, 500)] = 1
    c = convolve_G(s, g)
    return s, c


def centroid(A: xr.DataArray, verbose=False) -> pd.DataFrame:
    """
    Compute centroids of spatial footprint of each cell.

    Parameters
    ----------
    A : xr.DataArray
        Input spatial footprints.
    verbose : bool, optional
        Whether to print message and progress bar. By default `False`.

    Returns
    -------
    cents_df : pd.DataFrame
        Centroid of spatial footprints for each cell. Has columns "unit_id",
        "height", "width" and any other additional metadata dimension.
    """

    def rel_cent(im):
        im_nan = np.isnan(im)
        if im_nan.all():
            return np.array([np.nan, np.nan])
        if im_nan.any():
            im = np.nan_to_num(im)
        cent = np.array(center_of_mass(im))
        return cent / im.shape

    gu_rel_cent = da.gufunc(
        rel_cent,
        signature="(h,w)->(d)",
        output_dtypes=float,
        output_sizes=dict(d=2),
        vectorize=True,
    )
    cents = xr.apply_ufunc(
        gu_rel_cent,
        A.chunk(dict(height=-1, width=-1)),
        input_core_dims=[["height", "width"]],
        output_core_dims=[["dim"]],
        dask="allowed",
    ).assign_coords(dim=["height", "width"])
    if verbose:
        log.info("computing centroids")
        cents = cents.compute()
    cents_df = (
        cents.rename("cents")
        .to_series()
        .dropna()
        .unstack("dim")
        .rename_axis(None, axis="columns")
        .reset_index()
    )
    h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
    w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
    cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
    cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
    return cents_df


def NNsort(cents: pd.DataFrame) -> pd.Series:
    """
    Sort centroids of cells into close-by groups.

    Walk through centroids of cells using a nearest neighbors tree such that the
    resulting walk order can be used to sort cells into close-by groups.

    Parameters
    ----------
    cents : pd.DataFrame
        Input centroids of cells. Should contain column "height" and "width".

    Returns
    -------
    result : pd.Series
        A series with same index as input `cents` whose values represent the
        order of nearest-neighbor walk.
    """
    cents_hw = cents[["height", "width"]]
    kdtree = cKDTree(cents_hw)
    idu_start = cents_hw.sum(axis="columns").idxmin()
    result = pd.Series(0, index=cents.index)
    remain_list = cents.index.tolist()
    idu_next = idu_start
    NNord = 0
    while remain_list:
        result.loc[idu_next] = NNord
        remain_list.remove(idu_next)
        for k in range(1, int(np.ceil(np.log2(len(result)))) + 1):
            qry = kdtree.query(cents_hw.loc[idu_next], 2**k)
            NNs = qry[1][np.isfinite(qry[0])].squeeze()
            NNs = NNs[np.sort(np.unique(NNs, return_index=True)[1])]
            NNs = np.array(result.iloc[NNs].index)
            NN_idxs = np.argwhere(np.isin(NNs, remain_list, assume_unique=True))
            if len(NN_idxs) > 0:
                NN = NNs[NN_idxs[0]][0]
                idu_next = NN
                NNord = NNord + 1
                break
    return result
