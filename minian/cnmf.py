import functools as fct
import os
import warnings
from typing import List, Optional, Tuple, Union

import cv2
import cvxpy as cvx
import dask as da
import dask.array as darr
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
import pyfftw.interfaces.numpy_fft as numpy_fft
import pymetis
import scipy.sparse
import sparse
import xarray as xr
import zarr
from distributed import get_client
from scipy.linalg import lstsq, toeplitz
from scipy.ndimage import label
from scipy.signal import butter, lfilter, welch
from scipy.sparse import dia_matrix
from skimage import morphology as moph
from sklearn.linear_model import LassoLars
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acovf

from .utilities import (
    custom_arr_optimize,
    custom_delay_optimize,
    open_minian,
    rechunk_like,
    save_minian,
    med_baseline,
)


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
        output_dtypes=[np.float],
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


def update_spatial(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    sn: xr.DataArray,
    b: xr.DataArray = None,
    f: xr.DataArray = None,
    dl_wnd=5,
    sparse_penal=0.5,
    update_background=False,
    normalize=True,
    size_thres=(9, None),
    in_memory=False,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Update spatial components given the input data and temporal dynamic for each
    cell.

    This function carries out spatial update of the CNMF algorithm. The update
    is done in parallel and independently for each pixel. To save computation
    time, we compute a subsetting matrix `sub` by dilating the initial
    spatial foorprint of each cell. The window size of the dilation is
    controled by `dl_wnd`. Then for each pixel, only cells that have a non-zero
    value in `sub` at the current pixel will be considered for update.
    Optionally, the spatial footprint of the background can be updated in the
    same fashion based on the temporal dynamic of the background. After the
    update, the spatial footprint of each cell can be optionally noramlized to
    unit sum, so that difference in fluorescent intensity will not be reflected
    in spatial footprint. A `size_thres` can be passed in to filter out cells
    whose size (number of non-zero values in spatial footprint) is outside the
    specified range. Finally, the temporal dynamic of cells `C` can either be
    load in memory before the update or lazy-loaded during the update. Note that
    if `in_memory` is `False`, then `C` must be stored under the intermediate
    folder specified as environment variable `MINIAN_INTERMEDIATE`.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    A : xr.DataArray
        Previous estimation of spatial footprints. Should have dimension
        "height", "width" and "unit_id".
    C : xr.DataArray
        Estimation of temporal component for each cell. Should have dimension
        "frame" and "unit_id".
    sn : xr.DataArray
        Estimation of noise level for each pixel. Should have dimension "height"
        and "width".
    b : xr.DataArray, optional
        Previous estimation of spatial footprint of background. Fhould have
        dimension "height" and "width".
    f : xr.DataArray, optional
        Estimation of temporal dynamic of background. Should have dimension
        "frame".
    dl_wnd : int, optional
        Window of morphological dilation in pixel when computing the subsetting
        matrix. By default `5`.
    sparse_penal : float, optional
        Global scalar controlling sparsity of the result. The higher the value,
        the sparser the spatial footprints. By default `0.5`.
    update_background : bool, optional
        Whether to update the spatial footprint of background. If `True`, then
        both `b` and `f` need to be provided. By default `False`.
    normalize : bool, optional
        Whether to normalize resulting spatial footprints of each cell to unit
        sum. By default `True`
    size_thres : tuple, optional
        The range of size in pixel allowed for the resulting spatial footprints.
        If `None`, then no filtering will be done. By default `(9, None)`.
    in_memory : bool, optional
        Whether to load `C` into memory before spatial update. By default
        `False`.

    Returns
    -------
    A_new : xr.DataArray
        New estimation of spatial footprints. Same shape as `A` except the
        "unit_id" dimension might be smaller due to filtering.
    mask : xr.DataArray
        Boolean mask of whether a cell passed size filtering. Has dimension
        "unit_id" that is same as input `A`. Useful for subsetting other
        variables based on the result of spatial update.
    b_new : xr.DataArray
        New estimation of spatial footprint of background. Only returned if
        `update_background` is `True`. Same shape as `b`.
    norm_fac : xr.DataArray
        Normalizing factor. Userful to scale temporal activity of cells. Only
        returned if `normalize` is `True`.
    Notes
    -------
    During spatial update, the algorithm solve the following optimization
    problem for each pixel:

    .. math::
        \\begin{aligned}
        & \\underset{\mathbf{a}}{\\text{minimize}}
        & & \\left \\lVert \mathbf{y} - \mathbf{a}^T \mathbf{C} \\right \\rVert
        ^2 + \\alpha \\left \\lvert \mathbf{a} \\right \\rvert \\\\
        & \\text{subject to} & & \mathbf{a} \geq 0
        \\end{aligned}

    Where :math:`\mathbf{y}` is the fluorescent dynamic of the pixel,
    :math:`\mathbf{a}` is spatial footprint values across all cells on that
    pixel, :math:`\mathbf{C}` is temporal component matrix across all cells. The
    parameter :math:`\\alpha` is the product of the noise level on each pixel
    `sn` and the global scalar `sparse_penal`. Higher value of :math:`\\alpha`
    will result in more sparse estimation of spatial footprints.
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if in_memory:
        C_store = C.compute().values
    else:
        C_path = os.path.join(intpath, C.name + ".zarr", C.name)
        C_store = zarr.open_array(C_path)
    print("estimating penalty parameter")
    alpha = sparse_penal * sn
    alpha = rechunk_like(alpha.compute(), sn)
    print("computing subsetting matrix")
    selem = moph.disk(dl_wnd)
    sub = xr.apply_ufunc(
        cv2.dilate,
        A,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        kwargs=dict(kernel=selem),
        dask="parallelized",
        output_dtypes=[A.dtype],
    )
    sub = sub > 0
    sub.data = sub.data.map_blocks(sparse.COO)
    if update_background:
        assert b is not None, "`b` must be provided when updating background"
        assert f is not None, "`f` must be provided when updating background"
        b_in = rechunk_like(b > 0, Y).assign_coords(unit_id=-1).expand_dims("unit_id")
        b_in.data = b_in.data.map_blocks(sparse.COO)
        b_in = b_in.compute()
        sub = xr.concat([sub, b_in], "unit_id")
        f_in = f.compute().data
    else:
        f_in = None
    sub = rechunk_like(sub.transpose("height", "width", "unit_id").compute(), Y)
    print("fitting spatial matrix")
    ssub = darr.map_blocks(
        sps_any,
        sub.data,
        drop_axis=2,
        chunks=((1, 1)),
        meta=sparse.ones(1).astype(bool),
    ).compute()
    Y_trans = Y.transpose("height", "width", "frame")
    # take fast route if a lot of chunks are empty
    if ssub.sum() < 500:
        A_new = np.empty(sub.data.numblocks, dtype=object)
        for (hblk, wblk), has_unit in np.ndenumerate(ssub):
            cur_sub = sub.data.blocks[hblk, wblk, :]
            if has_unit:
                cur_blk = update_spatial_block(
                    Y_trans.data.blocks[hblk, wblk, :],
                    alpha.data.blocks[hblk, wblk],
                    cur_sub,
                    C_store=C_store,
                    f=f_in,
                )
                if not isinstance(cur_blk, sparse.SparseArray):
                    cur_blk = cur_blk.map_blocks(sparse.COO)
            else:
                cur_blk = darr.array(sparse.zeros((cur_sub.shape)))
            A_new[hblk, wblk, 0] = cur_blk
        A_new = darr.block(A_new.tolist())
    else:
        A_new = update_spatial_block(
            Y_trans.data,
            alpha.data,
            sub.data,
            C_store=C_store,
            f=f_in,
        )
    with da.config.set(**{"optimization.fuse.ave-width": 6}):
        A_new = da.optimize(A_new)[0]
    A_new = xr.DataArray(
        darr.moveaxis(A_new, -1, 0).map_blocks(lambda a: a.todense(), dtype=A.dtype),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": sub.coords["unit_id"],
            "height": A.coords["height"],
            "width": A.coords["width"],
        },
    )
    A_new = save_minian(
        A_new.rename("A_new"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    add_rets = []
    if update_background:
        b_new = A_new.sel(unit_id=-1).compute()
        A_new = A_new[:-1, :, :]
        add_rets.append(b_new)
    if size_thres:
        low, high = size_thres
        A_bin = A_new > 0
        mask = np.ones(A_new.sizes["unit_id"], dtype=bool)
        if low:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) > low).compute(), mask
            )
        if high:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) < high).compute(), mask
            )
        mask = xr.DataArray(
            mask, dims=["unit_id"], coords={"unit_id": A_new.coords["unit_id"].values}
        )
    else:
        mask = (A_new.sum(["height", "width"]) > 0).compute()
    print("{} out of {} units dropped".format(len(mask) - mask.sum().values, len(mask)))
    A_new = A_new.sel(unit_id=mask)
    if normalize:
        norm_fac = A_new.max(["height", "width"]).compute()
        A_new = A_new / norm_fac
        add_rets.append(norm_fac)
    return (A_new, mask, *add_rets)


def sps_any(x: sparse.COO) -> np.ndarray:
    """
    Compute `any` on a sparse array.

    Parameters
    ----------
    x : sparse.COO
        Input sparse array.

    Returns
    -------
    x_any : np.ndarray
        2d boolean numpy array.
    """
    return np.atleast_2d(x.nnz > 0)


def update_spatial_perpx(
    y: np.ndarray,
    alpha: float,
    sub: sparse.COO,
    C_store: Union[np.ndarray, zarr.core.Array],
    f: Optional[np.ndarray],
) -> sparse.COO:
    """
    Update spatial footprints across all the cells for a single pixel.

    This function use :class:`sklearn.linear_model.LassoLars` to solve the
    optimization problem. `C_store` can either be a in-memory numpy array, or a
    zarr array, in which case it will be lazy-loaded. If `f` is not `None`, then
    `sub[-1]` is expected to be the subsetting mask for background, and the last
    element of the return value will be the spatial footprint of background.

    Parameters
    ----------
    y : np.ndarray
        Input fluorescent trace for the given pixel.
    alpha : float
        Parameter of the optimization problem controlling sparsity.
    sub : sparse.COO
        Subsetting matrix.
    C_store : Union[np.ndarray, zarr.core.Array]
        Estimation of temporal dynamics of cells.
    f : np.ndarray, optional
        Temporal dynamic of background.

    Returns
    -------
    A_px : sparse.COO
        Spatial footprint values across all cells for the given pixel.

    See Also
    -------
    update_spatial : for more explanation of parameters
    """
    if f is not None:
        idx = sub[:-1].nonzero()[0]
    else:
        idx = sub.nonzero()[0]
    try:
        C = C_store.get_orthogonal_selection((idx, slice(None))).T
    except AttributeError:
        C = C_store[idx, :].T
    if (f is not None) and sub[-1]:
        C = np.concatenate([C, f.reshape((-1, 1))], axis=1)
        idx = np.concatenate([idx, np.array(len(sub) - 1).reshape(-1)])
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LassoLars(alpha=alpha, positive=True, normalize=False),
    )
    coef = model.fit(C, y).named_steps["lassolars"].coef_
    mask = coef > 0
    coef = coef[mask]
    idx = idx[mask]
    return sparse.COO(coords=idx, data=coef, shape=sub.shape)


@darr.as_gufunc(signature="(f),(),(u)->(u)", output_dtypes=float)
def update_spatial_block(
    y: np.ndarray, alpha: np.ndarray, sub: sparse.COO, **kwargs
) -> sparse.COO:
    """
    Carry out spatial update for each 3d block of data.

    This function wraps around :func:`update_spatial_perpx` so that it can be
    applied to 3d blocks of data. Keyword arguments are passed to
    :func:`update_spatial_perpx`.

    Parameters
    ----------
    y : np.ndarray
        Input data, should have dimension (height, width, frame).
    alpha : np.ndarray
        Alpha parameter for the optimization problem. Should have dimension
        (height, width).
    sub : sparse.COO
        Subsetting matrix. Should have dimension (height, width, unit_id).

    Returns
    -------
    A_blk : sparse.COO
        Resulting spatial footprints. Should have dimension (height, width,
        unit_id).

    See Also
    -------
    update_spatial_perpx
    update_spatial
    """
    C_store = kwargs.get("C_store")
    f = kwargs.get("f")
    crd_ls = []
    data_ls = []
    for h, w in zip(*sub.any(axis=-1).nonzero()):
        res = update_spatial_perpx(y[h, w, :], alpha[h, w], sub[h, w, :], C_store, f)
        crd = res.coords
        crd = np.concatenate([np.full_like(crd, h), np.full_like(crd, w), crd], axis=0)
        crd_ls.append(crd)
        data_ls.append(res.data)
    if data_ls:
        return sparse.COO(
            coords=np.concatenate(crd_ls, axis=1),
            data=np.concatenate(data_ls),
            shape=sub.shape,
        )
    else:
        return sparse.zeros(sub.shape)


def compute_trace(
    Y: xr.DataArray, A: xr.DataArray, b: xr.DataArray, C: xr.DataArray, f: xr.DataArray
) -> xr.DataArray:
    """
    Compute the residule traces `YrA` for each cell.

    `YrA` is computed as `C + A_norm(YtA - CtA)`, where `YtA` is `(Y -
    b.dot(f)).tensordot(A, ["height", "width"])`, representing the projection of
    background-subtracted movie onto the spatial footprints, and `CtA` is
    `C.dot(AtA, ["unit_id"])` with `AtA = A.tensordot(A, ["height", "width"])`,
    hence `CtA` represent for each cell the sum of temporal activities that's
    shared with any other cells, then finally `A_norm` is a "unit_id"x"unit_id"
    diagonal matrix that normalize the result with sum of squares of spatial
    footprints for each cell. Together, the `YrA` trace is a "unit_id"x"frame"
    matrix, representing the sum of previous temporal components and the
    residule temporal fluctuations as estimated by projecting the data onto the
    spatial footprints and subtracting the cross-talk fluctuations.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions ("frame", "height", "width").
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id", "height",
        "width").
    b : xr.DataArray
        Spatial footprint of background. Should have dimensions ("height", "width").
    C : xr.DataArray
        Temporal components of cells. Should have dimensions ("frame", "unit_id").
    f : xr.DataArray
        Temporal dynamic of background. Should have dimension "frame".

    Returns
    -------
    YrA : xr.DataArray
        Residule traces for each cell. Should have dimensions("frame", "unit_id").
    """
    fms = Y.coords["frame"]
    uid = A.coords["unit_id"]
    Y = Y.data.map_blocks(sparse.COO)
    A = darr.from_array(A.data.map_blocks(sparse.COO).compute(), chunks=-1)
    C = C.data.map_blocks(sparse.COO).T
    b = (
        b.fillna(0)
        .data.map_blocks(sparse.COO)
        .reshape((1, Y.shape[1], Y.shape[2]))
        .compute()
    )
    f = f.fillna(0).data.reshape((-1, 1)).map_blocks(sparse.COO)
    AtA = darr.tensordot(A, A, axes=[(1, 2), (1, 2)]).compute()
    A_norm = (
        (1 / (A**2).sum(axis=(1, 2)))
        .map_blocks(
            lambda a: sparse.diagonalize(sparse.COO(a)), chunks=(A.shape[0], A.shape[0])
        )
        .compute()
    )
    B = darr.tensordot(f, b, axes=[(1), (0)])
    Y = Y - B
    YtA = darr.tensordot(Y, A, axes=[(1, 2), (1, 2)])
    YtA = darr.dot(YtA, A_norm)
    CtA = darr.dot(C, AtA)
    CtA = darr.dot(CtA, A_norm)
    YrA = (YtA - CtA + C).clip(0)
    arr_opt = fct.partial(
        custom_arr_optimize,
        inline_patterns=["from-getitem-transpose"],
        rename_dict={"tensordot": "tensordot_restricted"},
    )
    with da.config.set(array_optimize=arr_opt):
        YrA = da.optimize(YrA)[0]
    YrA = xr.DataArray(
        YrA,
        dims=["frame", "unit_id"],
        coords={"frame": fms, "unit_id": uid},
    )
    YrA = YrA.transpose("unit_id", "frame")
    YrA.data = YrA.data.map_blocks(lambda x: x.todense())
    return YrA


def update_temporal(
    A: xr.DataArray,
    C: xr.DataArray,
    b: Optional[xr.DataArray] = None,
    f: Optional[xr.DataArray] = None,
    Y: Optional[xr.DataArray] = None,
    YrA: Optional[xr.DataArray] = None,
    noise_freq=0.25,
    p=2,
    add_lag="p",
    jac_thres=0.1,
    sparse_penal=1,
    bseg: Optional[np.ndarray] = None,
    med_wd: Optional[int] = None,
    zero_thres=1e-8,
    max_iters=200,
    use_smooth=True,
    normalize=True,
    warm_start=False,
    post_scal=False,
    scs_fallback=False,
    concurrent_update=False,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
]:
    """
    Update temporal components and deconvolve calcium traces for each cell given
    spatial footprints.

    This function carries out temporal update of the CNMF algorithm. The update
    is done in parallel and independently for each group of cells. The grouping
    of cells is controlled by `jac_thres`. The relationship between calcium and
    deconvolved spikes is modeled as an Autoregressive process (AR) of order
    `p`. The AR coefficients are estimated from autocovariances of `YrA` traces
    for each cell, with `add_lag` controls how many timesteps of autocovariances
    are used. Optionally, the `YrA` traces can be smoothed for the estimation of
    AR coefficients only. The noise level for each cell is estimated using FFT
    with `noise_freq` as cut-off, and controls the sparsity of the result
    together with the global `sparse_penal` parameter. `YrA` traces for each
    cells can be optionally normalized to unit sum to make `sparse_penal` to
    have comparable effects across cells. If abrupt change of baseline
    fluorescence is expected, a `bseg` vector can be passed to enable estimation
    of independent baseline for different segments of time. The temporal update
    itself is performed by solving an optimization problem using `cvxpy`, with
    `concurrent_update`, `warm_start`, `max_iters`, `scs_fallback` controlling
    different aspects of the optimization. Finally, the results can be filtered
    with `zero_thres` to suppress small values caused by numerical errors, and a
    post-hoc scaling process can be optionally used to scale the result based on
    `YrA` to get around unwanted effects from sparse penalty or normalization.

    Parameters
    ----------
    A : xr.DataArray
        Estimation of spatial footprints for each cell. Should have dimensions
        ("unit_id", "height", "width").
    C : xr.DataArray
        Previous estimation of calcium dynamic of cells. Should have dimensions
        ("frame", "unit_id"). Only used if `warm_start = True` or if `YrA is
        None`.
    b : xr.DataArray, optional
        Estimation of spatial footprint of background. Should have dimensions
        ("height", "width"). Only used if `YrA is None`. By default `None`.
    f : xr.DataArray, optional
        Estimation of temporal dynamic of background. Should have dimension
        "frame". Only used if `YrA is None`. By default `None`.
    Y : xr.DataArray, optional
        Input movie data. Should have dimensions ("frame", "height", "width").
        Only used if `YrA is None`. By default `None`.
    YrA : xr.DataArray, optional
        Estimation of residule traces for each cell. Should have dimensions
        ("frame", "unit_id"). If `None` then one will be computed using
        `computea_trace` with relevant inputs. By default `None`.
    noise_freq : float, optional
        Frequency cut-off for both the estimation of noise level and the
        optional smoothing, specified as a fraction of sampling frequency. By
        default `0.25`.
    p : int, optional
        Order of the AR process. By default `2`.
    add_lag : str, optional
        Additional number of timesteps in covariance to use for the estimation
        of AR coefficients. If `0`, then only the first `p` number of timesteps
        will be used to estimate the `p` number of AR coefficients. If greater
        than `0`, then the system is over-determined and least square will be
        used to estimate AR coefficients. If `"p"`, then `p` number of
        additional timesteps will be used. By default `"p"`.
    jac_thres : float, optional
        Threshold for Jaccard Index. Cells whose overlap in spatial footprints
        (number of common pixels divided by number of total pixels) exceeding
        this threshold will be grouped together transitively for temporal
        update. By default `0.1`.
    sparse_penal : int, optional
        Global scalar controlling sparsity of the result. The higher the value,
        the sparser the deconvolved spikes. By default `1`.
    bseg : np.ndarray, optional
        1d vector with length "frame" representing segments for which baseline
        should be estimated independently. An independent baseline will be
        estimated for frames corresponding to each unique label in this vector.
        If `None` then a single scalar baseline will be estimated for each cell.
        By default `None`.
    med_wd : int, optional
        Window size for the median filter used for baseline correction. For each
        cell, the baseline flurescence is estimated by median-filtering the
        temporal activity. Then the baseline is subtracted from the temporal
        activity right before the optimization step. If `None` then no baseline
        correction will be performed. By default `None`.
    zero_thres : float, optional
        Threshold to filter out small values in the result. Any values smaller
        than this threshold will be set to zero. By default `1e-8`.
    max_iters : int, optional
        Maximum number of iterations for optimization. Can be increased to get
        around sub-optimal results. By default `200`.
    use_smooth : bool, optional
        Whether to smooth the `YrA` for the estimation of AR coefficients. If
        `True`, then a smoothed version of `YrA` will be computed by low-pass
        filter with `noise_freq` and used for the estimation of AR coefficients
        only. By default `True`.
    normalize : bool, optional
        Whether to normalize `YrA` for each cell to unit sum such that sparse
        penalty has simlar effect for all the cells. Each group of cell will be
        normalized together (with mean of the sum for each cell) to preserve
        relative amplitude of fluorescence between overlapping cells. By default
        `True`.
    warm_start : bool, optional
        Whether to use previous estimation of `C` to warm start the
        optimization. Can lead to faster convergence in theory. Experimental. By
        default `False`.
    post_scal : bool, optional
        Whether to apply the post-hoc scaling process, where a scalar will be
        estimated with least square for each cell to scale the amplitude of
        temporal component to `YrA`. Useful to get around unwanted dampening of
        result values caused by high `sparse_penal` or to revert the per-cell
        normalization. By default `False`.
    scs_fallback : bool, optional
        Whether to fall back to `scs` solver if the default `ecos` solver fails.
        By default `False`.
    concurrent_update : bool, optional
        Whether to update a group of cells as a single optimization problem.
        Yields slightly more accurate estimation when cross-talk between cells
        are severe, but significantly increase convergence time and memory
        demand. By default `False`.

    Returns
    -------
    C_new : xr.DataArray
        New estimation of the calcium dynamic for each cell. Should have same
        shape as `C` except the "unit_id" dimension might be smaller due to
        dropping of cells and filtering.
    S_new : xr.DataArray
        New estimation of the deconvolved spikes for each cell. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`.
    b0_new : xr.DataArray
        New estimation of baseline fluorescence for each cell. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`. Each cell
        should only have one unique value if `bseg is None`.
    c0_new : xr.DataArray
        New estimation of a initial calcium decay, in theory triggered by
        calcium events happened before the recording starts. Should have
        dimensions ("frame", "unit_id") and same shape as `C_new`.
    g : xr.DataArray
        Estimation of AR coefficient for each cell. Useful for visualizing
        modeled calcium dynamic. Should have dimensions ("lag", "unit_id") with
        "lag" having length `p`.
    mask : xr.DataArray
        Boolean mask of whether a cell has any temporal dynamic after the update
        and optional filtering. Has dimension "unit_id" that is same as input
        `C`. Useful for subsetting other variables based on the result of
        temporal update.


    Notes
    -------
    During temporal update, the algorithm solve the following optimization
    problem for each cell:

    .. math::
        \\begin{aligned}
        & \\underset{\mathbf{c} \, \mathbf{b_0} \,
        \mathbf{c_0}}{\\text{minimize}}
        & & \\left \\lVert \mathbf{y} - \mathbf{c} - \mathbf{c_0} -
        \mathbf{b_0} \\right \\rVert ^2 + \\alpha \\left \\lvert \mathbf{G}
        \mathbf{c} \\right \\rvert \\\\
        & \\text{subject to}
        & & \mathbf{c} \geq 0, \; \mathbf{G} \mathbf{c} \geq 0
        \\end{aligned}

    Where :math:`\mathbf{y}` is the estimated residule trace (`YrA`) for the
    cell, :math:`\mathbf{c}` is the calcium dynamic of the cell,
    :math:`\mathbf{G}` is a "frame"x"frame" matrix constructed from the
    estimated AR coefficients of cell, such that the deconvolved spikes of the
    cell is given by :math:`\mathbf{G}\mathbf{c}`. If `bseg is None`, then
    :math:`\mathbf{b_0}` is a single scalar, otherwise it is a 1d vector with
    dimension "frame" constrained to have multiple independent values, each
    corresponding to a segment of time specified in `bseg`. :math:`\mathbf{c_0}`
    is a 1d vector with dimension "frame" constrained to be the product of a
    scalar (representing initial calcium concentration) and the decay dynamic
    given by the estimated AR coefficients. The parameter :math:`\\alpha` is the
    product of estimated noise level of the cell and the global scalar
    `sparse_penal`. Higher value of :math:`\\alpha` will result in more sparse
    estimation of deconvolved spikes.
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if YrA is None:
        YrA = compute_trace(Y, A, b, C, f).persist()
    Ymask = (YrA > 0).any("frame").compute()
    if hasattr(Ymask.data, "todense"):
        Ymask.data = Ymask.data.todense()
    A, C, YrA = A.sel(unit_id=Ymask), C.sel(unit_id=Ymask), YrA.sel(unit_id=Ymask)
    print("grouping overlaping units")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).compute().astype(np.float32)
    A_inter = sparse.tensordot(A_sps, A_sps, axes=[(1, 2), (1, 2)]).todense()
    A_usum = np.tile(A_sps.sum(axis=(1, 2)).todense(), (A_sps.shape[0], 1))
    A_usum = A_usum + A_usum.T
    jac = scipy.sparse.csc_matrix(A_inter / (A_usum - A_inter) > jac_thres)
    unit_labels = label_connected(jac)
    YrA = YrA.assign_coords(unit_labels=("unit_id", unit_labels))
    print("updating temporal components")
    c_ls = []
    s_ls = []
    b_ls = []
    c0_ls = []
    g_ls = []
    uid_ls = []
    grp_dim = "unit_labels"
    C = C.assign_coords(unit_labels=("unit_id", unit_labels))
    if warm_start:
        C.data = C.data.map_blocks(scipy.sparse.csr_matrix)
    inline_opt = fct.partial(
        custom_delay_optimize,
        inline_patterns=["getitem", "rechunk-merge"],
    )
    for cur_YrA, cur_C in zip(YrA.groupby(grp_dim), C.groupby(grp_dim)):
        uid_ls.append(cur_YrA[1].coords["unit_id"].values.reshape(-1))
        cur_YrA, cur_C = cur_YrA[1].data.rechunk(-1), cur_C[1].data.rechunk(-1)
        # peak memory demand for cvxpy is roughly 500 times input
        mem_cvx = cur_YrA.nbytes if concurrent_update else cur_YrA[0].nbytes
        mem_cvx = mem_cvx * 500
        mem_demand = max(mem_cvx, cur_YrA.nbytes * 5) / 1e6
        # issue a warning if expected memory demand is larger than 1G
        if mem_demand > 1e3:
            warnings.warn(
                "{} cells will be updated togeter, "
                "which takes roughly {} MB of memory. "
                "Consider merging the units "
                "or changing jac_thres".format(cur_YrA.shape[0], mem_demand)
            )
        if not warm_start:
            cur_C = None
        if cur_YrA.shape[0] > 1:
            dl_opt = inline_opt
        else:
            dl_opt = custom_delay_optimize
        # explicitly using delay (rather than gufunc) seem to promote the
        # depth-first behavior of dask
        with da.config.set(delayed_optimize=dl_opt):
            res = da.optimize(
                da.delayed(update_temporal_block)(
                    cur_YrA,
                    noise_freq=noise_freq,
                    p=p,
                    add_lag=add_lag,
                    normalize=normalize,
                    concurrent=concurrent_update,
                    use_smooth=use_smooth,
                    c_last=cur_C,
                    bseg=bseg,
                    med_wd=med_wd,
                    sparse_penal=sparse_penal,
                    max_iters=max_iters,
                    scs_fallback=scs_fallback,
                    zero_thres=zero_thres,
                )
            )[0]
        c_ls.append(darr.from_delayed(res[0], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        s_ls.append(darr.from_delayed(res[1], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        b_ls.append(darr.from_delayed(res[2], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        c0_ls.append(
            darr.from_delayed(res[3], shape=cur_YrA.shape, dtype=cur_YrA.dtype)
        )
        g_ls.append(
            darr.from_delayed(res[4], shape=(cur_YrA.shape[0], p), dtype=cur_YrA.dtype)
        )
    uids_new = np.concatenate(uid_ls)
    C_new = xr.DataArray(
        darr.concatenate(c_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"],
        },
        name="C_new",
    )
    S_new = xr.DataArray(
        darr.concatenate(s_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="S_new",
    )
    b0_new = xr.DataArray(
        darr.concatenate(b_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="b0_new",
    )
    c0_new = xr.DataArray(
        darr.concatenate(c0_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new,
            "frame": YrA.coords["frame"].values,
        },
        name="c0_new",
    )
    g = xr.DataArray(
        darr.concatenate(g_ls, axis=0),
        dims=["unit_id", "lag"],
        coords={"unit_id": uids_new, "lag": np.arange(p)},
        name="g",
    )
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^update_temporal_block"])
    with da.config.set(array_optimize=arr_opt):
        da.compute(
            [
                save_minian(
                    var.chunk({"unit_id": 1}), intpath, compute=False, overwrite=True
                )
                for var in [C_new, S_new, b0_new, c0_new, g]
            ]
        )
    int_ds = open_minian(intpath, return_dict=True)
    C_new, S_new, b0_new, c0_new, g = (
        int_ds["C_new"],
        int_ds["S_new"],
        int_ds["b0_new"],
        int_ds["c0_new"],
        int_ds["g"],
    )
    mask = (S_new.sum("frame") > 0).compute()
    print("{} out of {} units dropped".format((~mask).sum().values, len(Ymask)))
    C_new, S_new, b0_new, c0_new, g = (
        C_new[mask],
        S_new[mask],
        b0_new[mask],
        c0_new[mask],
        g[mask],
    )
    sig_new = C_new + b0_new + c0_new
    sig_new = da.optimize(sig_new)[0]
    YrA_new = YrA.sel(unit_id=mask)
    if post_scal and len(sig_new) > 0:
        print("post-hoc scaling")
        scal = lstsq_vec(sig_new.data, YrA_new.data).compute().reshape((-1, 1))
        C_new, S_new, b0_new, c0_new = (
            C_new * scal,
            S_new * scal,
            b0_new * scal,
            c0_new * scal,
        )
    return C_new, S_new, b0_new, c0_new, g, mask


@darr.as_gufunc(signature="(f),(f)->()", output_dtypes=float)
def lstsq_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Estimate a least-square scaling from `a` to `b` in vectorized fashion.

    Parameters
    ----------
    a : np.ndarray
        Source of the scaling.
    b : np.ndarray
        Target of the scaling.

    Returns
    -------
    scale : np.ndarray
        A scaler that scales `a` to `b`.
    """
    a = a.reshape((-1, 1))
    return np.linalg.lstsq(a, b.squeeze(), rcond=-1)[0]


def get_ar_coef(
    y: np.ndarray, sn: float, p: int, add_lag: int, pad: Optional[int] = None
) -> np.ndarray:
    """
    Estimate Autoregressive coefficients of order `p` given a timeseries `y`.

    Parameters
    ----------
    y : np.ndarray
        Input timeseries.
    sn : float
        Estimated noise level of the input `y`.
    p : int
        Order of the autoregressive process.
    add_lag : int
        Additional number of timesteps of covariance to use for the estimation.
    pad : int, optional
        Length of the output. If not `None` then the resulting coefficients will
        be zero-padded to this length. By default `None`.

    Returns
    -------
    g : np.ndarray
        The estimated AR coefficients.
    """
    if add_lag == "p":
        max_lag = p * 2
    else:
        max_lag = p + add_lag
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p]) - sn**2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1 : max_lag + 1])[0]
    if pad:
        res = np.zeros(pad)
        res[: len(g)] = g
        return res
    else:
        return g


def get_p(y):
    dif = np.append(np.diff(y), 0)
    rising = dif > 0
    prd_ris, num_ris = label(rising)
    ext_prd = np.zeros(num_ris)
    for id_prd in range(num_ris):
        prd = y[prd_ris == id_prd + 1]
        ext_prd[id_prd] = prd[-1] - prd[0]
    id_max_prd = np.argmax(ext_prd)
    return np.sum(rising[prd_ris == id_max_prd + 1])


def update_temporal_block(
    YrA: np.ndarray,
    noise_freq: float,
    p: int,
    add_lag="p",
    normalize=True,
    use_smooth=True,
    med_wd=None,
    concurrent=False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update temporal components given residule traces of a group of cells.

    This function wraps around :func:`update_temporal_cvxpy`, but also carry out
    additional initial steps given `YrA` of a group of cells. Additional keyword
    arguments are passed through to :func:`update_temporal_cvxpy`.

    Parameters
    ----------
    YrA : np.ndarray
        Residule traces of a group of cells. Should have dimension ("unit_id",
        "frame").
    noise_freq : float
        Frequency cut-off for both the estimation of noise level and the
        optional smoothing. Specified as a fraction of sampling frequency.
    p : int
        Order of the AR process.
    add_lag : str, optional
        Additional number of timesteps in covariance to use for the estimation
        of AR coefficients. By default "p".
    normalize : bool, optional
        Whether to normalize `YrA` for each cell to unit sum. By default `True`.
    use_smooth : bool, optional
        Whether to smooth the `YrA` for the estimation of AR coefficients. By
        default `True`.
    med_wd : int, optional
        Median window used for baseline correction.
    concurrent : bool, optional
        Whether to update a group of cells as a single optimization problem. By
        default `False`.

    Returns
    -------
    c : np.ndarray
        New estimation of the calcium dynamic of the group of cells. Should have
        dimensions ("unit_id", "frame") and same shape as `YrA`.
    s : np.ndarray
        New estimation of the deconvolved spikes of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    b : np.ndarray
        New estimation of baseline fluorescence of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    c0 : np.ndarray
        New estimation of a initial calcium decay of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    g : np.ndarray
        Estimation of AR coefficient for each cell. Should have dimensions
        ("unit_id", "lag") with "lag" having length `p`.

    See Also
    -------
    update_temporal : for more explanation of parameters
    """
    vec_get_noise = np.vectorize(
        noise_fft,
        otypes=[float],
        excluded=["noise_range", "noise_method"],
        signature="(f)->()",
    )
    vec_get_p = np.vectorize(get_p, otypes=[int], signature="(f)->()")
    vec_get_ar_coef = np.vectorize(
        get_ar_coef,
        otypes=[float],
        excluded=["pad", "add_lag"],
        signature="(f),(),()->(l)",
    )
    if hasattr(YrA, "todense"):
        YrA = YrA.todense()
    if normalize:
        amean = YrA.sum(axis=1).mean()
        norm_factor = YrA.shape[1] / amean
        YrA *= norm_factor
    else:
        norm_factor = np.ones(YrA.shape[0])
    tn = vec_get_noise(YrA, noise_range=(noise_freq, 1))
    if use_smooth:
        YrA_ar = filt_fft_vec(YrA, noise_freq, "low")
        tn_ar = vec_get_noise(YrA_ar, noise_range=(noise_freq, 1))
    else:
        YrA_ar, tn_ar = YrA, tn
    # auto estimation of p is disabled since it's never used and makes it
    # impossible to pre-determine the shape of output
    # if p is None:
    #     p = np.clip(vec_get_p(YrA_ar), 1, None)
    pmax = np.max(p)
    g = vec_get_ar_coef(YrA_ar, tn_ar, p, pad=pmax, add_lag=add_lag)
    del YrA_ar, tn_ar
    if med_wd is not None:
        for i, cur_yra in enumerate(YrA):
            YrA[i, :] = med_baseline(cur_yra, med_wd)
    if concurrent:
        c, s, b, c0 = update_temporal_cvxpy(YrA, g, tn, **kwargs)
    else:
        res_ls = []
        for cur_yra, cur_g, cur_tn in zip(YrA, g, tn):
            res = update_temporal_cvxpy(cur_yra, cur_g, cur_tn, **kwargs)
            res_ls.append(res)
        c = np.concatenate([r[0] for r in res_ls], axis=0) / norm_factor
        s = np.concatenate([r[1] for r in res_ls], axis=0) / norm_factor
        b = np.concatenate([r[2] for r in res_ls], axis=0) / norm_factor
        c0 = np.concatenate([r[3] for r in res_ls], axis=0) / norm_factor
    return c, s, b, c0, g


def update_temporal_cvxpy(
    y: np.ndarray, g: np.ndarray, sn: np.ndarray, A=None, bseg=None, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the temporal update optimization problem using `cvxpy`

    Parameters
    ----------
    y : np.ndarray
        Input residule trace of one or more cells.
    g : np.ndarray
        Estimated AR coefficients of one or more cells.
    sn : np.ndarray
        Noise level of one or more cells.
    A : np.ndarray, optional
        Spatial footprint of one or more cells. Not used. By default `None`.
    bseg : np.ndarray, optional
        1d vector with length "frame" representing segments for which baseline
        should be estimated independently. By default `None`.

    Returns
    -------
    c : np.ndarray
        New estimation of the calcium dynamic of the group of cells. Should have
        dimensions ("unit_id", "frame") and same shape as `y`.
    s : np.ndarray
        New estimation of the deconvolved spikes of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    b : np.ndarray
        New estimation of baseline fluorescence of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.
    c0 : np.ndarray
        New estimation of a initial calcium decay of the group of cells. Should
        have dimensions ("unit_id", "frame") and same shape as `c`.

    Other Parameters
    -------
    sparse_penal : float
        Sparse penalty parameter for all the cells.
    max_iters : int
        Maximum number of iterations.
    use_cons : bool, optional
        Whether to try constrained version of the problem first. By default
        `False`.
    scs_fallback : bool
        Whether to fall back to `scs` solver if the default `ecos` solver fails.
    c_last : np.ndarray, optional
        Initial estimation of calcium traces for each cell used to warm start.
    zero_thres : float
        Threshold to filter out small values in the result.

    See Also
    -------
    update_temporal : for more explanation of parameters
    """
    # spatial:
    # (d, f), (u, p), (d), (d, u)
    # (d, f), (p), (d), (d)
    # trace:
    # (u, f), (u, p), (u)
    # (f), (p), ()

    # get_parameters
    sparse_penal = kwargs.get("sparse_penal")
    max_iters = kwargs.get("max_iters")
    use_cons = kwargs.get("use_cons", False)
    scs = kwargs.get("scs_fallback")
    c_last = kwargs.get("c_last")
    zero_thres = kwargs.get("zero_thres")
    # conform variables to generalize multiple unit case
    if y.ndim < 2:
        y = y.reshape((1, -1))
    if g.ndim < 2:
        g = g.reshape((1, -1))
    sn = np.atleast_1d(sn)
    if A is not None:
        if A.ndim < 2:
            A = A.reshape((-1, 1))
    # get count of frames and units
    _T = y.shape[-1]
    _u = g.shape[0]
    if A is not None:
        _d = A.shape[0]
    # construct G matrix and decay vector per unit
    dc_vec = np.zeros((_u, _T))
    G_ls = []
    for cur_u in range(_u):
        cur_g = g[cur_u, :]
        # construct first column and row
        cur_c = np.zeros(_T)
        cur_c[0] = 1
        cur_c[1 : len(cur_g) + 1] = -cur_g
        # update G with toeplitz matrix
        G_ls.append(
            cvx.Constant(
                dia_matrix(
                    (
                        np.tile(np.concatenate(([1], -cur_g)), (_T, 1)).T,
                        -np.arange(len(cur_g) + 1),
                    ),
                    shape=(_T, _T),
                ).tocsc()
            )
        )
        # update dc_vec
        cur_gr = np.roots(cur_c)
        dc_vec[cur_u, :] = np.max(cur_gr.real) ** np.arange(_T)
    # get noise threshold
    thres_sn = sn * np.sqrt(_T)
    # construct variables
    if bseg is not None:
        nseg = int(np.max(bseg) + 1)
        b_temp = np.zeros((nseg, _T))
        for iseg in range(nseg):
            b_temp[iseg, bseg == iseg] = 1
        b_cmp = cvx.Variable((_u, nseg))
    else:
        b_temp = np.ones((1, _T))
        b_cmp = cvx.Variable((_u, 1))
    b = b_cmp @ b_temp  # baseline fluorescence per unit
    c0 = cvx.Variable(_u)  # initial fluorescence per unit
    c = cvx.Variable((_u, _T))  # calcium trace per unit
    if c_last is not None:
        c.value = c_last
        warm_start = True
    else:
        warm_start = False
    s = cvx.vstack([G_ls[u] @ c[u, :] for u in range(_u)])  # spike train per unit
    # residual noise per unit
    if A is not None:
        sig = cvx.vstack(
            [
                (A * c)[px, :] + (A * b)[px, :] + (A * cvx.diag(c0) * dc_vec)[px, :]
                for px in range(_d)
            ]
        )
        noise = y - sig
    else:
        sig = cvx.vstack([c[u, :] + b[u, :] + c0[u] * dc_vec[u, :] for u in range(_u)])
        noise = y - sig
    noise = cvx.vstack([cvx.norm(noise[i, :], 2) for i in range(noise.shape[0])])
    # construct constraints
    cons = []
    cons.append(
        b >= np.broadcast_to(np.min(y, axis=-1).reshape((-1, 1)), y.shape)
    )  # baseline larger than minimum
    cons.append(c0 >= 0)  # initial fluorescence larger than 0
    cons.append(s >= 0)  # spike train non-negativity
    # noise constraints
    cons_noise = [noise[i] <= thres_sn[i] for i in range(thres_sn.shape[0])]
    try:
        obj = cvx.Minimize(cvx.sum(cvx.norm(s, 1, axis=1)))
        prob = cvx.Problem(obj, cons + cons_noise)
        if use_cons:
            _ = prob.solve(solver="ECOS")
        if not (prob.status == "optimal" or prob.status == "optimal_inaccurate"):
            if use_cons:
                warnings.warn("constrained version of problem infeasible")
            raise ValueError
    except (ValueError, cvx.SolverError):
        lam = sn * sparse_penal
        obj = cvx.Minimize(
            cvx.sum(cvx.sum(noise, axis=1) + cvx.multiply(lam, cvx.norm(s, 1, axis=1)))
        )
        prob = cvx.Problem(obj, cons)
        try:
            _ = prob.solve(solver="ECOS", warm_start=warm_start, max_iters=max_iters)
            if prob.status in ["infeasible", "unbounded", None]:
                raise ValueError
        except (cvx.SolverError, ValueError):
            try:
                if scs:
                    _ = prob.solve(solver="SCS", max_iters=200)
                if prob.status in ["infeasible", "unbounded", None]:
                    raise ValueError
            except (cvx.SolverError, ValueError):
                warnings.warn(
                    "problem status is {}, returning zero".format(prob.status),
                    RuntimeWarning,
                )
                return [np.zeros(c.shape, dtype=float)] * 4
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)
    c = np.where(c.value > zero_thres, c.value, 0)
    s = np.where(s.value > zero_thres, s.value, 0)
    b = np.where(b.value > zero_thres, b.value, 0)
    c0 = c0.value.reshape((-1, 1)) * dc_vec
    c0 = np.where(c0 > zero_thres, c0, 0)
    return c, s, b, c0


def unit_merge(
    A: xr.DataArray,
    C: xr.DataArray,
    add_list: Optional[List[xr.DataArray]] = None,
    thres_corr=0.9,
    noise_freq: Optional[float] = None,
) -> Tuple[xr.DataArray, xr.DataArray, Optional[List[xr.DataArray]]]:
    """
    Merge cells given spatial footprints and temporal components

    This function merge all cells that have common pixels based on correlation
    of their temporal components. The cells to be merged will become one cell,
    with spatial and temporal components taken as mean across all the cells to
    be merged. Additionally any variables specified in `add_list` will be merged
    in the same manner. Optionally the temporal components can be smoothed
    before being used to caculate correlation. Despite the name any timeseries
    be passed as `C` and used to calculate the correlation.

    Parameters
    ----------
    A : xr.DataArray
        Spatial footprints of the cells.
    C : xr.DataArray
        Temporal component of cells.
    add_list : List[xr.DataArray], optional
        List of additional variables to be merged. By default `None`.
    thres_corr : float, optional
        The threshold of correlation. Any pair of spatially overlapping cells
        with correlation higher than this threshold will be transitively grouped
        together and merged. By default `0.9`.
    noise_freq : float, optional
        The cut-off frequency used to smooth `C` before calculation of
        correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    A_merge : xr.DataArray
        Merged spatial footprints of cells.
    C_merge : xr.DataArray
        Merged temporal components of cells.
    add_list : List[xr.DataArray], optional
        List of additional merged variables. Only returned if input `add_list`
        is not `None`.
    """
    print("computing spatial overlap")
    with da.config.set(
        array_optimize=darr.optimization.optimize,
        **{"optimization.fuse.subgraphs": False}
    ):
        A_sps = (A.data.map_blocks(sparse.COO) > 0).rechunk(-1).persist()
        A_inter = sparse.tril(
            darr.tensordot(
                A_sps.astype(np.float32),
                A_sps.astype(np.float32),
                axes=[(1, 2), (1, 2)],
            ).compute(),
            k=-1,
        )
    print("computing temporal correlation")
    nod_df = pd.DataFrame({"unit_id": A.coords["unit_id"].values})
    adj = adj_corr(C, A_inter, nod_df, freq=noise_freq)
    print("labeling units to be merged")
    adj = adj > thres_corr
    adj = adj + adj.T
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj,
        input_core_dims=[["unit_id", "unit_id_cp"]],
        output_core_dims=[["unit_id"]],
    )
    print("merging units")
    A_merge = (
        A.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    C_merge = (
        C.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    if add_list:
        for ivar, var in enumerate(add_list):
            var_mrg = (
                var.assign_coords(unit_labels=("unit_id", unit_labels))
                .groupby("unit_labels")
                .mean("unit_id")
                .rename(unit_labels="unit_id")
            )
            add_list[ivar] = var_mrg
        return A_merge, C_merge, add_list
    else:
        return A_merge, C_merge


def label_connected(adj: np.ndarray, only_connected=False) -> np.ndarray:
    """
    Label connected components given adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix. Should be 2d symmetric matrix.
    only_connected : bool, optional
        Whether to keep only the labels of connected components. If `True`, then
        all components with only one node (isolated) will have their labels set
        to -1. Otherwise all components will have unique label. By default
        `False`.

    Returns
    -------
    labels : np.ndarray
        The labels for each components. Should have length `adj.shape[0]`.
    """
    try:
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj)
        g = nx.convert_matrix.from_numpy_matrix(adj)
    except:
        g = nx.convert_matrix.from_scipy_sparse_matrix(adj)
    labels = np.zeros(adj.shape[0], dtype=np.int)
    for icomp, comp in enumerate(nx.connected_components(g)):
        comp = list(comp)
        if only_connected and len(comp) == 1:
            labels[comp] = -1
        else:
            labels[comp] = icomp
    return labels


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

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

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
    xfft = numpy_fft.rfft(x)
    xfft[zero_range] = 0
    return numpy_fft.irfft(xfft, len(x))


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
    Vectorized wrapper of :func:`filt_fft`.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries. Should have 2 dimensions, and the filtering will be
        applied along the last dimension.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries
    """
    for ix, xx in enumerate(x):
        x[ix, :] = filt_fft(xx, freq, btype)
    return x


def compute_AtC(A: xr.DataArray, C: xr.DataArray) -> xr.DataArray:
    """
    Compute the outer product of spatial and temporal components.

    This funtion computes the outer product of spatial and temporal components.
    The result is a 3d array representing the movie data as estimated by the
    spatial and temporal components.

    Parameters
    ----------
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id",
        "height", "width").
    C : xr.DataArray
        Temporal components of cells. Should have dimensions "frame" and
        "unit_id".

    Returns
    -------
    AtC : xr.DataArray
        The outer product representing estimated movie data. Has dimensions
        ("frame", "height", "width").
    """
    fm, h, w = (
        C.coords["frame"].values,
        A.coords["height"].values,
        A.coords["width"].values,
    )
    A = darr.from_array(
        A.data.map_blocks(sparse.COO, dtype=A.dtype).compute(), chunks=-1
    )
    C = C.transpose("frame", "unit_id").data.map_blocks(sparse.COO, dtype=C.dtype)
    AtC = darr.tensordot(C, A, axes=(1, 0)).map_blocks(
        lambda a: a.todense(), dtype=A.dtype
    )
    arr_opt = fct.partial(
        custom_arr_optimize, rename_dict={"tensordot": "tensordot_restricted"}
    )
    with da.config.set(array_optimize=arr_opt):
        AtC = da.optimize(AtC)[0]
    return xr.DataArray(
        AtC,
        dims=["frame", "height", "width"],
        coords={"frame": fm, "height": h, "width": w},
    )


def graph_optimize_corr(
    varr: xr.DataArray,
    G: nx.Graph,
    freq: float,
    idx_dims=["height", "width"],
    chunk=600,
    step_size=50,
) -> pd.DataFrame:
    """
    Compute correlation in an optimized fashion given a computation graph.

    This function carry out out-of-core computation of large correaltion matrix.
    It takes in a computaion graph whose node represent timeseries and edges
    represent the desired pairwise correlation to be computed. The actual
    timeseries are stored in `varr` and indexed with node attributes. The
    function can carry out smoothing of timeseries before computation of
    correlation. To minimize re-computation of smoothing for each pixel, the
    graph is first partitioned using a minial-cut algorithm. Then the
    computation is performed in chunks with size `chunk`, with nodes from the
    same partition being in the same chunk as much as possible.

    Parameters
    ----------
    varr : xr.DataArray
        Input timeseries. Should have "frame" dimension in addition to those
        specified in `idx_dims`.
    G : nx.Graph
        Graph representing computation to be carried out. Should be undirected
        and un-weighted. Each node should have unique attributes with keys
        specified in `idx_dims`, which will be used to index the timeseries in
        `varr`. Each edge represent a desired correlation.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.
    idx_dims : list, optional
        The dimension used to index the timeseries in `varr`. By default
        `["height", "width"]`.
    chunk : int, optional
        Chunk size of each computation. By default `600`.
    step_size : int, optional
        Step size to iterate through all edges. If too small then the iteration
        will take a long time. If too large then the variances in the actual
        chunksize of computation will be large. By default `50`.

    Returns
    -------
    eg_df : pd.DataFrame
        Dataframe representation of edge list. Has column "source" and "target"
        representing the node index of the edge (correlation), and column "corr"
        with computed value of correlation.
    """
    # a heuristic to make number of partitions scale with nodes
    n_cuts, membership = pymetis.part_graph(
        max(int(np.ceil(G.number_of_nodes() / chunk)), 1), adjacency=adj_list(G)
    )
    nx.set_node_attributes(
        G, {k: {"part": v} for k, v in zip(sorted(G.nodes), membership)}
    )
    eg_df = nx.to_pandas_edgelist(G)
    part_map = nx.get_node_attributes(G, "part")
    eg_df["part_src"] = eg_df["source"].map(part_map)
    eg_df["part_tgt"] = eg_df["target"].map(part_map)
    eg_df["part_diff"] = (eg_df["part_src"] - eg_df["part_tgt"]).astype(bool)
    corr_ls = []
    idx_ls = []
    npxs = []
    egd_same, egd_diff = eg_df[~eg_df["part_diff"]], eg_df[eg_df["part_diff"]]
    idx_dict = {d: nx.get_node_attributes(G, d) for d in idx_dims}

    def construct_comput(edf, pxs):
        px_map = {k: v for v, k in enumerate(pxs)}
        ridx = edf["source"].map(px_map).values
        cidx = edf["target"].map(px_map).values
        idx_arr = {
            d: xr.DataArray([dd[p] for p in pxs], dims="pixels")
            for d, dd in idx_dict.items()
        }
        vsub = varr.sel(**idx_arr).data
        if len(idx_arr) > 1:  # vectorized indexing
            vsub = vsub.T
        else:
            vsub = vsub.rechunk(-1)
        with da.config.set(**{"optimization.fuse.ave-width": vsub.shape[0]}):
            return da.optimize(smooth_corr(vsub, ridx, cidx, freq=freq))[0]

    for _, eg_sub in egd_same.groupby("part_src"):
        pixels = list(set(eg_sub["source"]) | set(eg_sub["target"]))
        corr_ls.append(construct_comput(eg_sub, pixels))
        idx_ls.append(eg_sub.index)
        npxs.append(len(pixels))
    pixels = set()
    eg_ls = []
    grp = np.arange(len(egd_diff)) // step_size
    for igrp, eg_sub in egd_diff.sort_values("source").groupby(grp):
        pixels = pixels | set(eg_sub["source"]) | set(eg_sub["target"])
        eg_ls.append(eg_sub)
        if (len(pixels) > chunk - step_size / 2) or igrp == max(grp):
            pixels = list(pixels)
            edf = pd.concat(eg_ls)
            corr_ls.append(construct_comput(edf, pixels))
            idx_ls.append(edf.index)
            npxs.append(len(pixels))
            pixels = set()
            eg_ls = []
    print("pixel recompute ratio: {}".format(sum(npxs) / G.number_of_nodes()))
    print("computing correlations")
    corr_ls = da.compute(corr_ls)[0]
    corr = pd.Series(np.concatenate(corr_ls), index=np.concatenate(idx_ls), name="corr")
    eg_df["corr"] = corr
    return eg_df


def adj_corr(
    varr: xr.DataArray, adj: np.ndarray, nod_df: pd.DataFrame, freq: float
) -> scipy.sparse.csr_matrix:
    """
    Compute correlation in an optimized fashion given an adjacency matrix and
    node attributes.

    Wraps around :func:`graph_optimize_corr` and construct computation graph
    from `adj` and `nod_df`. Also convert the result into a sparse matrix with
    same shape as `adj`.

    Parameters
    ----------
    varr : xr.DataArray
        Input time series. Should have "frame" dimension in addition to column
        names of `nod_df`.
    adj : np.ndarray
        Adjacency matrix.
    nod_df : pd.DataFrame
        Dataframe containing node attributes. Should have length `adj.shape[0]`
        and only contain columns relevant to index the time series.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.

    Returns
    -------
    adj_corr : scipy.sparse.csr_matrix
        Sparse matrix of the same shape as `adj` but with values corresponding
        the computed correlation.
    """
    G = nx.Graph()
    G.add_nodes_from([(i, d) for i, d in enumerate(nod_df.to_dict("records"))])
    G.add_edges_from([(s, t) for s, t in zip(*adj.nonzero())])
    corr_df = graph_optimize_corr(varr, G, freq, idx_dims=nod_df.columns)
    return scipy.sparse.csr_matrix(
        (corr_df["corr"], (corr_df["source"], corr_df["target"])), shape=adj.shape
    )


def adj_list(G: nx.Graph) -> List[np.ndarray]:
    """
    Generate adjacency list representation from graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    adj_ls : List[np.ndarray]
        The adjacency list representation of graph.
    """
    gdict = nx.to_dict_of_dicts(G)
    return [np.array(list(gdict[k].keys())) for k in sorted(gdict.keys())]


@darr.as_gufunc(signature="(p,f),(i),(i)->(i)", output_dtypes=[float])
def smooth_corr(
    X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray, freq: float
) -> np.ndarray:
    """
    Wraps around :func:`filt_fft_vec` and :func:`idx_corr` to carry out both
    smoothing and computation of partial correlation.

    Parameters
    ----------
    X : np.ndarray
        Input time series.
    ridx : np.ndarray
        Row index of the resulting correlation.
    cidx : np.ndarray
        Column index of the resulting correlation.
    freq : float
        Cut-off frequency for the smoothing.

    Returns
    -------
    corr : np.ndarray
        Resulting partial correlation.
    """
    if freq:
        X = filt_fft_vec(X, freq, "low")
    return idx_corr(X, ridx, cidx)


@nb.jit(nopython=True, nogil=True, cache=True)
def idx_corr(X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray) -> np.ndarray:
    """
    Compute partial pairwise correlation based on index.

    This function compute a subset of a pairwise correlation matrix. The
    correlation to be computed are specified by two vectors `ridx` and `cidx` of
    same length, representing the row and column index of the full correlation
    matrix. The function use them to index the timeseries matrix `X` and compute
    only the requested correlations. The result is returned flattened.

    Parameters
    ----------
    X : np.ndarray
        Input time series. Should have 2 dimensions, where the last dimension
        should be the time dimension.
    ridx : np.ndarray
        Row index of the correlation.
    cidx : np.ndarray
        Column index of the correlation.

    Returns
    -------
    res : np.ndarray
        Flattened resulting correlations. Has same shape as `ridx` or `cidx`.
    """
    res = np.zeros(ridx.shape[0])
    std = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
        std[i] = np.sqrt((X[i, :] ** 2).sum())
    for i, (r, c) in enumerate(zip(ridx, cidx)):
        cur_std = std[r] * std[c]
        if cur_std > 0:
            res[i] = (X[r, :] * X[c, :]).sum() / cur_std
        else:
            res[i] = 0
    return res


def update_background(
    Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray, b: xr.DataArray = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Update background terms given spatial and temporal components of cells.

    A movie representation (with dimensions "height" "width" and "frame") of
    estimated cell activities are computed as the product between the spatial
    components matrix and the temporal components matrix of cells over the
    "unit_id" dimension. Then the residule movie is computed by subtracting the
    estimated cell activity movie from the input movie. Then the spatial
    footprint of background `b` is the mean of the residule movie over "frame"
    dimension, and the temporal component of background `f` is the least-square
    solution between the residule movie and the spatial footprint `b`.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions ("frame", "height", "width").
    A : xr.DataArray
        Estimation of spatial footprints of cells. Should have dimensions
        ("unit_id", "height", "width").
    C : xr.DataArray
        Estimation of temporal activities of cells. Should have dimensions
        ("unit_id", "frame").
    b : xr.DataArray, optional
        Previous estimation of spatial footprint of background. If provided it
        will be returned as-is, and only temporal activity of background will be
        updated

    Returns
    -------
    b_new : xr.DataArray
        New estimation of the spatial footprint of background. Has
        dimensions ("height", "width").
    f_new : xr.DataArray
        New estimation of the temporal activity of background. Has dimension
        "frame".
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    AtC = compute_AtC(A, C)
    Yb = (Y - AtC).clip(0)
    Yb = save_minian(Yb.rename("Yb"), intpath, overwrite=True)
    if b is None:
        b_new = Yb.mean("frame").persist()
    else:
        b_new = b.persist()
    b_stk = (
        b_new.stack(spatial=["height", "width"])
        .transpose("spatial")
        .expand_dims("dummy", axis=-1)
        .chunk(-1)
    )
    Yb_stk = Yb.stack(spatial=["height", "width"]).transpose("spatial", "frame")
    f_new = darr.linalg.lstsq(b_stk.data, Yb_stk.data)[0]
    f_new = xr.DataArray(
        f_new.squeeze(), dims=["frame"], coords={"frame": Yb.coords["frame"]}
    ).persist()
    return b_new, f_new
