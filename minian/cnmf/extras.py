"""Leaf routines previously split across tiny modules (`trace`,
`merge_units`, `temporal_update`, `projection`, `background`).
"""

import functools as fct
import logging
import os
import warnings
from typing import List, Optional, Tuple

import dask as da
import dask.array as darr
import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import xarray as xr

from ..utilities import (
    custom_arr_optimize,
    custom_delay_optimize,
    open_minian,
    save_minian,
)
from .graphs import adj_corr, label_connected
from .temporal import lstsq_vec, update_temporal_block

log = logging.getLogger(__name__)


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
    Y = Y.data
    A = darr.from_array(A.data.map_blocks(sparse.COO).compute(), chunks=-1)
    C = C.data.map_blocks(sparse.COO).T
    b = (
        b.fillna(0)
        .data.map_blocks(sparse.COO)
        .reshape((1, Y.shape[1], Y.shape[2]))
        .compute()
    )
    f = f.fillna(0).data.reshape((-1, 1))
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
    return YrA.transpose("unit_id", "frame")


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
    log.info("computing spatial overlap")
    with da.config.set(
        array_optimize=darr.optimization.optimize,
        **{"optimization.fuse.subgraphs": False},
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
    log.info("computing temporal correlation")
    nod_df = pd.DataFrame({"unit_id": A.coords["unit_id"].values})
    adj = adj_corr(C, A_inter, nod_df, freq=noise_freq)
    log.info("labeling units to be merged")
    adj = adj > thres_corr
    adj = adj + adj.T
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj,
        input_core_dims=[["unit_id", "unit_id_cp"]],
        output_core_dims=[["unit_id"]],
    )
    log.info("merging units")
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
    A, C, YrA = A.sel(unit_id=Ymask), C.sel(unit_id=Ymask), YrA.sel(unit_id=Ymask)
    log.info("grouping overlaping units")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).compute().astype(np.float32)
    A_inter = sparse.tensordot(A_sps, A_sps, axes=[(1, 2), (1, 2)])
    A_usum = np.tile(A_sps.sum(axis=(1, 2)).todense(), (A_sps.shape[0], 1))
    A_usum = A_usum + A_usum.T
    jac = scipy.sparse.csc_matrix(A_inter / (A_usum - A_inter) > jac_thres)
    unit_labels = label_connected(jac)
    YrA = YrA.assign_coords(unit_labels=("unit_id", unit_labels))
    log.info("updating temporal components")
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
    log.info("{} out of {} units dropped".format((~mask).sum().values, len(Ymask)))
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
        log.info("post-hoc scaling")
        scal = lstsq_vec(sig_new.data, YrA_new.data).compute().reshape((-1, 1))
        C_new, S_new, b0_new, c0_new = (
            C_new * scal,
            S_new * scal,
            b0_new * scal,
            c0_new * scal,
        )
    return C_new, S_new, b0_new, c0_new, g, mask


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


def update_background(
    Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray, b: Optional[xr.DataArray] = None
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
