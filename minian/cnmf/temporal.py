"""CNMF decomposition and helpers (combined module)."""

import warnings
from typing import Any, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
from scipy.linalg import lstsq, toeplitz
from scipy.ndimage import label
from scipy.sparse import dia_matrix
from statsmodels.tsa.stattools import acovf

from ..utilities import (
    med_baseline,
)
from .filters import filt_fft_vec
from .noise_estimation import noise_fft


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
    zero_thres_raw = kwargs.get("zero_thres")
    zero_thres_f = 0.0 if zero_thres_raw is None else float(zero_thres_raw)
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
                z = np.zeros(c.shape, dtype=float)
                return (z, z.copy(), z.copy(), z.copy())
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)

    def _cvx_vals(x: Any, shape: Tuple[int, ...]) -> np.ndarray:
        v = x.value
        if v is None:
            return np.zeros(shape, dtype=float)
        return np.asarray(v, dtype=float)

    c_val = _cvx_vals(c, c.shape)
    s_val = _cvx_vals(s, c.shape)
    b_val = _cvx_vals(b, b.shape)
    c0_val = _cvx_vals(c0, (int(c0.shape[0]),))
    c_arr = np.where(c_val > zero_thres_f, c_val, 0.0)
    s_arr = np.where(s_val > zero_thres_f, s_val, 0.0)
    b_arr = np.where(b_val > zero_thres_f, b_val, 0.0)
    c0_mid = c0_val.reshape((-1, 1)) * dc_vec
    c0_arr = np.where(c0_mid > zero_thres_f, c0_mid, 0.0)
    return c_arr, s_arr, b_arr, c0_arr


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
    y: np.ndarray,
    sn: float,
    p: int,
    add_lag: Union[int, str],
    pad: Optional[int] = None,
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
    add_lag : int or ``"p"``
        Additional lag in covariance, or ``"p"`` to use ``2 * p`` lags.
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
        if not isinstance(add_lag, int):
            raise TypeError("add_lag must be 'p' or an int")
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
    add_lag: Union[int, str] = "p",
    normalize=True,
    use_smooth=True,
    med_wd=None,
    concurrent=False,
    **kwargs,
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
    vec_get_ar_coef = np.vectorize(
        get_ar_coef,
        otypes=[float],
        excluded=["pad", "add_lag"],
        signature="(f),(),()->(l)",
    )
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
