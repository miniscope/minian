import contextlib
import os

import numba as nb
import numpy as np
import xarray as xr
from cv2 import GaussianBlur
from numpy import random

from ..cnmf import *
from ..initialization import *
from ..motion_correction import *
from ..preprocessing import *
from ..utilities import save_minian
from ..visualization import write_video


def gauss_cell(height: int, width: int, sigma: float, cov_coef: float, cent=None, nsamp=1000):
    # generate centroid
    if cent is None:
        cent = (random.randint(height), random.randint(width))
    # generate covariance
    while True:
        cov_var = random.rand(2, 2)
        cov_var = (cov_var + cov_var.T) / 2 * cov_coef
        cov = np.eye(2) * sigma + cov_var
        if np.all(np.linalg.eigvals(cov) > 0):
            break  # ensure cov is positive definite
    # generate samples of coordinates
    crds = np.clip(
        np.round(random.multivariate_normal(cent, cov, size=nsamp)).astype(int),
        0,
        None,
    )
    # generate spatial footprint
    A = np.zeros((height, width))
    for crd in np.unique(crds, axis=0):
        with contextlib.suppress(IndexError):
            A[tuple(crd)] = np.sum(np.all(crds == crd, axis=1))
    return A / A.max()


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_arcoef(s: np.ndarray, g: np.ndarray):
    c = np.zeros_like(s)
    for idx in range(len(g), len(s)):
        c[idx] = s[idx] + c[idx - len(g) : idx] @ g
    return c


def ar_trace(frame: int, pfire: float, g: np.ndarray):
    S = random.binomial(n=1, p=pfire, size=frame).astype(float)
    C = apply_arcoef(S, g)
    return C, S


def generate_data(
    ncell: int,
    dims: dict,
    sp_noise: float,
    tmp_noise: float,
    sp_sigma: float,
    sp_cov_coef: float,
    tmp_pfire: float,
    tmp_g_avg: float,
    tmp_g_var: float,
    bg_sigma=0,
    bg_strength=0,
    mo_sigma=0,
    cent=None,
):
    ff, hh, ww = (
        dims["frame"],
        dims["height"],
        dims["width"],
    )
    hh_pad, ww_pad = hh + mo_sigma * 4, ww + mo_sigma * 4
    if cent is None:
        cent = np.stack(
            (
                np.random.randint(0, hh_pad, size=ncell),
                np.random.randint(0, ww_pad, size=ncell),
            ),
            axis=1,
        )
    A = xr.DataArray(
        np.stack(
            [gauss_cell(hh_pad, ww_pad, sp_sigma, cov_coef=sp_cov_coef, cent=c) for c in cent]
        ),
        dims=["unit_id", "height", "width"],
        coords={
            "height": np.arange(hh_pad),
            "width": np.arange(ww_pad),
            "unit_id": np.arange(ncell),
        },
        name="A",
    )
    tmp_g = np.clip(random.normal(tmp_g_avg, tmp_g_var, size=ncell), a_min=0.8, a_max=0.95)
    traces = [ar_trace(ff, tmp_pfire, np.array([g])) for g in tmp_g]
    C = xr.DataArray(
        np.stack([t[0] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="C",
    )
    S = xr.DataArray(
        np.stack([t[1] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="S",
    )
    C_noise = C + random.normal(scale=tmp_noise, size=(ncell, ff))
    Y = C_noise.dot(A).rename("Y")
    Y = Y / Y.max()
    if bg_strength:
        A_bg = xr.apply_ufunc(
            GaussianBlur,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={
                "ksize": (int(hh // 2 * 2 - 1), int(ww // 2 * 2 - 1)),
                "sigmaX": bg_sigma,
            },
        )
        Y_bg = C_noise.dot(A_bg)
        Y_bg = Y_bg / Y_bg.max()
        Y = Y + Y_bg * bg_strength
    shifts = xr.DataArray(
        np.clip(
            random.normal(scale=mo_sigma, size=(ff, 2)),
            a_min=-2 * mo_sigma,
            a_max=2 * mo_sigma,
        ),
        dims=["frame", "variable"],
        coords={"frame": np.arange(ff), "variable": ["height", "width"]},
        name="shifts",
    )
    Y = (
        apply_shifts(Y, shifts)
        .compute()
        .isel(
            height=slice(2 * mo_sigma, -2 * mo_sigma),
            width=slice(2 * mo_sigma, -2 * mo_sigma),
        )
    )
    Y = (Y / Y.max() + random.normal(scale=sp_noise, size=(ff, hh, ww))).rename("Y")
    return (
        Y,
        A.isel(
            height=slice(2 * mo_sigma, -2 * mo_sigma),
            width=slice(2 * mo_sigma, -2 * mo_sigma),
        ),
        C,
        S,
        shifts,
    )


if __name__ == "__main__":
    # optimal parameters
    param_denoise = {"method": "median", "ksize": 7}
    param_background_removal = {"method": "tophat", "wnd": 15}
    param_estimate_shift = {"dim": "frame", "max_sh": 20}
    param_seeds_init = {
        "wnd_size": 100,
        "method": "rolling",
        "stp_size": 50,
        "nchunk": 100,
        "max_wnd": 15,
        "diff_thres": 3,
    }
    param_pnr_refine = {"noise_freq": 0.1, "thres": 1, "med_wnd": None}
    param_ks_refine = {"sig": 0.05}
    param_seeds_merge = {"thres_dist": 2, "thres_corr": 0.7, "noise_freq": 0.1}
    param_initialize = {"thres_corr": 0.8, "wnd": 15, "noise_freq": 0.1}
    param_get_noise = {"noise_range": (0.1, 0.5), "noise_method": "logmexp"}
    param_first_spatial = {
        "dl_wnd": 15,
        "sparse_penal": 0.1,
        "update_background": True,
        "normalize": True,
        "zero_thres": "eps",
    }
    param_first_temporal = {
        "noise_freq": 0.1,
        "sparse_penal": 1e-3,
        "p": 1,
        "add_lag": 20,
        "use_spatial": False,
        "jac_thres": 0.2,
        "zero_thres": 1e-8,
        "max_iters": 200,
        "use_smooth": True,
        "scs_fallback": False,
        "post_scal": True,
    }
    param_first_merge = {"thres_corr": 0.8}
    # generate toy data
    testpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "toy")
    Y, A, C, S, shifts = generate_data(
        ncell=100,
        dims={"height": 100, "width": 100, "frame": 1000},
        sp_noise=0.05,
        tmp_noise=0.08,
        sp_sigma=3,
        sp_cov_coef=2,
        tmp_pfire=0.02,
        tmp_g_avg=0.9,
        tmp_g_var=0.03,
        bg_sigma=20,
        bg_strength=1,
        mo_sigma=1,
    )
    Y_true = C.dot(A)
    Y_true = (
        ((Y_true - Y_true.min()) / (Y_true.max() - Y_true.min()) * 255)
        .astype(np.uint8)
        .rename("Y_true")
    )
    Y = ((Y - Y.min()) / (Y.max() - Y.min()) * 255).astype(np.uint8).chunk({"frame": 500})
    write_video(Y, "toy.mp4", testpath)
    for dat in [Y, Y_true, A, C, S, shifts]:
        save_minian(
            dat,
            dpath=testpath,
            fname="minian",
            backend="zarr",
            meta_dict={"session": -1, "animal": -2},
            overwrite=True,
        )
    # run pipeline
    print("pre-processing")
    Y_glow = (Y - Y.min("frame").compute()).rename("Y_glow")
    Y_dn = denoise(Y_glow, **param_denoise).rename("Y_denoise")
    Y_bg = remove_background(Y_dn, **param_background_removal).rename("Y_bg")
    shifts_est = estimate_shifts(Y_bg, **param_estimate_shift)
    Y_mc = apply_shifts(Y_bg, shifts_est).fillna(0).rename("Y_mc")
    for dat in [Y_glow, Y_dn, Y_bg, Y_mc, shifts_est]:
        save_minian(
            dat,
            dpath=testpath,
            fname="minian",
            backend="zarr",
            meta_dict={"session": -1, "animal": -2},
            overwrite=True,
        )
    print("initialization")
    Y_flt = Y_mc.compute().stack(spatial=["height", "width"])
    seeds_in = seeds_init(Y_mc, **param_seeds_init)
    seeds_pnr, _, _ = pnr_refine(Y_flt, seeds_in.copy(), **param_pnr_refine)
    seeds_ks = ks_refine(Y_flt, seeds_pnr[seeds_pnr["mask_pnr"]], **param_ks_refine)
    seeds_mrg = seeds_merge(
        Y_flt, seeds_ks[seeds_ks["mask_ks"]].reset_index(drop=True), **param_seeds_merge
    )
    A_init, C_init, b_init, f_init = initialize(
        Y_mc, seeds_mrg[seeds_mrg["mask_mrg"]], **param_initialize
    )
    print("cnmf")
    sn = get_noise_fft(Y_mc, **param_get_noise).persist()
    A_sp1, b_sp1, C_sp1, f_sp1 = update_spatial(
        Y_mc, A_init, b_init, C_init, f_init, sn, **param_first_spatial
    )
    YrA, C_tp1, S_tp1, B_tp1, C0_tp1, sig_tp1, g_tp1, scale_tp1 = update_temporal(
        Y_mc, A_sp1, b_sp1, C_sp1, f_sp1, sn, **param_first_temporal
    )
    A_tp1 = A_sp1.sel(unit_id=C_tp1.coords["unit_id"]).rename("A_tp1")
    A_mrg, sig_mrg = unit_merge(A_tp1, sig_tp1, [], **param_first_merge)
    print(A_mrg)
    # save results
    for dat in [
        A_init.rename("A_init").rename(unit_id="unit_id_init"),
        C_init.rename("C_init").rename(unit_id="unit_id_init"),
        b_init.rename("b_init"),
        f_init.rename("f_init"),
        sn.rename("sn"),
        A_sp1.rename("A_sp1"),
        b_sp1.rename("b_sp1"),
        C_sp1.rename("C_sp1"),
        f_sp1.rename("f_sp1"),
        C_tp1.rename("C_tp1"),
        S_tp1.rename("S_tp1"),
        sig_tp1.rename("sig_tp1"),
        A_mrg.rename("A_mrg"),
        sig_mrg.rename("sig_mrg"),
    ]:
        save_minian(
            dat,
            dpath=testpath,
            fname="minian",
            backend="zarr",
            meta_dict={"session": -1, "animal": -2},
            overwrite=True,
        )
