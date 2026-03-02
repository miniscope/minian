"""Default parameters for the source detection (CNMF) module.

Extracted from pipeline.ipynb parameter cells.
"""
from __future__ import annotations


def get_defaults() -> dict:
    """Return the default source detection config matching pipeline.ipynb."""
    return {
        # Initialization parameters
        "param_seeds_init": {
            "wnd_size": 1000,
            "method": "rolling",
            "stp_size": 500,
            "max_wnd": 15,
            "diff_thres": 3,
        },
        "param_pnr_refine": {"noise_freq": 0.06, "thres": 1},
        "param_ks_refine": {"sig": 0.05},
        "param_seeds_merge": {
            "thres_dist": 10,
            "thres_corr": 0.8,
            "noise_freq": 0.06,
        },
        "param_initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
        "param_init_merge": {"thres_corr": 0.8},
        # CNMF parameters
        "param_get_noise": {"noise_range": (0.06, 0.5)},
        "param_first_spatial": {
            "dl_wnd": 10,
            "sparse_penal": 0.01,
            "size_thres": (25, None),
        },
        "param_first_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 1,
            "p": 1,
            "add_lag": 20,
            "jac_thres": 0.2,
        },
        "param_first_merge": {"thres_corr": 0.8},
        "param_second_spatial": {
            "dl_wnd": 10,
            "sparse_penal": 0.01,
            "size_thres": (25, None),
        },
        "param_second_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 1,
            "p": 1,
            "add_lag": 20,
            "jac_thres": 0.4,
        },
    }
