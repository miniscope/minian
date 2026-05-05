"""CNMF-style spatial / temporal decomposition and related graph utilities."""

from .extras import (
    compute_AtC,
    compute_trace,
    unit_merge,
    update_background,
    update_temporal,
)
from .filters import filt_butter, filt_fft, filt_fft_vec, smooth_sig
from .graphs import (
    adj_corr,
    adj_list,
    graph_optimize_corr,
    idx_corr,
    label_connected,
    smooth_corr,
)
from .noise_estimation import get_noise_fft, get_noise_welch, noise_fft, noise_welch
from .spatial import (
    sps_any,
    update_spatial,
    update_spatial_block,
    update_spatial_perpx,
)
from .temporal import (
    get_ar_coef,
    get_p,
    lstsq_vec,
    update_temporal_block,
    update_temporal_cvxpy,
)

__all__ = (
    "adj_corr",
    "adj_list",
    "compute_AtC",
    "compute_trace",
    "filt_butter",
    "filt_fft",
    "filt_fft_vec",
    "get_ar_coef",
    "get_noise_fft",
    "get_noise_welch",
    "get_p",
    "graph_optimize_corr",
    "idx_corr",
    "label_connected",
    "lstsq_vec",
    "noise_fft",
    "noise_welch",
    "sps_any",
    "smooth_corr",
    "smooth_sig",
    "unit_merge",
    "update_background",
    "update_spatial",
    "update_spatial_block",
    "update_spatial_perpx",
    "update_temporal",
    "update_temporal_block",
    "update_temporal_cvxpy",
)
