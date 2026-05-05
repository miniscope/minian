"""The :class:`PipelineConfig` dataclass and process-wide active config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..constants import get_minian_intermediate_path
from .cluster import (
    PipelineEnv,
    apply_blas_thread_env,
    apply_thread_env,
    resolve_n_workers,
    _thread_env_same,
)


def _default_thread_env() -> Dict[str, str]:
    return _thread_env_same("1")


def _default_spatial_cnmf() -> Dict[str, Any]:
    return {
        "dl_wnd": 10,
        "sparse_penal": 0.01,
        "size_thres": (25, None),
    }


def _default_temporal_cnmf_core() -> Dict[str, Any]:
    return {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
    }


def _default_param_load_videos() -> Dict[str, Any]:
    return {
        "pattern": r"msCam[0-9]+\.avi$",
        "dtype": np.uint8,
        "downsample": {"frame": 1, "height": 1, "width": 1},
        "downsample_strategy": "subset",
    }


@dataclass
class PipelineConfig:
    """Algorithm defaults for a headless driver or JSON export.

    Video roots (``dpath``) are chosen by the driver; :mod:`minian.pipelines.cnmf_process` merges
    ``param_save_minian['dpath']`` at run time under the ``--data`` directory.
    Load from JSON with :func:`load_pipeline_config` or use these field defaults.
    """

    intpath: str = field(default_factory=get_minian_intermediate_path)
    subset: Mapping[str, Any] = field(default_factory=lambda: {"frame": slice(0, None)})
    subset_mc: Optional[Mapping[str, Any]] = None
    interactive: bool = True
    output_size: int = 100
    #: Use ``None`` to auto-pick from CPUs (see :func:`resolve_n_workers`).
    n_workers: Optional[int] = None
    reserve_cores_for_os: int = 1
    #: ``None`` → :attr:`PipelineEnv.DEFAULT_WORKER_CPU_RATIO` when deriving worker count from CPUs.
    worker_cpu_ratio: Optional[float] = None
    #: Applied by :meth:`apply_environment` unless ``blas_threads`` is passed there.
    thread_env: Dict[str, str] = field(default_factory=_default_thread_env)
    #: ``LocalCluster(memory_limit=...)`` string.
    dask_worker_memory: str = field(
        default_factory=lambda: PipelineEnv.DEFAULT_DASK_WORKER_MEMORY
    )
    dask_threads_per_worker: int = field(
        default=PipelineEnv.DEFAULT_DASK_THREADS_PER_WORKER
    )
    dask_chunk_target_mb: int = field(default=PipelineEnv.DEFAULT_DASK_CHUNK_TARGET_MB)
    param_save_minian: Dict[str, Any] = field(
        default_factory=lambda: {
            "meta_dict": {"session": -1, "animal": -2},
            "overwrite": True,
        }
    )
    param_load_videos: Dict[str, Any] = field(
        default_factory=_default_param_load_videos
    )
    param_denoise: Dict[str, Any] = field(
        default_factory=lambda: {"method": "median", "ksize": 7}
    )
    param_background_removal: Dict[str, Any] = field(
        default_factory=lambda: {"method": "tophat", "wnd": 15}
    )
    param_estimate_motion: Dict[str, Any] = field(
        default_factory=lambda: {"dim": "frame"}
    )
    param_seeds_init: Dict[str, Any] = field(
        default_factory=lambda: {
            "wnd_size": 1000,
            "method": "rolling",
            "stp_size": 500,
            "max_wnd": 15,
            "diff_thres": 3,
        }
    )
    param_pnr_refine: Dict[str, Any] = field(
        default_factory=lambda: {"noise_freq": 0.06, "thres": 1}
    )
    param_ks_refine: Dict[str, Any] = field(default_factory=lambda: {"sig": 0.05})
    param_seeds_merge: Dict[str, Any] = field(
        default_factory=lambda: {
            "thres_dist": 10,
            "thres_corr": 0.8,
            "noise_freq": 0.06,
        }
    )
    param_initialize: Dict[str, Any] = field(
        default_factory=lambda: {
            "thres_corr": 0.8,
            "wnd": 10,
            "noise_freq": 0.06,
        }
    )
    param_init_merge: Dict[str, Any] = field(
        default_factory=lambda: {"thres_corr": 0.8}
    )
    param_get_noise: Dict[str, Any] = field(
        default_factory=lambda: {"noise_range": (0.06, 0.5)}
    )
    param_first_spatial: Dict[str, Any] = field(default_factory=_default_spatial_cnmf)
    param_first_temporal: Dict[str, Any] = field(
        default_factory=lambda: {**_default_temporal_cnmf_core(), "jac_thres": 0.2}
    )
    param_first_merge: Dict[str, Any] = field(
        default_factory=lambda: {"thres_corr": 0.8}
    )
    param_second_spatial: Dict[str, Any] = field(default_factory=_default_spatial_cnmf)
    param_second_temporal: Dict[str, Any] = field(
        default_factory=lambda: {**_default_temporal_cnmf_core(), "jac_thres": 0.4}
    )

    def __post_init__(self) -> None:
        self.intpath = os.path.abspath(str(self.intpath))
        if self.thread_env:
            self.thread_env = {str(k): str(v) for k, v in self.thread_env.items()}
        mem = str(self.dask_worker_memory).strip()
        self.dask_worker_memory = mem or PipelineEnv.DEFAULT_DASK_WORKER_MEMORY
        self.dask_threads_per_worker = max(1, int(self.dask_threads_per_worker))
        self.dask_chunk_target_mb = max(1, int(self.dask_chunk_target_mb))

    def resolved_worker_cpu_ratio(self) -> float:
        """Effective ratio passed to Rust when :meth:`resolved_n_workers` uses CPU defaults."""
        if self.worker_cpu_ratio is not None:
            r = float(self.worker_cpu_ratio)
            if r != r or r <= 0.0:
                return PipelineEnv.DEFAULT_WORKER_CPU_RATIO
            return min(1.0, max(r, 1e-12))
        return PipelineEnv.DEFAULT_WORKER_CPU_RATIO

    def resolved_n_workers(self) -> int:
        if self.n_workers is not None:
            return max(1, int(self.n_workers))
        return resolve_n_workers(
            reserve=self.reserve_cores_for_os,
            worker_cpu_ratio=self.worker_cpu_ratio,
        )

    def algorithm_param_dicts(self) -> Dict[str, Dict[str, Any]]:
        """``param_*`` kwargs for CNMF stages, excluding ``param_save_minian`` (set per run).

        Keys match what :mod:`minian.pipelines.cnmf_process` passes to ``load_videos``, ``denoise``, etc.
        """
        skip = frozenset({"param_save_minian"})
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name.startswith("param_") and f.name not in skip
        }

    def with_paths_resolved(self) -> PipelineConfig:
        """Copy with ``intpath`` and ``param_save_minian['dpath']`` (if present) absolutized."""
        import copy

        c = copy.deepcopy(self)
        c.intpath = os.path.abspath(c.intpath)
        ps = dict(c.param_save_minian)
        if ps.get("dpath") not in (None, ""):
            ps["dpath"] = os.path.abspath(str(ps["dpath"]))
        c.param_save_minian = ps
        return c

    def apply_environment(self, *, blas_threads: Optional[int] = None) -> None:
        """Register this config as the process-wide active pipeline and apply BLAS/OpenMP env from :attr:`thread_env`.

        Downstream code reads :attr:`intpath` via :func:`get_active_pipeline_config`.
        """
        set_active_pipeline_config(self)
        if blas_threads is not None:
            apply_blas_thread_env(blas_threads)
        else:
            apply_thread_env(self.thread_env)


_active_pipeline_config: Optional[PipelineConfig] = None


def set_active_pipeline_config(cfg: PipelineConfig) -> None:
    """Set the active :class:`PipelineConfig` for this process (normally via :meth:`PipelineConfig.apply_environment`)."""
    global _active_pipeline_config
    _active_pipeline_config = cfg


def clear_active_pipeline_config() -> None:
    """Clear the active pipeline config (e.g. between tests)."""
    global _active_pipeline_config
    _active_pipeline_config = None


def get_active_pipeline_config() -> PipelineConfig:
    """Return the config last passed to :meth:`PipelineConfig.apply_environment` (or :func:`set_active_pipeline_config`)."""
    if _active_pipeline_config is None:
        raise RuntimeError(
            "No active pipeline configuration. Call PipelineConfig.apply_environment() "
            "after loading your run config (same object the driver uses)."
        )
    return _active_pipeline_config
