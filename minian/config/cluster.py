"""Dask worker sizing, BLAS/OpenMP thread env, and :class:`PipelineEnv` defaults."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Tuple

# Keys shared by :func:`apply_blas_thread_env` and default :attr:`PipelineConfig.thread_env`.
_BLAS_THREAD_KEYS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


class PipelineEnv:
    """Default numeric/string constants for :class:`PipelineConfig` and helpers.

    Dask cluster sizing (``n_workers``, ``worker_cpu_ratio``, ``dask_*`` fields) is
    configured in pipeline JSON / :class:`PipelineConfig`, not via ``MINIAN_*`` shell
    variables. This class still names legacy env keys where they appear in docs or
    tooling, and holds defaults such as :attr:`DEFAULT_DASK_WORKER_MEMORY`.
    """

    #: Default ``worker_cpu_ratio`` for :func:`resolve_n_workers` (matches Rust ``DEFAULT_WORKER_CPU_RATIO``).
    DEFAULT_WORKER_CPU_RATIO: float = 2.0 / 3.0

    MINIAN_NWORKERS: str = "MINIAN_NWORKERS"
    MINIAN_WORKER_CPU_RATIO: str = "MINIAN_WORKER_CPU_RATIO"
    MINIAN_WORKER_MEMORY: str = "MINIAN_WORKER_MEMORY"
    MINIAN_THREADS_PER_WORKER: str = "MINIAN_THREADS_PER_WORKER"
    MINIAN_CHUNK_MB: str = "MINIAN_CHUNK_MB"

    DEFAULT_DASK_WORKER_MEMORY: str = "2GB"
    DEFAULT_DASK_THREADS_PER_WORKER: int = 2
    DEFAULT_DASK_CHUNK_TARGET_MB: int = 200


def _thread_env_same(limit: str) -> Dict[str, str]:
    """One string value applied to OMP/MKL/OpenBLAS."""
    return {k: limit for k in _BLAS_THREAD_KEYS}


def _get_active_pipeline_config():
    """Lazy import avoids import cycle with :mod:`minian.config.pipeline_config`."""
    from .pipeline_config import get_active_pipeline_config

    return get_active_pipeline_config()


def dask_worker_memory_limit() -> str:
    """``LocalCluster(memory_limit=...)`` string from the active :class:`PipelineConfig` or built-in default."""
    try:
        return _get_active_pipeline_config().dask_worker_memory
    except RuntimeError:
        return PipelineEnv.DEFAULT_DASK_WORKER_MEMORY


def dask_threads_per_worker() -> int:
    """``LocalCluster(threads_per_worker=...)`` from the active :class:`PipelineConfig` or built-in default."""
    try:
        return _get_active_pipeline_config().dask_threads_per_worker
    except RuntimeError:
        return PipelineEnv.DEFAULT_DASK_THREADS_PER_WORKER


def dask_chunk_target_mb() -> int:
    """Chunk budget (MB) for ``get_optimal_chk`` from the active :class:`PipelineConfig` or built-in default."""
    try:
        return _get_active_pipeline_config().dask_chunk_target_mb
    except RuntimeError:
        return PipelineEnv.DEFAULT_DASK_CHUNK_TARGET_MB


def resolve_n_workers(
    *,
    reserve: int = 1,
    worker_cpu_ratio: Optional[float] = None,
) -> int:
    """
    Worker count for ``dask.distributed.LocalCluster(..., n_workers=...)`` from CPUs.

    Uses :func:`minian.minian_rs.thread_allocation` (requires the Rust extension)
    with ``reserve`` and ``worker_cpu_ratio`` when that argument is a finite positive
    value (clamped to ``(0, 1]``); otherwise uses
    :attr:`PipelineEnv.DEFAULT_WORKER_CPU_RATIO` (``2/3``).

    Prefer :meth:`PipelineConfig.resolved_n_workers` in drivers; set
    :attr:`PipelineConfig.n_workers` in JSON for a fixed count.
    """
    try:
        from minian.minian_rs import thread_allocation as _thread_allocation
    except ImportError as e:
        raise ImportError(
            "minian.minian_rs is required to resolve CPU-based n_workers "
            "(install / build the package so the Rust extension is present)."
        ) from e
    ratio = worker_cpu_ratio
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        ratio = PipelineEnv.DEFAULT_WORKER_CPU_RATIO
    else:
        ratio = min(1.0, max(float(ratio), 1e-12))
    return int(_thread_allocation(reserve, ratio).cluster_workers)


def apply_thread_env(env: Mapping[str, Any]) -> None:
    """Apply key/value pairs to ``os.environ`` (values are stringified)."""
    for k, v in env.items():
        os.environ[str(k)] = str(v)


def apply_blas_thread_env(threads: int = 1) -> None:
    """Set OMP/MKL/OpenBLAS thread caps to the same integer (Dask + NumPy workers)."""
    apply_thread_env(_thread_env_same(str(int(threads))))
