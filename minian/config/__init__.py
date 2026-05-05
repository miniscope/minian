"""Pipeline defaults: paths, Dask cluster sizing, and BLAS thread caps.

CPU-based ``n_workers`` defaults come from ``minian.minian_rs`` — see
:func:`minian.minian_rs.thread_allocation` (logical CPUs, optional
``worker_cpu_ratio``, and derived worker count). The default ratio is **2/3**
of ``(logical CPUs − reserve)`` (floored, at least one worker).

Dask ``distributed.LocalCluster`` memory / threads / chunk sizing come from
:class:`PipelineConfig` (``dask_worker_memory``, ``dask_threads_per_worker``,
``dask_chunk_target_mb``). The :func:`dask_worker_memory_limit` helpers return those
values when a pipeline config is active (see :func:`get_active_pipeline_config`);
otherwise they return the same built-in defaults (environment variables are not
read for cluster sizing).

Implementation is split under :mod:`minian.config` — :mod:`minian.config.cluster`,
:mod:`minian.config.pipeline_config`, and :mod:`minian.config.serialize`.
"""

from __future__ import annotations

from .cluster import (
    PipelineEnv,
    apply_blas_thread_env,
    apply_thread_env,
    dask_chunk_target_mb,
    dask_threads_per_worker,
    dask_worker_memory_limit,
    resolve_n_workers,
)
from .pipeline_config import (
    PipelineConfig,
    clear_active_pipeline_config,
    get_active_pipeline_config,
    set_active_pipeline_config,
)
from .serialize import (
    _pipeline_config_from_json_dict,
    build_pipeline_effective_record,
    load_pipeline_config,
    main,
    pipeline_config_to_jsonable,
    resolve_pipeline_config_candidate,
)

__all__ = [
    "PipelineConfig",
    "PipelineEnv",
    "apply_blas_thread_env",
    "apply_thread_env",
    "build_pipeline_effective_record",
    "clear_active_pipeline_config",
    "dask_chunk_target_mb",
    "dask_threads_per_worker",
    "dask_worker_memory_limit",
    "get_active_pipeline_config",
    "load_pipeline_config",
    "main",
    "pipeline_config_to_jsonable",
    "resolve_n_workers",
    "resolve_pipeline_config_candidate",
    "set_active_pipeline_config",
    "_pipeline_config_from_json_dict",
]
