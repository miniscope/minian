"""Load/save pipeline JSON, effective-run records, and ``minian-pipeline-defaults`` CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

import numpy as np

from .._version import get_package_version
from ..constants import MINIAN_CONFIG_FILENAME
from .pipeline_config import PipelineConfig


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge dicts; override wins. Non-dicts replace."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override


def _from_jsonable(x: Any) -> Any:
    """Inverse of :func:`_to_jsonable` for JSON loaded with :func:`json.load`."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, list):
        return [_from_jsonable(v) for v in x]
    if isinstance(x, dict):
        if set(x.keys()) == {"__slice__"} and isinstance(x["__slice__"], list):
            s0, s1, s2 = x["__slice__"]
            return slice(s0, s1, s2)
        out: Dict[str, Any] = {}
        for k, v in x.items():
            if k == "dtype" and isinstance(v, str):
                out[k] = getattr(np, v, np.dtype(v))
            else:
                out[k] = _from_jsonable(v)
        return out
    return x


def _to_jsonable(x: Any) -> Any:
    """Convert values to JSON-friendly structures (slices, numpy dtypes, etc.)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, slice):
        return {
            "__slice__": [
                _to_jsonable(x.start),
                _to_jsonable(x.stop),
                _to_jsonable(x.step),
            ]
        }
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    mod = getattr(x, "__module__", "") or ""
    if mod.startswith("numpy"):
        if isinstance(x, np.dtype):
            return str(x)
        if isinstance(x, type):
            return x.__name__
    return str(x)


def pipeline_config_to_jsonable(
    cfg: PipelineConfig,
    *,
    resolve_paths: bool = False,
    include_resolved_workers: bool = False,
) -> Dict[str, Any]:
    """
    Export :class:`PipelineConfig` as a JSON-serializable dict.

    ``subset`` slices use ``{"__slice__": [start, stop, step]}``. NumPy dtypes
    in ``param_load_videos`` become dtype names (e.g. ``\"uint8\"``).

    ``resolve_paths=False`` matches the canonical **built-in** defaults used for
    digests and diffs in :func:`build_pipeline_effective_record`. ``True`` copies
    ``cfg`` with :meth:`PipelineConfig.with_paths_resolved` so ``intpath`` and any
    ``param_save_minian['dpath']`` are absolute, matching what a driver wrote to disk.

    ``include_resolved_workers`` adds ``resolved_n_workers`` and
    ``resolved_worker_cpu_ratio`` to the dict (not part of the dataclass fields).
    """
    if resolve_paths:
        cfg = cfg.with_paths_resolved()
    data = _to_jsonable(asdict(cfg))
    if include_resolved_workers:
        data["resolved_n_workers"] = cfg.resolved_n_workers()
        data["resolved_worker_cpu_ratio"] = cfg.resolved_worker_cpu_ratio()
    return data


def _pipeline_config_delta_json(run: Any, base: Any) -> Optional[Any]:
    """Return only the parts of ``run`` that differ from ``base`` (JSON-like trees).

    For dicts, keys are compared recursively; keys present only in ``run`` are
    included in full. For non-dict leaves, inequality uses ``==``; differing lists
    or scalars replace the subtree with the **entire** ``run`` value at that branch.
    Returns ``None`` when ``run == base`` (caller may normalize to ``{}``).
    """
    if run == base:
        return None
    if isinstance(run, dict) and isinstance(base, dict):
        out: Dict[str, Any] = {}
        for k in sorted(set(run) | set(base)):
            if k not in run:
                continue
            if k not in base:
                out[k] = run[k]
            else:
                sub = _pipeline_config_delta_json(run[k], base[k])
                if sub is not None:
                    out[k] = sub
        return out if out else None
    return run


def build_pipeline_effective_record(
    cfg: PipelineConfig,
    *,
    n_workers: int,
    worker_memory_limit: str,
    threads_per_worker: int,
    chunk_target_mb: int,
    cli_worker_cpu_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable audit record after a headless pipeline run.

    The CNMF driver writes this payload to
    :data:`~minian.constants.MINIAN_CONFIG_EFFECTIVE_FILENAME` (next to ``--data``)
    when a run completes successfully. It answers: which **minian version** ran,
    whether **built-in defaults** changed (digest), what **actually differed** from
    those defaults for this session, and how the **Dask cluster** was sized.

    **Baseline** for ``defaults_digest`` and ``delta_from_builtin_defaults`` is
    :func:`pipeline_config_to_jsonable` on a fresh :class:`PipelineConfig` with
    ``resolve_paths=False``. **Effective** config uses ``resolve_paths=True`` on
    ``cfg`` so paths match the merged driver state (e.g. ``intpath`` under the data
    directory).

    Parameters
    ----------
    cfg
        Final :class:`PipelineConfig` for the run (after the driver applies
        ``intpath``, optional CLI ``worker_cpu_ratio``, etc.).
    n_workers
        ``LocalCluster(n_workers=...)`` value used (same as ``cfg``\'s resolved
        worker count from :meth:`PipelineConfig.resolved_n_workers`).
    worker_memory_limit
        ``LocalCluster`` worker memory string (same as ``cfg.dask_worker_memory``).
    threads_per_worker
        ``cfg.dask_threads_per_worker``.
    chunk_target_mb
        Chunk budget (MB) from ``cfg.dask_chunk_target_mb``.
    cli_worker_cpu_ratio
        If the driver received ``--worker-cpu-ratio``, pass it here so it appears
        under ``resolved_cluster``; otherwise omit (key not present).

    Returns
    -------
    dict
        * ``timestamp`` — UTC time the record was built (ISO-8601, ``Z`` suffix).
        * ``minian_version`` — :func:`get_package_version` (installed package version).
        * ``defaults_digest`` — first 16 hex chars of SHA-256 of stable JSON
          (``sort_keys=True``, compact separators) of the baseline dict above.
        * ``delta_from_builtin_defaults`` — sparse nested dict of values that
          differ from the builtin baseline (empty ``{}`` if identical).
        * ``resolved_cluster`` — ``n_workers``, ``worker_memory_limit``,
          ``threads_per_worker``, ``chunk_target_mb``,
          ``resolved_worker_cpu_ratio`` (:meth:`PipelineConfig.resolved_worker_cpu_ratio`),
          and optionally ``cli_worker_cpu_ratio``.

    See Also
    --------
    pipeline_config_to_jsonable
    load_pipeline_config
    """
    baseline = pipeline_config_to_jsonable(
        PipelineConfig(),
        resolve_paths=False,
        include_resolved_workers=False,
    )
    baseline_blob = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    defaults_digest = hashlib.sha256(baseline_blob.encode()).hexdigest()[:16]

    effective = pipeline_config_to_jsonable(
        cfg,
        resolve_paths=True,
        include_resolved_workers=False,
    )
    delta = _pipeline_config_delta_json(effective, baseline)
    if delta is None:
        delta = {}

    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    resolved_cluster: Dict[str, Any] = {
        "n_workers": n_workers,
        "worker_memory_limit": worker_memory_limit,
        "threads_per_worker": threads_per_worker,
        "chunk_target_mb": chunk_target_mb,
        "resolved_worker_cpu_ratio": cfg.resolved_worker_cpu_ratio(),
    }
    if cli_worker_cpu_ratio is not None:
        resolved_cluster["cli_worker_cpu_ratio"] = cli_worker_cpu_ratio

    return {
        "timestamp": ts,
        "minian_version": get_package_version(),
        "defaults_digest": defaults_digest,
        "delta_from_builtin_defaults": delta,
        "resolved_cluster": resolved_cluster,
    }


def _pipeline_config_from_json_dict(raw: Dict[str, Any]) -> PipelineConfig:
    """Merge decoded JSON objects into :class:`PipelineConfig` field defaults."""
    raw = dict(raw)
    raw.pop("resolved_n_workers", None)
    raw.pop("resolved_worker_cpu_ratio", None)
    decoded = _from_jsonable(raw)
    defaults = PipelineConfig()
    fs = fields(PipelineConfig)
    base = {f.name: getattr(defaults, f.name) for f in fs}
    merged = _deep_merge(base, decoded)
    return PipelineConfig(**{f.name: merged[f.name] for f in fs})


def _expanded_config_path(path: Optional[str], cwd: Optional[str]) -> str:
    """Absolute path to JSON: either explicit ``path`` or ``<cwd>/<MINIAN_CONFIG_FILENAME>``."""
    if path is None:
        return os.path.join(os.path.abspath(cwd or os.getcwd()), MINIAN_CONFIG_FILENAME)
    return os.path.abspath(os.path.expanduser(path))


def resolve_pipeline_config_candidate(
    path: Optional[str] = None,
    *,
    cwd: Optional[str] = None,
) -> str:
    """
    Absolute path to the pipeline JSON that :func:`load_pipeline_config` opens first.

    If ``path`` is ``None``, the candidate is ``os.path.join`` of ``abs(cwd)`` and
    :data:`~minian.constants.MINIAN_CONFIG_FILENAME`. If ``path`` is set, the
    candidate is ``os.path.abspath(os.path.expanduser(path))``. The file may or may
    not exist; when missing,
    :func:`load_pipeline_config` returns built-in :class:`PipelineConfig` defaults.

    Parameters
    ----------
    path
        Explicit JSON path, or ``None`` for the conventional filename under ``cwd``.
    cwd
        Directory used when ``path`` is ``None``; defaults to :func:`os.getcwd`.

    See Also
    --------
    load_pipeline_config
    """
    return _expanded_config_path(path, cwd)


def load_pipeline_config(
    path: Optional[str] = None,
    *,
    cwd: Optional[str] = None,
) -> PipelineConfig:
    """
    Load pipeline JSON from disk, or return built-in defaults.

    Resolution order matches :func:`resolve_pipeline_config_candidate`: by default
    the file named :data:`~minian.constants.MINIAN_CONFIG_FILENAME` under ``cwd``,
    unless ``path`` points to another file.

    Decoded JSON is deep-merged into :class:`PipelineConfig` built-in defaults; only
    known dataclass fields are kept when constructing the result. Strips
    ``resolved_n_workers`` / ``resolved_worker_cpu_ratio`` if present in legacy files.

    If the candidate path is missing, returns a fresh :class:`PipelineConfig` with
    built-in defaults (same object shape drivers use before optional export). Invalid
    JSON or unreadable files propagate exceptions from :func:`json.load`.

    See Also
    --------
    resolve_pipeline_config_candidate
    build_pipeline_effective_record
    """
    candidate = resolve_pipeline_config_candidate(path, cwd=cwd)

    if not os.path.isfile(candidate):
        return PipelineConfig()

    with open(candidate, encoding="utf-8") as f:
        raw = json.load(f)

    return _pipeline_config_from_json_dict(raw)


def main() -> None:
    """Write default pipeline config JSON to ``--dest``/:data:`~minian.constants.MINIAN_CONFIG_FILENAME`, or ``--stdout``."""
    parser = argparse.ArgumentParser(
        description=(
            "Write default :class:`PipelineConfig` as JSON "
            "(for notebooks / pipeline drivers)."
        )
    )
    parser.add_argument(
        "--dest",
        "-d",
        default=".",
        metavar="DIR",
        help=(
            f"Directory where {MINIAN_CONFIG_FILENAME} is written (created if needed). "
            "Default: current directory. Use --stdout instead of creating a file."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing the file.",
    )
    parser.add_argument(
        "--resolve-paths",
        action="store_true",
        help="Use absolute intpath and param_save_minian['dpath'] (if set) before export.",
    )
    parser.add_argument(
        "--include-resolved-workers",
        action="store_true",
        help="Add resolved_n_workers and resolved_worker_cpu_ratio (CPU-based from JSON fields).",
    )
    args = parser.parse_args()
    cfg = PipelineConfig()
    payload = pipeline_config_to_jsonable(
        cfg,
        resolve_paths=args.resolve_paths,
        include_resolved_workers=args.include_resolved_workers,
    )
    text = json.dumps(payload, indent=2)
    if args.stdout:
        print(text)
        return
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    out_path = os.path.join(dest, MINIAN_CONFIG_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
