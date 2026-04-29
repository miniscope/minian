"""Pipeline defaults: paths, Dask ``n_workers``, and BLAS thread caps.

CPU-based ``n_workers`` defaults come only from ``minian.minian_rs`` — see
:func:`minian.minian_rs.thread_allocation` (logical CPUs and derived worker count).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .constants import MINIAN_CONFIG_FILENAME, get_minian_intermediate_path

# Keys shared by :func:`apply_blas_thread_env` and default :attr:`PipelineConfig.thread_env`.
_BLAS_THREAD_KEYS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)

__all__ = [
    "PipelineConfig",
    "apply_blas_thread_env",
    "apply_minian_intermediate",
    "apply_thread_env",
    "load_pipeline_config",
    "main",
    "pipeline_config_to_jsonable",
    "resolve_n_workers",
]


def _thread_env_same(limit: str) -> Dict[str, str]:
    """One string value applied to OMP/MKL/OpenBLAS."""
    return {k: limit for k in _BLAS_THREAD_KEYS}


def _env_nonempty_positive_int(var: str) -> Optional[int]:
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return None
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return None


def resolve_n_workers(
    *,
    reserve: int = 1,
    env_var: str = "MINIAN_NWORKERS",
) -> int:
    """
    Worker count for ``dask.distributed.LocalCluster(..., n_workers=...)``.

    If ``env_var`` is set to a positive integer string, that value is used (minimum 1).
    Otherwise defers to :func:`minian.minian_rs.thread_allocation` (requires the
    Rust extension).
    """
    from_env = _env_nonempty_positive_int(env_var)
    if from_env is not None:
        return from_env
    try:
        from minian.minian_rs import thread_allocation as _thread_allocation
    except ImportError as e:
        raise ImportError(
            "minian.minian_rs is required to resolve CPU-based n_workers "
            "(install / build the package so the Rust extension is present)."
        ) from e
    return int(_thread_allocation(reserve).cluster_workers)


def apply_thread_env(env: Mapping[str, Any]) -> None:
    """Apply key/value pairs to :func:`os.environ` (values are stringified)."""
    for k, v in env.items():
        os.environ[str(k)] = str(v)


def apply_blas_thread_env(threads: int = 1) -> None:
    """Set OMP/MKL/OpenBLAS thread caps to the same integer (Dask + NumPy workers)."""
    apply_thread_env(_thread_env_same(str(int(threads))))


def apply_minian_intermediate(intpath: str) -> None:
    os.environ["MINIAN_INTERMEDIATE"] = intpath


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

    Video roots and similar layout (what notebooks call ``dpath``) are intentionally
    not modeled here — set those in a notebook or in your own ``.py`` pipeline wrapper
    (e.g. pass paths into ``load_videos`` / ``param_save_minian``).
    """

    intpath: str = field(default_factory=get_minian_intermediate_path)
    subset: Mapping[str, Any] = field(
        default_factory=lambda: {"frame": slice(0, None)}
    )
    subset_mc: Optional[Mapping[str, Any]] = None
    interactive: bool = True
    output_size: int = 100
    #: Use ``None`` to auto-pick from CPUs (see :func:`resolve_n_workers`).
    n_workers: Optional[int] = None
    reserve_cores_for_os: int = 1
    #: Applied by :meth:`apply_environment` unless ``blas_threads`` is passed there.
    thread_env: Dict[str, str] = field(default_factory=_default_thread_env)
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
    param_ks_refine: Dict[str, Any] = field(
        default_factory=lambda: {"sig": 0.05}
    )
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
        if self.thread_env:
            self.thread_env = {str(k): str(v) for k, v in self.thread_env.items()}

    def resolved_n_workers(self) -> int:
        if self.n_workers is not None:
            return max(1, int(self.n_workers))
        return resolve_n_workers(reserve=self.reserve_cores_for_os)

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
        """Set ``MINIAN_INTERMEDIATE`` and BLAS/OpenMP env vars from :attr:`thread_env` (or ``blas_threads``)."""
        apply_minian_intermediate(self.intpath)
        if blas_threads is not None:
            apply_blas_thread_env(blas_threads)
        else:
            apply_thread_env(self.thread_env)


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
    """
    if resolve_paths:
        cfg = cfg.with_paths_resolved()
    data = _to_jsonable(asdict(cfg))
    if include_resolved_workers:
        data["resolved_n_workers"] = cfg.resolved_n_workers()
    return data


def _pipeline_config_from_json_dict(raw: Dict[str, Any]) -> PipelineConfig:
    """Merge decoded JSON objects into :class:`PipelineConfig` field defaults."""
    raw = dict(raw)
    raw.pop("resolved_n_workers", None)
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


def load_pipeline_config(
    path: Optional[str] = None,
    *,
    cwd: Optional[str] = None,
) -> PipelineConfig:
    """
    Load :data:`~minian.constants.MINIAN_CONFIG_FILENAME` from disk when present.

    If ``path`` is ``None``, uses ``cwd`` (default current working directory) and
    the standard filename; otherwise reads that file path directly.

    If the chosen path is not an existing file, returns a fresh
    :class:`PipelineConfig` with built-in defaults. Invalid JSON raises.
    """
    candidate = _expanded_config_path(path, cwd)

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
        help="Add resolved_n_workers (env MINIAN_NWORKERS or CPU-based).",
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
