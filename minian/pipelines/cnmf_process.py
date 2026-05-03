#!/usr/bin/env python
"""Headless minian CNMF process: preprocessing, motion correction, and CNMF (Dask LocalCluster).

Run as ``python -m minian.pipelines.cnmf_process``, ``python -m minian.pipeline``
(legacy alias), or the ``minian-pipeline`` console script.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import xarray as xr
from dask.distributed import Client, LocalCluster

from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.config import load_pipeline_config
from minian.constants import (
    MINIAN_CONFIG_FILENAME,
    get_minian_intermediate_path,
    minian_folder_under,
)
from minian.initialization import (
    initA,
    initC,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    TaskAnnotation,
    configure_logging,
    ensure_ffmpeg,
    get_optimal_chk,
    load_videos,
    save_minian,
)
from minian.utilities.logger import (
    ANSIColor,
    format_wall_duration,
    print_wall_elapsed,
    wall_section,
)
from minian.visualization import generate_videos, write_video

log = logging.getLogger(__name__)

WALL_PREFIX = "[MINIAN PIPELINE]"


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved filesystem locations for the demo pipeline."""

    root: Path
    dpath: str
    intpath: str
    param_save_minian: dict[str, Any]


def _spatial_chunks_full_frame() -> dict[str, int]:
    """Chunk spec for ``save_minian`` when persisting full 2D footprints per unit."""
    return {"unit_id": 1, "height": -1, "width": -1}


def _spatial_update_with_masked_c(
    Y_hw_chk: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    C_chk: xr.DataArray,
    sn_spatial: xr.DataArray,
    intpath: str,
    spatial_kw: dict[str, Any],
) -> Tuple[xr.DataArray, Any, Any, xr.DataArray, xr.DataArray]:
    """Run ``update_spatial`` and persist ``C_new`` / ``C_chk_new`` under ``intpath``."""
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **spatial_kw)
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"),
        intpath,
        overwrite=True,
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    return A_new, mask, norm_fac, C_new, C_chk_new


def _commit_spatial_round(
    A_new: xr.DataArray,
    C_new: xr.DataArray,
    C_chk_new: xr.DataArray,
    b_new: xr.DataArray,
    f_new: xr.DataArray,
    *,
    intpath: str,
    frame_chunk: int,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Write ``A``, ``b``, ``f``, ``C``, ``C_chk`` after a spatial + background pass."""
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks=_spatial_chunks_full_frame(),
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": frame_chunk}).rename("f"),
        intpath,
        overwrite=True,
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    return A, b, f, C, C_chk


def _save_yra_from_state(
    Y_fm_chk: xr.DataArray,
    A: xr.DataArray,
    b: xr.DataArray,
    C_chk: xr.DataArray,
    f: xr.DataArray,
    intpath: str,
) -> xr.DataArray:
    """Compute residual traces ``YrA`` from current state and persist under ``intpath``."""
    return save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )


def _save_minian_after_temporal(
    da: xr.DataArray,
    name: str,
    intpath: str,
) -> xr.DataArray:
    """Rename, rechunk (``unit_id``=1, ``frame``=-1), and persist under ``intpath``."""
    ut_chunk = {"unit_id": 1, "frame": -1}
    return save_minian(
        da.rename(name).chunk(ut_chunk),
        intpath,
        overwrite=True,
    )


def _persist_after_temporal(
    C_new: xr.DataArray,
    S_new: xr.DataArray,
    b0_new: xr.DataArray,
    c0_new: xr.DataArray,
    A: xr.DataArray,
    *,
    intpath: str,
    frame_chunk: int,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
]:
    """Save ``C``/``C_chk``/``S``/``b0``/``c0`` and subset ``A`` to surviving units."""
    C = _save_minian_after_temporal(C_new, "C", intpath)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": frame_chunk},
    )
    S = _save_minian_after_temporal(S_new, "S", intpath)
    b0 = _save_minian_after_temporal(b0_new, "b0", intpath)
    c0 = _save_minian_after_temporal(c0_new, "c0", intpath)
    A_out = A.sel(unit_id=C.coords["unit_id"].values)
    return C, C_chk, S, b0, c0, A_out


def _start_cluster(
    n_workers: int,
    worker_memory_limit: str,
    threads_per_worker: int,
    chunk_target_mb: int,
) -> Tuple[Client, LocalCluster]:
    """Start a :class:`~dask.distributed.LocalCluster` and client (or replace stale globals)."""
    _client = globals().get("client")
    _cluster = globals().get("cluster")
    if _client is not None or _cluster is not None:
        if _client is not None:
            _client.close()
        if _cluster is not None:
            _cluster.close()
        print("Closing previously found cluster")

    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit=worker_memory_limit,
        resources={"MEM": 1},
        threads_per_worker=threads_per_worker,
        dashboard_address=":8787",
    )
    cluster.scheduler.add_plugin(TaskAnnotation())
    client = Client(cluster)
    print(
        f"Started Dask LocalCluster at {cluster.scheduler.address!r}\n"
        f"  n_workers={n_workers}, memory_limit={worker_memory_limit!r}, "
        f"threads_per_worker={threads_per_worker}, chunk_target_mb={chunk_target_mb}\n"
        f"  (MINIAN_NWORKERS / MINIAN_WORKER_CPU_RATIO / MINIAN_WORKER_MEMORY / "
        f"MINIAN_THREADS_PER_WORKER / MINIAN_CHUNK_MB)\n"
        f"  dashboard {client.dashboard_link!r}"
    )
    return client, cluster


def parse_pipeline_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI for :func:`run_pipeline` (``argv`` defaults to ``sys.argv[1:]``-style parse)."""
    ap = argparse.ArgumentParser(
        description="Run minian headless pipeline with a Dask LocalCluster.",
    )
    ap.add_argument(
        "-d",
        "--data",
        default="./demo_movies/",
        help="Directory containing demo videos (absolutized). Default: ./demo_movies/",
    )
    ap.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH",
        dest="config",
        help=(
            f"Pipeline JSON (see PipelineConfig). Default: {MINIAN_CONFIG_FILENAME} "
            "in the current working directory if present; else built-in defaults."
        ),
    )
    ap.add_argument(
        "--worker-cpu-ratio",
        type=float,
        default=argparse.SUPPRESS,
        dest="worker_cpu_ratio",
        metavar="RATIO",
        help=(
            "When MINIAN_NWORKERS is unset: fraction of (logical CPUs − reserve) used "
            "as LocalCluster n_workers. If omitted, use MINIAN_WORKER_CPU_RATIO env or 2/3."
        ),
    )
    return ap.parse_args(argv)


def run_pipeline(
    data_dir: str,
    *,
    worker_cpu_ratio: Optional[float] = None,
    config_path: Optional[str] = None,
) -> None:
    """Execute the demo CNMF pipeline on ``data_dir`` (absolute or relative path)."""
    configure_logging(os.getenv("MINIAN_LOG_LEVEL", "INFO"), force=True)
    ensure_ffmpeg()
    t_pipeline_total = time.perf_counter()

    dpath = os.path.abspath(data_dir)
    root = Path(dpath).parent
    print(f"root: {root}")
    print(f"dpath: {dpath}")

    intpath = get_minian_intermediate_path(str(root))
    cfg = load_pipeline_config(path=config_path)
    cfg = dataclasses.replace(cfg, intpath=intpath)
    if worker_cpu_ratio is not None:
        cfg = dataclasses.replace(cfg, worker_cpu_ratio=worker_cpu_ratio)
    cfg.apply_environment()

    subset = dict(cfg.subset)
    subset_mc = cfg.subset_mc
    n_workers = cfg.resolved_n_workers()
    worker_memory_limit = os.getenv("MINIAN_WORKER_MEMORY", "2GB")
    threads_per_worker = int(os.getenv("MINIAN_THREADS_PER_WORKER", "2"))
    chunk_target_mb = int(os.getenv("MINIAN_CHUNK_MB", "200"))

    save_kw = dict(cfg.param_save_minian)
    save_kw["dpath"] = minian_folder_under(dpath)
    paths = PipelinePaths(
        root=root,
        dpath=dpath,
        intpath=intpath,
        param_save_minian=save_kw,
    )

    params = cfg.algorithm_param_dicts()

    client, cluster = _start_cluster(
        n_workers, worker_memory_limit, threads_per_worker, chunk_target_mb
    )
    try:
        varr = load_videos(paths.dpath, **params["param_load_videos"])
        chk, _ = get_optimal_chk(varr, dtype=float, csize=chunk_target_mb)

        with wall_section(
            WALL_PREFIX,
            "save_minian varr (initial chunk & write)",
            color=ANSIColor.BRIGHT_RED,
        ):
            varr = save_minian(
                varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename(
                    "varr"
                ),
                paths.intpath,
                overwrite=True,
            )

        varr_ref = varr.sel(subset)

        with wall_section(
            WALL_PREFIX,
            "varr_ref baseline (per-frame min, subtract)",
            color=ANSIColor.BRIGHT_RED,
        ):
            varr_min = varr_ref.min("frame").compute()
            varr_ref = varr_ref - varr_min

        with wall_section(
            WALL_PREFIX, "denoise and remove_background", color=ANSIColor.BRIGHT_RED
        ):
            varr_ref = denoise(varr_ref, **params["param_denoise"])
            varr_ref = remove_background(varr_ref, **params["param_background_removal"])

        with wall_section(
            WALL_PREFIX,
            "save_minian varr_ref (after denoise & background removal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            varr_ref = save_minian(
                varr_ref.rename("varr_ref"), dpath=paths.intpath, overwrite=True
            )

        with wall_section(WALL_PREFIX, "estimate_motion", color=ANSIColor.BRIGHT_RED):
            motion = estimate_motion(
                varr_ref.sel(subset_mc), **params["param_estimate_motion"]
            )

        with wall_section(
            WALL_PREFIX, "save_minian motion", color=ANSIColor.BRIGHT_RED
        ):
            motion = save_minian(
                motion.rename("motion").chunk({"frame": chk["frame"]}),
                **paths.param_save_minian,
            )

        Y = apply_transform(varr_ref, motion, fill=0)

        with wall_section(
            WALL_PREFIX,
            "save_minian Y_fm_chk and Y_hw_chk (motion-corrected movie)",
            color=ANSIColor.BRIGHT_RED,
        ):
            Y_fm_chk = save_minian(
                Y.astype(float).rename("Y_fm_chk"), paths.intpath, overwrite=True
            )
            Y_hw_chk = save_minian(
                Y_fm_chk.rename("Y_hw_chk"),
                paths.intpath,
                overwrite=True,
                chunks={
                    "frame": -1,
                    "height": chk["height"],
                    "width": chk["width"],
                },
            )

        with wall_section(
            WALL_PREFIX,
            "write_video minian_mc.mp4 (before / after MC)",
            color=ANSIColor.BRIGHT_RED,
        ):
            vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
            write_video(vid_arr, "minian_mc.mp4", paths.dpath)

        max_proj = save_minian(
            Y_fm_chk.max("frame").rename("max_proj"), **paths.param_save_minian
        ).compute()

        with wall_section(WALL_PREFIX, "seeds_init", color=ANSIColor.BRIGHT_RED):
            seeds = seeds_init(Y_fm_chk, **params["param_seeds_init"])

        with wall_section(WALL_PREFIX, "pnr_refine", color=ANSIColor.BRIGHT_RED):
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **params["param_pnr_refine"])

        with wall_section(WALL_PREFIX, "ks_refine", color=ANSIColor.BRIGHT_RED):
            seeds = ks_refine(Y_hw_chk, seeds, **params["param_ks_refine"])

        with wall_section(WALL_PREFIX, "seeds_merge", color=ANSIColor.BRIGHT_RED):
            seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(
                drop=True
            )
            seeds_final = seeds_merge(
                Y_hw_chk, max_proj, seeds_final, **params["param_seeds_merge"]
            )

        with wall_section(
            WALL_PREFIX, "initA and save_minian A_init", color=ANSIColor.BRIGHT_RED
        ):
            A_init = initA(
                Y_hw_chk,
                seeds_final[seeds_final["mask_mrg"]],
                **params["param_initialize"],
            )
            A_init = save_minian(A_init.rename("A_init"), paths.intpath, overwrite=True)

        with wall_section(
            WALL_PREFIX, "initC and save_minian C_init", color=ANSIColor.BRIGHT_RED
        ):
            C_init = initC(Y_fm_chk, A_init)
            C_init = save_minian(
                C_init.rename("C_init"),
                paths.intpath,
                overwrite=True,
                chunks={"unit_id": 1, "frame": -1},
            )

        with wall_section(
            WALL_PREFIX,
            "unit_merge (init) and save_minian A, C, C_chk",
            color=ANSIColor.BRIGHT_RED,
        ):
            A, C, _ = unit_merge(A_init, C_init, **params["param_init_merge"])
            A = save_minian(A.rename("A"), paths.intpath, overwrite=True)
            C = save_minian(C.rename("C"), paths.intpath, overwrite=True)
            C_chk = save_minian(
                C.rename("C_chk"),
                paths.intpath,
                overwrite=True,
                chunks={"unit_id": -1, "frame": chk["frame"]},
            )

        with wall_section(
            WALL_PREFIX,
            "update_background (initial) and save_minian f, b",
            color=ANSIColor.BRIGHT_RED,
        ):
            b, f = update_background(Y_fm_chk, A, C_chk)
            f = save_minian(f.rename("f"), paths.intpath, overwrite=True)
            b = save_minian(b.rename("b"), paths.intpath, overwrite=True)

        with wall_section(
            WALL_PREFIX,
            "get_noise_fft and save_minian sn_spatial",
            color=ANSIColor.BRIGHT_RED,
        ):
            sn_spatial = get_noise_fft(Y_hw_chk, **params["param_get_noise"])
            sn_spatial = save_minian(
                sn_spatial.rename("sn_spatial"), paths.intpath, overwrite=True
            )

        with wall_section(
            WALL_PREFIX,
            "update_spatial (first, param_first_spatial) and save_minian C_new, C_chk_new",
            color=ANSIColor.BRIGHT_RED,
        ):
            A_new, mask, norm_fac, C_new, C_chk_new = _spatial_update_with_masked_c(
                Y_hw_chk,
                A,
                C,
                C_chk,
                sn_spatial,
                paths.intpath,
                params["param_first_spatial"],
            )

        with wall_section(
            WALL_PREFIX,
            "update_background (after first spatial update)",
            color=ANSIColor.BRIGHT_RED,
        ):
            b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

        with wall_section(
            WALL_PREFIX,
            "save_minian A, b, f, C, C_chk (commit first spatial + background update)",
            color=ANSIColor.BRIGHT_RED,
        ):
            A, b, f, C, C_chk = _commit_spatial_round(
                A_new,
                C_new,
                C_chk_new,
                b_new,
                f_new,
                intpath=paths.intpath,
                frame_chunk=chk["frame"],
            )

        with wall_section(
            WALL_PREFIX,
            "save_minian YrA (compute_trace before first full temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            YrA = _save_yra_from_state(Y_fm_chk, A, b, C_chk, f, paths.intpath)

        with wall_section(
            WALL_PREFIX,
            "update_temporal (param_first_temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            C_new, S_new, b0_new, c0_new, _, _ = update_temporal(
                A, C, YrA=YrA, **params["param_first_temporal"]
            )

        with wall_section(
            WALL_PREFIX,
            "save_minian C, C_chk, S, b0, c0 and align A (after first full temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            C, C_chk, S, b0, c0, A = _persist_after_temporal(
                C_new,
                S_new,
                b0_new,
                c0_new,
                A,
                intpath=paths.intpath,
                frame_chunk=chk["frame"],
            )

        with wall_section(
            WALL_PREFIX, "unit_merge (param_first_merge)", color=ANSIColor.BRIGHT_RED
        ):
            A_mrg, C_mrg, add_mrg = unit_merge(
                A, C, [C + b0 + c0], **params["param_first_merge"]
            )
            assert add_mrg is not None
            sig_mrg = add_mrg[0]

        with wall_section(
            WALL_PREFIX,
            "save_minian A_mrg, C_mrg, C_chk (C_mrg_chk), sig_mrg (post-merge)",
            color=ANSIColor.BRIGHT_RED,
        ):
            A = save_minian(A_mrg.rename("A_mrg"), paths.intpath, overwrite=True)
            C = save_minian(C_mrg.rename("C_mrg"), paths.intpath, overwrite=True)
            C_chk = save_minian(
                C.rename("C_mrg_chk"),
                paths.intpath,
                overwrite=True,
                chunks={"unit_id": -1, "frame": chk["frame"]},
            )
            _ = save_minian(sig_mrg.rename("sig_mrg"), paths.intpath, overwrite=True)

        with wall_section(
            WALL_PREFIX,
            "update_spatial (second, param_second_spatial) and save_minian C_new, C_chk_new",
            color=ANSIColor.BRIGHT_RED,
        ):
            A_new, _, _, C_new, C_chk_new = _spatial_update_with_masked_c(
                Y_hw_chk,
                A,
                C,
                C_chk,
                sn_spatial,
                paths.intpath,
                params["param_second_spatial"],
            )

        with wall_section(
            WALL_PREFIX,
            "update_background (after second spatial update)",
            color=ANSIColor.BRIGHT_RED,
        ):
            b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

        with wall_section(
            WALL_PREFIX,
            "save_minian A, b, f, C, C_chk (commit second spatial + background update)",
            color=ANSIColor.BRIGHT_RED,
        ):
            A, b, f, C, C_chk = _commit_spatial_round(
                A_new,
                C_new,
                C_chk_new,
                b_new,
                f_new,
                intpath=paths.intpath,
                frame_chunk=chk["frame"],
            )

        with wall_section(
            WALL_PREFIX,
            "save_minian YrA (compute_trace before second full temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            YrA = _save_yra_from_state(Y_fm_chk, A, b, C_chk, f, paths.intpath)

        with wall_section(
            WALL_PREFIX,
            "update_temporal (param_second_temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            C_new, S_new, b0_new, c0_new, _, _ = update_temporal(
                A, C, YrA=YrA, **params["param_second_temporal"]
            )

        with wall_section(
            WALL_PREFIX,
            "save_minian C, C_chk, S, b0, c0 and align A (after second full temporal)",
            color=ANSIColor.BRIGHT_RED,
        ):
            C, C_chk, S, b0, c0, A = _persist_after_temporal(
                C_new,
                S_new,
                b0_new,
                c0_new,
                A,
                intpath=paths.intpath,
                frame_chunk=chk["frame"],
            )

        with wall_section(WALL_PREFIX, "generate_videos", color=ANSIColor.BRIGHT_RED):
            generate_videos(varr.sel(subset), Y_fm_chk, A=A, C=C_chk, vpath=paths.dpath)

        with wall_section(
            WALL_PREFIX,
            "save_minian final A, C, S, c0, b0, b, f to param_save_minian dpath",
            color=ANSIColor.BRIGHT_RED,
        ):
            A = save_minian(A.rename("A"), **paths.param_save_minian)
            C = save_minian(C.rename("C"), **paths.param_save_minian)
            S = save_minian(S.rename("S"), **paths.param_save_minian)
            c0 = save_minian(c0.rename("c0"), **paths.param_save_minian)
            b0 = save_minian(b0.rename("b0"), **paths.param_save_minian)
            b = save_minian(b.rename("b"), **paths.param_save_minian)
            f = save_minian(f.rename("f"), **paths.param_save_minian)
    finally:
        client.close()
        cluster.close()
        elapsed_total = time.perf_counter() - t_pipeline_total
        log.info(
            "pipeline complete (total wall): %s",
            format_wall_duration(elapsed_total),
        )
        print_wall_elapsed(
            WALL_PREFIX,
            "pipeline complete (total wall)",
            elapsed_total,
            color=ANSIColor.BRIGHT_RED,
        )


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for ``python -m minian.pipelines.cnmf_process`` and ``minian-pipeline``."""
    args = parse_pipeline_argv(argv)
    ratio = vars(args).get("worker_cpu_ratio")
    run_pipeline(
        args.data,
        worker_cpu_ratio=ratio,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
