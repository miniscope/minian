"""Headless cross-registration across sessions (Dask progress + wall timers).

Run as ``python -m minian.pipelines.cross_reg`` or ``minian-cross-reg``
(``-d`` / ``--data`` defaults to ``.``; ``--param-dist`` defaults to ``5`` pixels).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Optional

import xarray as xr
from dask.diagnostics import ProgressBar

from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_centroids,
    calculate_mapping,
    fill_mapping,
    group_by_session,
    resolve_mapping,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.utilities import open_minian_mf
from minian.utilities.logger import (
    ANSIColor,
    configure_cli_logging,
    format_wall_duration,
    print_wall_elapsed,
    wall_section,
)

log = logging.getLogger(__name__)

WALL_PREFIX = "[MINIAN CROSS-REG]"

#: Default centroid-distance cutoff (pixels) for :func:`run_cross_reg` / CLI ``--param-dist``.
DEFAULT_PARAM_DIST: int = 5


class FileExtensions:
    PKL = ".pkl"
    NC = ".nc"


def set_window(wnd):
    return wnd == wnd.min()


def _positive_pixel_dist(value: str) -> int:
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return v


def parse_cross_reg_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI for :func:`run_cross_reg` (``argv`` defaults to ``sys.argv[1:]`` when ``None``)."""
    ap = argparse.ArgumentParser(
        description=(
            "Cross-register Minian outputs across sessions (writes mappings.pkl, "
            "cents.pkl, shiftds.nc under the data directory)."
        ),
    )
    ap.add_argument(
        "-d",
        "--data",
        default=".",
        dest="data",
        help=(
            "Directory containing session subfolders with minian results (e.g. "
            'session1/minian.nc). Default: "."'
        ),
    )
    ap.add_argument(
        "--param-dist",
        type=_positive_pixel_dist,
        default=DEFAULT_PARAM_DIST,
        metavar="PIXELS",
        dest="param_dist",
        help=(
            "Keep only cell pairs whose centroid Euclidean distance (height/width "
            f"coordinates) is strictly less than this value in pixels (default: {DEFAULT_PARAM_DIST})."
        ),
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    configure_cli_logging()
    t_total = time.perf_counter()

    args = parse_cross_reg_argv(argv)
    dpath = os.path.abspath(args.data)
    f_pattern = r"minian.nc$"
    id_dims = ["session"]
    param_dist = args.param_dist

    pbar = ProgressBar(minimum=2)
    pbar.register()

    log.info(
        "cross-reg: dpath=%r pattern=%r id_dims=%s param_dist=%s",
        dpath,
        f_pattern,
        id_dims,
        param_dist,
    )

    try:
        run_cross_reg(dpath, f_pattern, id_dims, param_dist)
    finally:
        pbar.unregister()
        elapsed_total = time.perf_counter() - t_total
        log.info(
            "cross-reg complete (total wall): %s",
            format_wall_duration(elapsed_total),
        )
        print_wall_elapsed(
            WALL_PREFIX,
            "cross-reg complete",
            elapsed_total,
            color=ANSIColor.BRIGHT_CYAN,
        )


def run_cross_reg(
    dpath: str, f_pattern: str, id_dims: list[str], param_dist: int
) -> None:
    """Execute cross-registration and write ``mappings.pkl``, ``cents.pkl``, ``shiftds.nc`` under ``dpath``."""
    with wall_section(WALL_PREFIX, "open_minian_mf", color=ANSIColor.BRIGHT_CYAN):
        minian_ds = open_minian_mf(dpath, id_dims, pattern=f_pattern)

    with wall_section(
        WALL_PREFIX,
        "estimate_motion (max_proj by session)",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        temps = minian_ds["max_proj"].rename("temps")
        shifts = estimate_motion(temps, dim="session").compute().rename("shifts")

    with wall_section(
        WALL_PREFIX,
        "apply_transform temps + merge shiftds",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        temps_sh = apply_transform(temps, shifts).compute().rename("temps_shifted")
        shiftds = xr.merge([temps, shifts, temps_sh])

    with wall_section(
        WALL_PREFIX,
        "apply_transform A (spatial footprints)",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        A_shifted = apply_transform(
            minian_ds["A"].chunk(dict(height=-1, width=-1)), shiftds["shifts"]
        )

    with wall_section(
        WALL_PREFIX,
        "window mask (broadcast + apply_ufunc)",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        window = shiftds["temps_shifted"].isnull().sum("session")
        window, _ = xr.broadcast(window, shiftds["temps_shifted"])
        window = xr.apply_ufunc(
            set_window,
            window,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
        )

    with wall_section(WALL_PREFIX, "calculate_centroids", color=ANSIColor.BRIGHT_CYAN):
        cents = calculate_centroids(A_shifted, window)

    id_work = list(id_dims)
    with wall_section(
        WALL_PREFIX,
        "centroid distances + filter + group_by_session",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        id_work.remove("session")
        dist = calculate_centroid_distance(cents, index_dim=id_work)
        dist_ft = dist[dist["variable", "distance"] < param_dist].copy()
        dist_ft = group_by_session(dist_ft)

    with wall_section(
        WALL_PREFIX,
        "calculate_mapping + resolve_mapping + fill_mapping",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        mappings = calculate_mapping(dist_ft)
        mappings_meta = resolve_mapping(mappings)
        mappings_meta_fill = fill_mapping(mappings_meta, cents)

    with wall_section(
        WALL_PREFIX,
        "persist outputs (pickle + netcdf)",
        color=ANSIColor.BRIGHT_CYAN,
    ):
        mappings_meta_fill.to_pickle(
            os.path.join(dpath, f"mappings{FileExtensions.PKL}")
        )
        cents.to_pickle(os.path.join(dpath, f"cents{FileExtensions.PKL}"))
        shiftds.to_netcdf(os.path.join(dpath, f"shiftds{FileExtensions.NC}"))


if __name__ == "__main__":
    main()
