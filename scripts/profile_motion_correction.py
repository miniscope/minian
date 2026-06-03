"""Profiling harness for minian's motion-correction step.

Two complementary measurements:

1. ``scaling``  - run :func:`minian.motion_correction.estimate_motion` on an
   increasing number of frames and time it. Run under the *synchronous* dask
   scheduler with a fixed ``chunk_nfm`` so the measurement reflects total
   algorithmic work (no worker-pool noise). This tests the claim that the cost
   is linear in N (~N-1 registrations). A threaded reference run is also taken.

2. ``profile`` - run the per-chunk numpy kernel
   :func:`minian.motion_correction.est_motion_chunk` directly on a materialized
   chunk and attribute time to individual functions with ``cProfile`` (and
   optionally ``line_profiler``). This is where we confirm *where* the time goes:
   ``phase_cross_correlation`` (FFT recompute) vs ``transform_perframe``
   (subpixel ``sitk.Resample`` template warps) vs averaging.

The registration cost is dominated by the FFT size (= frame size) and is
independent of pixel content, so raw frames are representative; preprocessing is
optional (``--preprocess``).

Usage examples
--------------
    python scripts/profile_motion_correction.py --mode both
    python scripts/profile_motion_correction.py --mode scaling --frames 100 250 500 1000 2000
    python scripts/profile_motion_correction.py --mode profile --profile-frames 500 --line-profile
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from contextlib import contextmanager

# allow running as `python scripts/profile_motion_correction.py` without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dask as da
import dask.array as darr
import numpy as np

from minian.motion_correction import (
    est_motion_chunk,
    est_motion_perframe,
    estimate_motion,
    transform_perframe,
)
from minian.utilities import load_videos


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_movie(dpath, pattern, max_frames, dtype, preprocess):
    """Load (optionally preprocess) the demo movie as a lazy xr.DataArray."""
    varr = load_videos(dpath, pattern=pattern, dtype=dtype)
    varr = varr.transpose("frame", "height", "width")
    if max_frames is not None:
        varr = varr.isel(frame=slice(0, max_frames))
    if preprocess:
        # match the pipeline's pre-MC steps so the template content is realistic
        from minian.preprocessing import denoise, remove_background

        varr = denoise(varr, method="median", ksize=7)
        varr = remove_background(varr, method="tophat", wnd=15)
    print(
        f"loaded movie: {varr.sizes['frame']} frames "
        f"@ {varr.sizes['height']}x{varr.sizes['width']} ({varr.dtype}), "
        f"preprocess={preprocess}"
    )
    return varr


@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    print(f"  {label}: {time.perf_counter() - t0:.2f}s")


# --------------------------------------------------------------------------- #
# 1. scaling study
# --------------------------------------------------------------------------- #
def run_scaling(varr, frame_counts, chunk_nfm, npart, alt_error, threaded_ref):
    avail = varr.sizes["frame"]
    frame_counts = [n for n in frame_counts if n <= avail]
    if not frame_counts:
        raise SystemExit(f"no requested frame count fits in {avail} available frames")

    print("\n" + "=" * 72)
    print("SCALING STUDY (synchronous scheduler, single-threaded => total work)")
    print(f"chunk_nfm={chunk_nfm}  npart={npart}  alt_error={alt_error}")
    print("=" * 72)
    header = f"{'frames':>8} {'#blocks':>8} {'time(s)':>10} {'ms/frame':>10} {'time/N*1e3':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    base_rate = None
    with da.config.set(scheduler="synchronous"):
        for n in frame_counts:
            va = varr.isel(frame=slice(0, n)).chunk(
                {"frame": chunk_nfm, "height": -1, "width": -1}
            )
            # warm the lazy video read into memory so we time MC, not disk I/O
            va = va.compute().chunk({"frame": chunk_nfm, "height": -1, "width": -1})
            nblocks = int(np.ceil(n / chunk_nfm))
            t0 = time.perf_counter()
            motion = estimate_motion(
                va, dim="frame", npart=npart, chunk_nfm=chunk_nfm, alt_error=alt_error
            ).compute()
            dt = time.perf_counter() - t0
            rate = dt / n
            if base_rate is None:
                base_rate = rate
            rows.append((n, nblocks, dt, rate * 1e3))
            print(f"{n:>8} {nblocks:>8} {dt:>10.2f} {rate*1e3:>10.3f} {rate*1e3:>12.3f}")
            assert motion.sizes["frame"] == n

    print("\nlinearity check (ms/frame should be ~flat if O(N)):")
    r0 = rows[0]
    for n, nb, dt, mspf in rows:
        factor_n = n / r0[0]
        factor_t = dt / r0[2]
        print(
            f"  {n:>6} frames: {factor_n:>5.1f}x frames -> {factor_t:>5.2f}x time"
            f"  (super/sub-linear if != {factor_n:.1f})"
        )

    if threaded_ref and len(frame_counts) >= 1:
        n = frame_counts[-1]
        va = (
            varr.isel(frame=slice(0, n))
            .compute()
            .chunk({"frame": chunk_nfm, "height": -1, "width": -1})
        )
        print(f"\nthreaded-scheduler reference @ {n} frames:")
        with da.config.set(scheduler="threads"):
            t0 = time.perf_counter()
            estimate_motion(
                va, dim="frame", npart=npart, chunk_nfm=chunk_nfm, alt_error=alt_error
            ).compute()
            dt_thr = time.perf_counter() - t0
        dt_syn = rows[-1][2]
        print(
            f"  synchronous={dt_syn:.2f}s  threads={dt_thr:.2f}s  "
            f"speedup={dt_syn/dt_thr:.2f}x"
        )


# --------------------------------------------------------------------------- #
# 2. cProfile breakdown of the numpy kernel
# --------------------------------------------------------------------------- #
# functions we care about attributing time to, matched by (filename-substr, func)
SUSPECTS = [
    ("_phase_cross_correlation", "phase_cross_correlation"),
    ("_phase_cross_correlation", "_upsampled_dft"),
    ("motion_correction", "est_motion_perframe"),
    ("motion_correction", "transform_perframe"),
    ("motion_correction", "est_motion_chunk"),
    ("motion_correction", "check_temp"),
    ("pocketfft", "fftn"),
    ("_basic", "fftn"),
]


def run_profile(varr, n, npart, alt_error, line_prof):
    print("\n" + "=" * 72)
    print(f"CPROFILE BREAKDOWN of est_motion_chunk on {n} frames (single chunk)")
    print(f"npart={npart}  alt_error={alt_error}")
    print("=" * 72)

    chunk = varr.isel(frame=slice(0, n)).values  # materialize a numpy chunk
    print(f"materialized chunk: {chunk.shape} {chunk.dtype} "
          f"({chunk.nbytes/1e6:.0f} MB)")

    pr = cProfile.Profile()
    pr.enable()
    tmp, motions = est_motion_chunk(
        chunk.copy(), None, npart=npart, alt_error=alt_error
    )
    pr.disable()

    stats = pstats.Stats(pr)
    stats.calc_callees()

    # total time
    total = sum(v[2] for v in stats.stats.values())  # tottime sum != wall; use ct
    print(f"\nestimated shifts: {motions.shape}; template: {tmp.shape}")

    # ----- attribute time to suspects -----
    print("\nkey-function attribution (cumtime = incl. children, "
          "tottime = self only):")
    print(f"  {'function':<42}{'ncalls':>10}{'tottime':>10}{'cumtime':>10}")
    print("  " + "-" * 70)
    seen = set()
    for fname_sub, func in SUSPECTS:
        for (fn, lineno, name), (cc, nc, tt, ct, callers) in stats.stats.items():
            if name == func and fname_sub in fn and (fn, name) not in seen:
                seen.add((fn, name))
                print(f"  {name:<42}{nc:>10}{tt:>10.3f}{ct:>10.3f}")

    # ----- empirical registration count vs N-1 -----
    pcc_calls = 0
    for (fn, lineno, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        if name == "phase_cross_correlation":
            pcc_calls += nc
    print(
        f"\nphase_cross_correlation calls = {pcc_calls}   "
        f"(N-1 = {n-1};  with alt_error, extra at reduction levels)"
    )

    # ----- top self-time functions -----
    print("\ntop 15 functions by self time (tottime):")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(15)
    print("\n".join("  " + ln for ln in s.getvalue().splitlines()[5:25]))

    if line_prof:
        run_line_profile(chunk, npart, alt_error)


def run_line_profile(chunk, npart, alt_error):
    from line_profiler import LineProfiler

    print("\n" + "=" * 72)
    print("LINE PROFILE of est_motion_chunk / est_motion_perframe / "
          "transform_perframe")
    print("=" * 72)
    lp = LineProfiler()
    lp.add_function(est_motion_perframe)
    lp.add_function(transform_perframe)
    wrapped = lp(est_motion_chunk)
    wrapped(chunk.copy(), None, npart=npart, alt_error=alt_error)
    lp.print_stats(output_unit=1e-3)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="demo_movies",
                    help="folder with msCam*.avi (default: demo_movies)")
    ap.add_argument("--pattern", default=r"msCam[0-9]+\.avi$")
    ap.add_argument("--mode", choices=["scaling", "profile", "both"], default="both")
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[100, 250, 500, 1000, 2000],
                    help="frame counts for the scaling study")
    ap.add_argument("--chunk-nfm", type=int, default=250,
                    help="frames per dask block / leaf chunk (default 250)")
    ap.add_argument("--npart", type=int, default=3)
    ap.add_argument("--alt-error", type=float, default=5)
    ap.add_argument("--profile-frames", type=int, default=500,
                    help="frames to materialize for the cProfile breakdown")
    ap.add_argument("--preprocess", action="store_true",
                    help="apply median denoise + tophat background removal first")
    ap.add_argument("--no-threaded-ref", action="store_true",
                    help="skip the threaded-scheduler reference run")
    ap.add_argument("--line-profile", action="store_true",
                    help="also run a line-by-line profile (needs line_profiler)")
    args = ap.parse_args()

    max_needed = max(args.frames + [args.profile_frames])
    varr = load_movie(
        args.data, args.pattern, max_needed, np.float32, args.preprocess
    )

    if args.mode in ("scaling", "both"):
        run_scaling(
            varr, args.frames, args.chunk_nfm, args.npart, args.alt_error,
            threaded_ref=not args.no_threaded_ref,
        )
    if args.mode in ("profile", "both"):
        run_profile(
            varr, args.profile_frames, args.npart, args.alt_error,
            line_prof=args.line_profile,
        )


if __name__ == "__main__":
    main()
