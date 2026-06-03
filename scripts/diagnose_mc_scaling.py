"""Diagnose the across-chunk superlinear blowup in minian motion correction.

The scaling study showed the per-chunk kernel is ~linear (~80 ms/frame) but the
end-to-end dask path grows superlinearly as the number of blocks increases.
This script isolates the cause by separating and measuring:

  * graph-build time (estimate_motion call, incl. da.optimize) vs compute time
  * number of tasks in the optimized graph
  * number of est_motion_chunk *executions* (detects task duplication / rerun)
  * peak RSS during compute (detects memory pressure / thrash)

Controlled test: hold the TOTAL frame count fixed and vary chunk_nfm so the
block count changes (1, 2, 4, 8). If wall time grows with block count at fixed
N, the cost is orchestration/memory, not the kernel. A scheduler comparison
(synchronous vs threads) is included.

Usage:
    python scripts/diagnose_mc_scaling.py --frames 1000 --chunks 1000 500 250 125
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dask as da
import numpy as np
import psutil

import minian.motion_correction as mc
from minian.motion_correction import estimate_motion
from minian.utilities import load_videos


def load_movie(dpath, pattern, n, dtype=np.float32):
    varr = load_videos(dpath, pattern=pattern, dtype=dtype)
    varr = varr.transpose("frame", "height", "width").isel(frame=slice(0, n))
    varr = varr.compute()  # materialize into RAM so we measure MC, not disk I/O
    print(f"loaded {varr.sizes['frame']} frames @ "
          f"{varr.sizes['height']}x{varr.sizes['width']} ({varr.dtype})")
    return varr


class RSSSampler:
    """Sample process RSS in a background thread; report peak delta."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self.baseline = 0
        self.peak = 0

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    @property
    def peak_delta_mb(self):
        return (self.peak - self.baseline) / 1e6

    @property
    def peak_mb(self):
        return self.peak / 1e6


def count_graph_tasks(lazy):
    """Number of tasks in the (already optimized) dask graph behind a DataArray."""
    try:
        return len(dict(lazy.data.__dask_graph__()))
    except Exception:
        return -1


def instrument_chunk_counter():
    """Wrap est_motion_chunk to count executions; returns (restore_fn, counter)."""
    counter = {"n": 0}
    orig = mc.est_motion_chunk

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    mc.est_motion_chunk = wrapped

    def restore():
        mc.est_motion_chunk = orig

    return restore, counter, orig


def expected_chunk_calls(n, chunk_nfm, npart):
    """Theoretical est_motion_chunk executions (dask leaf+reduction tasks plus
    the inner numpy recursion inside each call)."""

    def inner_calls(m):
        # est_motion_chunk on m frames: 1 call here + recursive calls while m>npart
        if m <= 1:
            return 1
        total = 1
        while m > npart:
            groups = int(np.ceil(m / npart))
            # each group is one recursive call on ~npart frames -> inner_calls
            sizes = [len(a) for a in np.array_split(np.arange(m), groups)]
            total += sum(inner_calls(s) for s in sizes)
            m = groups
        return total

    nblocks = int(np.ceil(n / chunk_nfm))
    # leaf blocks
    sizes = [len(a) for a in np.array_split(np.arange(n), nblocks)]
    leaf = sum(inner_calls(s) for s in sizes)
    # reduction levels: groups of npart templates each
    red = 0
    m = nblocks
    while m > 1:
        groups = int(np.ceil(m / npart))
        # each reduction task runs est_motion_chunk on <=npart templates
        sizes = [len(a) for a in np.array_split(np.arange(m), groups)]
        red += sum(inner_calls(s) for s in sizes)
        m = groups
    return leaf + red, nblocks


def run_one(varr, n, chunk_nfm, npart, alt_error, scheduler):
    va = varr.isel(frame=slice(0, n)).chunk(
        {"frame": chunk_nfm, "height": -1, "width": -1}
    )

    # --- graph build (includes da.optimize inside est_motion_part) ---
    restore, counter, _ = instrument_chunk_counter()
    try:
        t0 = time.perf_counter()
        lazy = estimate_motion(
            va, dim="frame", npart=npart, chunk_nfm=chunk_nfm, alt_error=alt_error
        )
        t_build = time.perf_counter() - t0
        ntasks = count_graph_tasks(lazy)
        build_side_calls = counter["n"]  # calls made during graph build (should be ~0)

        # --- compute ---
        counter["n"] = 0
        with RSSSampler() as rss:
            t0 = time.perf_counter()
            with da.config.set(scheduler=scheduler):
                lazy.compute()
            t_compute = time.perf_counter() - t0
        exec_calls = counter["n"]
    finally:
        restore()

    exp_calls, nblocks = expected_chunk_calls(n, chunk_nfm, npart)
    return {
        "n": n,
        "chunk_nfm": chunk_nfm,
        "nblocks": nblocks,
        "scheduler": scheduler,
        "t_build": t_build,
        "t_compute": t_compute,
        "ntasks": ntasks,
        "exec_calls": exec_calls,
        "exp_calls": exp_calls,
        "build_side_calls": build_side_calls,
        "peak_mb": rss.peak_mb,
        "peak_delta_mb": rss.peak_delta_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_movies")
    ap.add_argument("--pattern", default=r"msCam[0-9]+\.avi$")
    ap.add_argument("--frames", type=int, default=1000,
                    help="fixed total frame count for the controlled test")
    ap.add_argument("--chunks", type=int, nargs="+", default=[1000, 500, 250, 125],
                    help="chunk_nfm values to sweep (block count = frames/chunk)")
    ap.add_argument("--npart", type=int, default=3)
    ap.add_argument("--alt-error", type=float, default=5)
    ap.add_argument("--schedulers", nargs="+", default=["synchronous", "threads"])
    args = ap.parse_args()

    varr = load_movie(args.data, args.pattern, args.frames)

    print("\n" + "=" * 100)
    print(f"CONTROLLED TEST: fixed N={args.frames}, varying chunk_nfm "
          f"(npart={args.npart}, alt_error={args.alt_error})")
    print("If t_compute grows with #blocks at fixed N -> orchestration/memory, "
          "not the kernel.")
    print("=" * 100)
    hdr = (f"{'sched':>12}{'chunk':>7}{'blocks':>7}{'t_build':>9}{'t_compute':>11}"
           f"{'tasks':>8}{'exec':>7}{'exp':>7}{'peakRSS_MB':>12}{'ΔRSS_MB':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for scheduler in args.schedulers:
        for chunk_nfm in args.chunks:
            r = run_one(varr, args.frames, chunk_nfm, args.npart,
                        args.alt_error, scheduler)
            rows.append(r)
            print(f"{r['scheduler']:>12}{r['chunk_nfm']:>7}{r['nblocks']:>7}"
                  f"{r['t_build']:>9.2f}{r['t_compute']:>11.2f}{r['ntasks']:>8}"
                  f"{r['exec_calls']:>7}{r['exp_calls']:>7}"
                  f"{r['peak_mb']:>12.0f}{r['peak_delta_mb']:>10.0f}")

    # --- interpretation aids ---
    print("\nPer-scheduler: t_compute vs block count (normalized to 1-block case):")
    for scheduler in args.schedulers:
        srows = [r for r in rows if r["scheduler"] == scheduler]
        srows.sort(key=lambda r: r["nblocks"])
        base = srows[0]["t_compute"]
        for r in srows:
            print(f"  {scheduler:>12} {r['nblocks']:>2} blocks: "
                  f"{r['t_compute']:>7.2f}s  ({r['t_compute']/base:>4.2f}x 1-block)  "
                  f"exec/exp={r['exec_calls']}/{r['exp_calls']}"
                  f"{'  <-- DUPLICATION' if r['exec_calls'] > r['exp_calls']*1.05 else ''}")


if __name__ == "__main__":
    main()
