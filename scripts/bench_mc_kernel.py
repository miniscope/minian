"""Micro-benchmark for motion-correction kernel optimizations (b2).

Evaluates candidate replacements for the two hot per-frame operations against
the current implementations, measuring BOTH speed and accuracy-vs-current on
real demo frames. Nothing here changes library behavior; it only informs which
b2 changes are safe.

Two parts:

1. TRANSFORM (template/output warp) -- the current `transform_perframe` rigid
   path uses `sitk.Resample` (subpixel linear). Candidates: `cv2.warpAffine`
   (linear) and `scipy.ndimage.shift` (spline order 1). Accuracy is reported as
   max/mean abs pixel difference vs the sitk output over a sweep of known
   sub-pixel shifts; speed as per-call time.

2. REGISTRATION -- the current `est_motion_perframe` rigid path uses
   `skimage.phase_cross_correlation(upsample_factor=100, normalization=None)`
   (i.e. plain cross-correlation, not phase-normalized). Candidates:
   - skimage upsample sweep (10/20/50) to see precision vs speed
   - cv2.phaseCorrelate (phase-normalized, Hanning window)
   - manual rfft2 cross-correlation + parabolic subpixel (FFT-cacheable)
   Accuracy is reported as the shift difference vs the current method over
   consecutive demo frame pairs; speed as per-pair time.

Run in the minian-bleed env. Usage:
    python scripts/bench_mc_kernel.py --frames 60
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from minian.motion_correction import transform_perframe
from minian.utilities import load_videos


def load_frames(dpath, pattern, n):
    varr = load_videos(dpath, pattern=pattern, dtype=np.float32)
    varr = varr.transpose("frame", "height", "width").isel(frame=slice(0, n))
    return varr.compute().values


def timeit(fn, *a, repeat=5, **k):
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        best = min(best, time.perf_counter() - t0)
    return best, out


# --------------------------------------------------------------------------- #
# transform candidates (rigid translation by tx_coef = [dy, dx])
# --------------------------------------------------------------------------- #
def warp_cv2(fm, sh):
    dy, dx = float(sh[0]), float(sh[1])
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        fm, M, (fm.shape[1], fm.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_scipy(fm, sh):
    return ndimage.shift(fm, (sh[0], sh[1]), order=1, mode="constant", cval=0)


def bench_transform(frames):
    print("\n" + "=" * 72)
    print("TRANSFORM: cv2.warpAffine / scipy.ndimage.shift vs sitk (current)")
    print("=" * 72)
    fm = frames[len(frames) // 2]
    shifts = [(0.3, -0.7), (1.5, 2.5), (-3.2, 4.8), (10.0, -7.0)]
    # timing on a representative shift
    t_sitk, _ = timeit(transform_perframe, fm, np.array([1.5, 2.5]), repeat=20)
    t_cv2, _ = timeit(warp_cv2, fm, np.array([1.5, 2.5]), repeat=20)
    t_sp, _ = timeit(warp_scipy, fm, np.array([1.5, 2.5]), repeat=20)
    print(f"per-call: sitk={t_sitk*1e3:.2f}ms  cv2={t_cv2*1e3:.2f}ms "
          f"({t_sitk/t_cv2:.1f}x)  scipy={t_sp*1e3:.2f}ms ({t_sitk/t_sp:.1f}x)")
    print(f"{'shift(dy,dx)':>16}{'cv2 maxΔ':>12}{'cv2 meanΔ':>12}"
          f"{'scipy maxΔ':>12}{'scipy meanΔ':>12}  (vs sitk; frame range "
          f"{fm.min():.0f}-{fm.max():.0f})")
    for sh in shifts:
        ref = transform_perframe(fm, np.array(sh), fill=0)
        a = warp_cv2(fm, sh)
        b = warp_scipy(fm, sh)
        # compare interior (exclude the fill border, which differs by convention)
        m = 12
        ri, ai, bi = ref[m:-m, m:-m], a[m:-m, m:-m], b[m:-m, m:-m]
        print(f"{str(sh):>16}{np.abs(ai-ri).max():>12.4f}"
              f"{np.abs(ai-ri).mean():>12.5f}{np.abs(bi-ri).max():>12.4f}"
              f"{np.abs(bi-ri).mean():>12.5f}")


# --------------------------------------------------------------------------- #
# registration candidates
# --------------------------------------------------------------------------- #
def reg_skimage(src, dst, up):
    sh, _, _ = phase_cross_correlation(src, dst, upsample_factor=up,
                                       normalization=None)
    return -sh  # matches est_motion_perframe


def reg_cv2(src, dst, win):
    sh, _ = cv2.phaseCorrelate(src.astype(np.float64), dst.astype(np.float64),
                               window=win)
    # cv2 returns (dx, dy) shift of src relative to dst
    return np.array([sh[1], sh[0]])


def _xcorr_parabolic(src, dst):
    """Plain (un-normalized) cross-correlation peak with parabolic subpixel.

    Mirrors normalization=None. FFTs here are exactly what could be cached and
    reused across the per-frame loop.
    """
    f1 = np.fft.rfft2(src)
    f2 = np.fft.rfft2(dst)
    cc = np.fft.irfft2(f1 * np.conj(f2), s=src.shape)
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    sub = []
    for ax, p in enumerate(peak):
        n = cc.shape[ax]
        pm, pp = (p - 1) % n, (p + 1) % n
        idx_m = list(peak); idx_m[ax] = pm
        idx_p = list(peak); idx_p[ax] = pp
        cm, c0, cp = cc[tuple(idx_m)], cc[peak], cc[tuple(idx_p)]
        denom = (cm - 2 * c0 + cp)
        delta = 0.5 * (cm - cp) / denom if denom != 0 else 0.0
        s = p + delta
        if s > n // 2:
            s -= n
        sub.append(s)
    return np.array(sub)  # shift of src vs dst (to register, est uses -sh)


def reg_xcorr(src, dst):
    return -_xcorr_parabolic(src, dst)


def bench_registration(frames):
    print("\n" + "=" * 72)
    print("REGISTRATION: candidates vs skimage upsample=100 (current)")
    print("=" * 72)
    pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    # reference shifts (current)
    ref = np.array([reg_skimage(s, d, 100) for s, d in pairs])
    han = cv2.createHanningWindow((frames.shape[2], frames.shape[1]), cv2.CV_64F)

    def report(name, fn, *a):
        t, _ = timeit(lambda: [fn(s, d, *a) for s, d in pairs], repeat=3)
        got = np.array([fn(s, d, *a) for s, d in pairs])
        d = np.abs(got - ref)
        print(f"  {name:<34} {t/len(pairs)*1e3:>7.2f} ms/pair  "
              f"maxΔ={d.max():>7.3f}px  meanΔ={d.mean():>7.4f}px")

    t_ref, _ = timeit(lambda: [reg_skimage(s, d, 100) for s, d in pairs], repeat=3)
    print(f"  {'skimage upsample=100 (current)':<34} "
          f"{t_ref/len(pairs)*1e3:>7.2f} ms/pair  (reference)")
    report("skimage upsample=50", reg_skimage, 50)
    report("skimage upsample=20", reg_skimage, 20)
    report("skimage upsample=10", reg_skimage, 10)
    report("manual xcorr + parabolic", reg_xcorr)
    report("cv2.phaseCorrelate (Hanning)", lambda s, d: reg_cv2(s, d, han))
    report("cv2.phaseCorrelate (no window)", lambda s, d: reg_cv2(s, d, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_movies")
    ap.add_argument("--pattern", default=r"msCam[0-9]+\.avi$")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    frames = load_frames(args.data, args.pattern, args.frames)
    print(f"loaded {frames.shape} {frames.dtype}")
    bench_transform(frames)
    bench_registration(frames)


if __name__ == "__main__":
    main()
