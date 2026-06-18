"""Unit tests for :mod:`minian.motion_correction`.

Two tiers, both fast enough to run every CI cycle (the heavy notebook golden in
``test_pipeline.py`` stays the integration anchor):

* **Kernel** - the registration/warp primitives on hand-built frames, no
  simulator. A frame is shifted by a known amount and the estimator must recover
  it; the warp must round-trip; the FFT-cache fast path must match the plain one.
  Pure numpy, microseconds, fully deterministic.
* **Ground truth** - the full :func:`estimate_motion` / :func:`apply_transform`
  path against a minisim recording that ships the exact applied motion. The
  estimate is the *correction*, i.e. the negation of ``GroundTruth.shifts``, so it
  is scored with :func:`minisim.shift_rmse` (``correction=True``); ``align=True``
  absorbs the constant offset between the pipeline's registration template and
  minisim's zero-shift reference. See :mod:`minian.test._simulated`.
"""

import numpy as np
import pytest
from minisim import shift_rmse
from minisim.spec import BrainMotion

from ..motion_correction import (
    apply_transform,
    est_motion_perframe,
    estimate_motion,
    transform_perframe,
)
from ._simulated import PX_SIZE_UM, as_movie

# --- kernel: registration + warp primitives (no simulator) -----------------


def _smooth_frame(h: int = 96, w: int = 96) -> np.ndarray:
    """A smooth, non-periodic-enough texture for sub-pixel registration."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (np.sin(yy / 7.0) * np.cos(xx / 5.0) + 1.5).astype(np.float32)


@pytest.mark.parametrize("shift", [(4, -3), (-5, 2), (0, 6)])
def test_est_motion_perframe_recovers_integer_shift(shift):
    # Rolling content by (dy, dx) means the shift that registers it back is -(dy, dx);
    # that is the convention est_motion_perframe returns.
    rng = np.random.default_rng(0)
    dst = rng.random((64, 64)).astype(np.float32)
    src = np.roll(dst, shift, axis=(0, 1))
    mo = est_motion_perframe(src, dst, upsample=100)
    np.testing.assert_allclose(mo, [-shift[0], -shift[1]], atol=1e-6)


@pytest.mark.parametrize("shift", [(1.5, -2.5), (0.4, 3.2)])
def test_est_motion_perframe_subpixel(shift):
    # A sub-pixel content shift (applied with the same warp the pipeline uses) must
    # be recovered to within a fraction of a pixel by the parabolic peak refinement.
    dst = _smooth_frame()
    src = transform_perframe(dst, np.array(shift))
    mo = est_motion_perframe(src, dst, upsample=100)
    np.testing.assert_allclose(mo, [-shift[0], -shift[1]], atol=0.5)


def test_identical_frames_estimate_zero_shift():
    # The parabolic peak refinement leaves a ~1e-5 residual on a perfect
    # autocorrelation peak, so assert "negligibly small" rather than exactly zero.
    fm = _smooth_frame()
    np.testing.assert_allclose(est_motion_perframe(fm, fm, upsample=100), [0.0, 0.0], atol=1e-3)


def test_fft_cache_matches_plain_path():
    # The perf-branch fast path precomputes each frame's rfft2 and reuses it; passing
    # the cached transforms must give bit-identical shifts to computing them inline.
    dst = _smooth_frame()
    src = transform_perframe(dst, np.array([2.0, -1.0]))
    plain = est_motion_perframe(src, dst, upsample=100)
    cached = est_motion_perframe(
        src, dst, upsample=100, src_fft=np.fft.rfft2(src), dst_fft=np.fft.rfft2(dst)
    )
    np.testing.assert_array_equal(plain, cached)


def test_transform_perframe_roundtrip():
    # Shifting a frame then shifting back by the negation restores the interior
    # (edges are lost to the fill, so compare an inset window).
    fm = _smooth_frame()
    shifted = transform_perframe(fm, np.array([3.0, -4.0]))
    restored = transform_perframe(shifted, np.array([-3.0, 4.0]))
    np.testing.assert_allclose(restored[8:-8, 8:-8], fm[8:-8, 8:-8], atol=1e-4)


def test_transform_perframe_translates_by_the_given_shift():
    # A positive (dy, dx) moves content toward higher indices: a single hot pixel
    # lands shifted by exactly that amount.
    fm = np.zeros((32, 32), dtype=np.float32)
    fm[10, 12] = 1.0
    out = transform_perframe(fm, np.array([3.0, 2.0]))
    assert np.unravel_index(int(out.argmax()), out.shape) == (13, 14)


# --- ground truth: estimate_motion / apply_transform vs minisim -------------
#
# Tolerances reflect measured sub-pixel recovery on this fixture (aligned RMSE
# ~0.4-0.7 px); the 1.0 px bound leaves headroom for seed/platform variation while
# still failing loudly on a real regression.

_RMSE_TOL_PX = 1.0


def test_estimate_motion_recovers_random_walk(simulate_recording):
    rec = simulate_recording([BrainMotion(model="walk", walk_step_um=1.5, max_shift_um=10.0)])
    est = estimate_motion(as_movie(rec)).compute()
    rmse = shift_rmse(est.values, rec.ground_truth.shifts, correction=True, align=True)
    assert rmse < _RMSE_TOL_PX


def test_estimate_motion_recovers_known_trajectory(simulate_recording):
    # An exact, RNG-free trajectory: a ramp on height and a sinusoid on width, in
    # pixels, converted to the um the spec expects. estimate_motion must track it.
    n_frames = 120
    dy = np.linspace(0.0, 5.0, n_frames)
    dx = 3.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, n_frames))
    traj_um = [(float(y * PX_SIZE_UM), float(x * PX_SIZE_UM)) for y, x in zip(dy, dx)]
    rec = simulate_recording([BrainMotion(trajectory_um=traj_um, max_shift_um=10.0)])
    est = estimate_motion(as_movie(rec)).compute()
    rmse = shift_rmse(est.values, rec.ground_truth.shifts, correction=True, align=True)
    assert rmse < _RMSE_TOL_PX


def test_estimate_motion_on_static_movie_is_near_zero(simulate_recording):
    # No brain_motion step: the movie is motionless, so the estimate must stay near
    # zero (only sensor-noise jitter, well under a pixel RMS).
    rec = simulate_recording()
    assert rec.ground_truth.shifts is None
    est = estimate_motion(as_movie(rec)).compute()
    assert shift_rmse(est.values, np.zeros_like(est.values)) < _RMSE_TOL_PX


def test_apply_transform_sharpens_temporal_mean(simulate_recording):
    # Motion smears static structure across the time-average; correcting it should
    # sharpen the temporal mean. Compare a center crop to avoid the edge fill the
    # correction introduces at the frame borders.
    rec = simulate_recording([BrainMotion(model="walk", walk_step_um=1.5, max_shift_um=10.0)])
    mov = as_movie(rec)
    est = estimate_motion(mov).compute()
    corrected = apply_transform(mov, est).compute()
    crop = (slice(40, -40), slice(40, -40))
    raw_sharpness = float(np.asarray(mov.mean("frame"))[crop].std())
    corrected_sharpness = float(np.asarray(corrected.mean("frame"))[crop].std())
    assert corrected_sharpness > raw_sharpness
