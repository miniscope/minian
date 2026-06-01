"""Per-stage recovery demo: motion correction (migration Step 10b).

This is the first test that feeds a simulated recording to a *real minian
pipeline stage* and checks it recovers the exact known ground truth — the
capability the end-to-end golden-number test cannot offer. Here the stage is
:func:`minian.motion_correction.estimate_motion`, validated against the injected
``GroundTruth.shifts``.

It is ``slow``-marked (off the per-PR path; run with ``pytest -m slow``): motion
estimation needs a realistic, minutes-scale-ish recording with enough texture and
frames to converge — a 1 s clip would not exercise it. A 30 s, 128 px, ~100-cell
recording recovers the trajectory to a quarter-pixel and runs in ~1 min.

This is a capability demonstration with a deliberately generous bound (~2x the
observed RMSE), not the finely-calibrated threshold suite — that lands with the
golden-test replacement in a later PR.
"""

import numpy as np
import pytest
import xarray as xr

from minian.motion_correction import estimate_motion
from minian.simulation import (
    Acquisition,
    BrainMotion,
    CellActivity,
    CellOptics,
    ImageSensor,
    Optics,
    PlaceSomata,
    Render,
    Sensor,
    Spec,
    shift_rmse,
    simulate,
)

pytestmark = pytest.mark.slow


def _moving_recording():
    """A 30 s, 128 px, ~100-cell shallow/in-focus recording with a 4 µm motion walk."""
    acq = Acquisition(
        fps=20.0,
        duration_s=30.0,
        optics=Optics(magnification=8.0, na=0.45, focal_plane_um=5.0, depth_of_field_um=60.0),
        image_sensor=ImageSensor(n_px_height=128, n_px_width=128, pixel_pitch_um=8.0, bit_depth=8),
    )
    spec = Spec(
        acquisition=acq,
        seed=5,
        steps=[
            PlaceSomata(density_per_mm2=6000.0, soma_radius_um=4.0, depth_range_um=(0.0, 15.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Render(),
            BrainMotion(walk_step_um=0.3, max_shift_um=4.0),
            # ~200 / NA² (NA 0.45): compensates the NA²-collection dimming so the
            # recorded movie (and thus motion recovery) is unchanged. See cell.py.
            Sensor(photons_per_unit=1000.0),
        ],
    )
    return simulate(spec)


def _as_movie(observed: np.ndarray) -> xr.DataArray:
    """Wrap the observed counts as the dask-chunked DataArray estimate_motion expects."""
    n, h, w = observed.shape
    return xr.DataArray(
        observed.astype("float32"),
        dims=["frame", "height", "width"],
        coords={"frame": np.arange(n), "height": np.arange(h), "width": np.arange(w)},
    ).chunk({"frame": 100, "height": -1, "width": -1})


def test_estimate_motion_recovers_true_shifts():
    rec = _moving_recording()
    gt = rec.ground_truth.shifts

    est = np.asarray(estimate_motion(_as_movie(rec.observed), dim="frame").compute())

    # minian estimates the *correction* (the negation of the applied motion),
    # relative to its own template frame; re-reference both trajectories to frame 0
    # and flip the estimate's sign before comparing in the same convention.
    est_aligned = -(est - est[0])
    gt_ref = gt - gt[0]

    rmse = shift_rmse(est_aligned, gt_ref)
    assert rmse < 0.5, f"motion recovery RMSE {rmse:.3f} px exceeds the 0.5 px bound"

    # secondary sanity check: the recovered trajectory tracks the truth per axis
    for axis in (0, 1):
        if gt_ref[:, axis].std() > 0:
            r = np.corrcoef(est_aligned[:, axis], gt_ref[:, axis])[0, 1]
            assert r > 0.9, f"axis {axis} trajectory correlation {r:.3f} too low"
