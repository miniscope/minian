"""Motion-domain step: rigid x,y brain motion — the tissue→sensor boundary.

``brain_motion`` is the single step in the motion domain and the hinge of the
whole reference-frame design. Everything before it (cells, neuropil, bleaching)
is **brain-frame** content that moves with the tissue; everything after it
(vignette, leakage, sensor) is **sensor-frame** and static. This step is where
the brain frame is translated relative to the static sensor.

To keep the motion honest, the upstream tissue steps render on a canvas **larger
than the sensor** (``Scene.zeros(acq, margin_px=…)``): real, simulated tissue
sits just off the sensor FOV. This step shifts that canvas per frame and crops
the centered sensor FOV back out, so the content that moves in at the edges is
genuine tissue, never a fabricated fill — and because the FOV crop never reaches
the canvas edge (the margin is ≥ the maximum shift), no edge fill ever enters the
result. The per-frame displacement is recorded to ground truth as the motion a
correction stage must estimate.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import shift as ndimage_shift

from minian.simulation.scene import MOVIE_DIMS, Scene
from minian.simulation.steps.base import Step


def bounded_random_walk(
    n_frames: int, step_px: float, max_px: float, rng: np.random.Generator
) -> np.ndarray:
    """A 2-D random walk of per-frame ``(dy, dx)`` positions, bounded to a disk.

    Starts at ``(0, 0)`` on frame 0 (the reference view) and adds an independent
    ``Normal(0, step_px)`` increment per axis each frame; whenever the cumulative
    displacement would leave the radius-``max_px`` disk it is rescaled back onto
    the boundary. Returns an ``(n_frames, 2)`` array in pixels. The bound keeps
    motion within the tissue margin so the FOV crop always lands on real content.
    Sequential by construction (each position depends on the previous), so an
    explicit loop — cheap at the recording lengths the simulator targets.
    """
    pos = np.zeros((n_frames, 2))
    for f in range(1, n_frames):
        cand = pos[f - 1] + rng.normal(0.0, step_px, size=2)
        magnitude = float(np.hypot(cand[0], cand[1]))
        if magnitude > max_px:
            cand *= max_px / magnitude
        pos[f] = cand
    return pos


def shift_and_crop(
    canvas: np.ndarray, shifts_px: np.ndarray, fov_shape: tuple[int, int]
) -> np.ndarray:
    """Shift each canvas frame by its ``(dy, dx)`` and crop the centered FOV.

    ``canvas`` is ``(frame, H, W)``; each frame is translated by ``shifts_px[f]``
    with bilinear interpolation (``order=1`` — sub-pixel, no overshoot) and the
    centered ``fov_shape`` window is cropped out. Positive ``dy``/``dx`` move
    content toward higher indices (down/right). The ``constant`` fill is never
    seen in the output: the margin ``(H − fov) / 2`` is ≥ the maximum shift, so
    the vacated edge strip always lies outside the crop window.
    """
    n_frames, canvas_h, canvas_w = canvas.shape
    fov_h, fov_w = fov_shape
    top = (canvas_h - fov_h) // 2
    left = (canvas_w - fov_w) // 2
    out = np.empty((n_frames, fov_h, fov_w))
    for f in range(n_frames):
        shifted = ndimage_shift(
            canvas[f], shifts_px[f], order=1, mode="constant", cval=0.0
        )
        out[f] = shifted[top : top + fov_h, left : left + fov_w]
    return out


class BrainMotionStep(Step):
    """Rigidly translate the brain-frame canvas per frame, then crop the sensor FOV.

    Resolves the per-frame ``(dy, dx)`` displacement — an explicit
    ``trajectory_um`` (converted µm→px) if given, else a :func:`bounded_random_walk`
    from ``walk_step_um``/``max_shift_um`` — then :func:`shift_and_crop`s the
    canvas down to the sensor FOV. The canvas must carry a tissue margin
    (``Scene.zeros(acq, margin_px=…)``) at least as large as the maximum shift;
    otherwise the crop would expose the canvas edge and this step raises (rather
    than silently filling with fabricated tissue). ``simulate()`` (Step 6) sizes
    the margin automatically from this spec.

    Writes ``shifts`` ``(n_frames, 2)`` to ground truth: the applied content
    displacement ``(dy, dx)`` in **pixels**, matching minian's ``shift_dim``
    order. A motion-correction stage estimates the *correction* — the negation of
    this — so the Step 10 RMSE test compares against ``−shifts``.
    """

    name = "brain_motion"
    domain = "motion"

    def __call__(self, scene: Scene) -> None:
        acq = self.acq
        canvas = scene.movie.values
        n_frames, canvas_h, canvas_w = canvas.shape
        fov = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
        margin_h, margin_w = (canvas_h - fov[0]) // 2, (canvas_w - fov[1]) // 2
        if (
            margin_h < 0
            or margin_w < 0
            or canvas_h - 2 * margin_h != fov[0]
            or canvas_w - 2 * margin_w != fov[1]
        ):
            raise ValueError(
                f"canvas {(canvas_h, canvas_w)} is not the sensor FOV {fov} plus a "
                "symmetric margin; allocate the scene with Scene.zeros(acq, margin_px=…)."
            )

        shifts_px = self._resolve_shifts(n_frames)
        self._check_within_margin(shifts_px, margin_h, margin_w)

        cropped = shift_and_crop(canvas, shifts_px, fov)
        scene.movie = xr.DataArray(
            cropped,
            dims=list(MOVIE_DIMS),
            coords={
                "frame": np.arange(n_frames),
                "height": np.arange(fov[0]),
                "width": np.arange(fov[1]),
            },
            name="movie",
        )
        scene.truth.shifts = shifts_px

    def _resolve_shifts(self, n_frames: int) -> np.ndarray:
        """Per-frame ``(dy, dx)`` in pixels — explicit trajectory or random walk."""
        spec, acq = self.spec, self.acq
        if spec.trajectory_um is not None:
            if len(spec.trajectory_um) != n_frames:
                raise ValueError(
                    f"trajectory_um has {len(spec.trajectory_um)} entries but the "
                    f"recording has {n_frames} frames; they must match."
                )
            return np.array(
                [[acq.um_to_px(dy), acq.um_to_px(dx)] for dy, dx in spec.trajectory_um]
            )
        return bounded_random_walk(
            n_frames,
            acq.um_to_px(spec.walk_step_um),
            acq.um_to_px(spec.max_shift_um),
            self.rng,
        )

    @staticmethod
    def _check_within_margin(shifts_px: np.ndarray, margin_h: int, margin_w: int) -> None:
        """Fail fast if any shift exceeds the tissue margin (the crop would expose
        the canvas edge), telling the caller to enlarge the margin."""
        if len(shifts_px) == 0:
            return
        max_dy = float(np.abs(shifts_px[:, 0]).max())
        max_dx = float(np.abs(shifts_px[:, 1]).max())
        if max_dy > margin_h + 1e-9 or max_dx > margin_w + 1e-9:
            raise ValueError(
                f"shift up to (dy={max_dy:.2f}, dx={max_dx:.2f}) px exceeds the tissue "
                f"margin ({margin_h}, {margin_w}) px; allocate the scene with a larger "
                "margin_px (simulate() sizes it from max_shift_um automatically)."
            )
