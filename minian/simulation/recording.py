"""Typed simulator output — ``Recording`` / ``GroundTruth`` — and ``finalize()``.

Where :mod:`minian.simulation.scene` is the mutable working state a pipeline of
steps fills in, this module is the *frozen, typed result* that state is distilled
into once the pipeline is exhausted. :func:`finalize` is the transform: it turns a
run-out ``Scene`` into a ``Recording`` carrying the observed movie, per-stage
snapshots, and the numpydantic-typed ``GroundTruth`` that tests, metrics, and the
training notebooks consume.

Two things that were deliberately deferred from earlier steps land here:

* **FOV cropping.** Under motion the tissue steps render on a canvas larger than
  the sensor (see :mod:`minian.simulation.steps.motion`); cells carry canvas-sized
  footprints and canvas-frame positions. ``finalize`` crops them to the sensor
  FOV at the reference (zero-shift) frame — the template motion correction aligns
  back to — and drops cells whose reference footprint falls entirely in the
  margin (real tissue, but background that only flickers in transiently, not a
  recoverable unit).
* **Detectability.** ``detectable`` is not an optics-only property: a cell's peak
  signal (``optical_brightness``) is further dimmed by the illumination/vignette
  field at its position and then judged against a sensor-derived noise floor.
  ``finalize`` is the first point all three exist, so it is where the flag is set.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field

from minian.simulation.scene import MOVIE_DIMS, Cell, Scene
from minian.simulation.spec import Acquisition, Spec

# Minimum realized peak SNR (signal electrons over the sensor noise floor) for a
# cell to count as detectable. A provisional value: the Step 10 threshold
# calibration revisits it against observed metric distributions. Kept here, named,
# rather than buried as a literal so that calibration is a one-line change.
DETECT_SNR_THRESHOLD = 3.0

# On-disk layout for save()/load() (zarr group + sibling spec.json). The format
# version is stamped in the group attrs so a future layout change can be detected
# rather than silently misread.
_FORMAT_VERSION = 1
# GroundTruth array fields, split by whether they are always present or optional
# (None when their producing step is absent). Order is the construction order.
_GT_REQUIRED = (
    "A_planted", "A_observed", "C", "S",
    "centers_um", "amplitude_per_cell", "in_focus", "detectable",
)
_GT_OPTIONAL = (
    "shifts", "vignette", "leakage", "bleaching",
    "neuropil_temporal", "neuropil_spatial",
)


class GroundTruth(BaseModel):
    """The per-recording truth: structural targets + per-cell and per-effect fields.

    Numpydantic annotations declare the dim names, dtype, and rank of every array
    (validated on construction). The **planted vs observed footprint split is
    load-bearing**: ``A_observed`` is what CNMF can actually recover (tests match
    against it), while ``A_planted`` is the ideal, optics-free target that
    quantifies the irreducible limit. Per-effect fields are ``None`` when their
    step is absent from the recording.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # structural truth ------------------------------------------------------
    A_planted: NDArray[Shape["* unit, * height, * width"], float]
    A_observed: NDArray[Shape["* unit, * height, * width"], float]
    C: NDArray[Shape["* unit, * frame"], float]
    S: NDArray[Shape["* unit, * frame"], float]

    # per-cell physical attributes -----------------------------------------
    centers_um: NDArray[Shape["* unit, 3"], float]
    # Per-cell brightness/expression gain (the clean input). NaN for a cell whose
    # ``cell_activity`` step has not run. SNR is deliberately absent: noise is a
    # downstream physical effect, so any SNR is computed later, not stored here.
    amplitude_per_cell: NDArray[Shape["* unit"], float]
    in_focus: NDArray[Shape["* unit"], bool]
    detectable: NDArray[Shape["* unit"], bool]

    # per-effect ground truth (None when that step is absent) ---------------
    shifts: NDArray[Shape["* frame, 2"], float] | None = None
    vignette: NDArray[Shape["* height, * width"], float] | None = None
    leakage: NDArray[Shape["* height, * width"], float] | None = None
    bleaching: NDArray[Shape["* frame"], float] | None = None
    neuropil_temporal: NDArray[Shape["* component, * frame"], float] | None = None
    neuropil_spatial: NDArray[Shape["* component, * height, * width"], float] | None = None

    @property
    def n_units(self) -> int:
        """Number of ground-truth cells (units) in the recording."""
        return int(self.A_planted.shape[0])

    @property
    def depth_um(self) -> np.ndarray:
        """Per-cell depth ``z`` (µm) — the first column of ``centers_um``.

        Exposed as a derived view rather than stored, to avoid drift. Lateral
        pixel coordinates are likewise ``centers_um[:, 1:] / pixel_size_um`` using
        the owning ``Recording.spec.acquisition``.
        """
        return self.centers_um[:, 0]

    def detectable_subset(self) -> GroundTruth:
        """Subset to detectable cells — the fair denominator for recall metrics.

        Slices the per-unit arrays by the ``detectable`` mask; the per-effect
        fields (shifts, vignette, …) are not per-unit and are carried unchanged.
        """
        m = self.detectable
        return GroundTruth(
            A_planted=self.A_planted[m],
            A_observed=self.A_observed[m],
            C=self.C[m],
            S=self.S[m],
            centers_um=self.centers_um[m],
            amplitude_per_cell=self.amplitude_per_cell[m],
            in_focus=self.in_focus[m],
            detectable=self.detectable[m],
            shifts=self.shifts,
            vignette=self.vignette,
            leakage=self.leakage,
            bleaching=self.bleaching,
            neuropil_temporal=self.neuropil_temporal,
            neuropil_spatial=self.neuropil_spatial,
        )


class Recording(BaseModel):
    """A complete simulated recording: the spec, the observed movie, and the truth.

    ``observed`` holds the integer-valued sensor counts in a float container (per
    ``Output.store_dtype``). ``snapshots`` is populated only when
    ``Output.save_intermediates`` is set, keyed by each step's stage ``name`` (see
    ``simulation-spec.md`` §7); ``stage()`` reads them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    spec: Spec
    observed: NDArray[Shape["* frame, * height, * width"], float]
    ground_truth: GroundTruth
    snapshots: dict[str, xr.DataArray] = Field(default_factory=dict)

    def stage(self, name: str) -> xr.DataArray:
        """Return the snapshot taken after the named stage (see §7 stage names)."""
        if name not in self.snapshots:
            raise KeyError(
                f"Stage '{name}' unavailable. Set Output.save_intermediates=True, "
                f"or pick from {sorted(self.snapshots)}."
            )
        return self.snapshots[name]

    def save(self, path: str | Path) -> None:
        """Persist this recording to a self-contained zarr directory at ``path``.

        Layout (one portable directory; ``path`` is conventionally
        ``{spec.cache_key()}.zarr`` but any path works)::

            {path}/
                spec.json            human-readable, diffable spec (see §13)
                (group attrs)        format_version, spec_cache_key, gt_present,
                                     snapshot_names
                observed             (frame, height, width) in store_dtype
                ground_truth/        the GroundTruth arrays (optional ones only
                                     when not None; listed in the gt_present attr)
                snapshots/           per-stage movie values, only when non-empty

        Snapshot coordinates are not stored — they are the trivial ``arange`` grid
        over ``MOVIE_DIMS`` and are rebuilt on :meth:`load`. The write is atomic: it
        builds a sibling ``{path}.tmp`` and renames it into place, so a crash never
        leaves a half-written directory that :meth:`load` would trust.
        """
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)

        root = zarr.open_group(str(tmp), mode="w")
        root.create_dataset("observed", data=np.asarray(self.observed))

        gt_group = root.create_group("ground_truth")
        for name in _GT_REQUIRED:
            gt_group.create_dataset(name, data=np.asarray(getattr(self.ground_truth, name)))
        present = []
        for name in _GT_OPTIONAL:
            value = getattr(self.ground_truth, name)
            if value is not None:
                gt_group.create_dataset(name, data=np.asarray(value))
                present.append(name)

        snapshot_names = sorted(self.snapshots)
        if snapshot_names:
            snap_group = root.create_group("snapshots")
            for name in snapshot_names:
                snap_group.create_dataset(name, data=np.asarray(self.snapshots[name].values))

        root.attrs["format_version"] = _FORMAT_VERSION
        root.attrs["spec_cache_key"] = self.spec.cache_key()
        root.attrs["gt_present"] = present
        root.attrs["snapshot_names"] = snapshot_names
        (tmp / "spec.json").write_text(self.spec.model_dump_json(indent=2))

        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Load a recording written by :meth:`save`, verifying its spec hash.

        Reads back ``spec.json``, re-validates it, and checks that its
        :attr:`Spec.cache_key` matches the one stamped at save time — guarding
        against a stale cache or a hand-edited ``spec.json``. Snapshots are
        rebuilt as ``DataArray``s over ``MOVIE_DIMS`` with ``arange`` coordinates.
        """
        path = Path(path)
        root = zarr.open_group(str(path), mode="r")

        spec = Spec.model_validate_json((path / "spec.json").read_text())
        stored_key = root.attrs.get("spec_cache_key")
        if stored_key != spec.cache_key():
            raise ValueError(
                f"Spec hash mismatch loading {path}: stored {stored_key!r} != "
                f"recomputed {spec.cache_key()!r}. The cache is stale or spec.json "
                f"was edited; delete it and re-simulate."
            )

        gt_group = root["ground_truth"]
        fields = {name: np.asarray(gt_group[name]) for name in _GT_REQUIRED}
        for name in root.attrs.get("gt_present", []):
            fields[name] = np.asarray(gt_group[name])
        ground_truth = GroundTruth(**fields)

        snapshots = {
            name: _movie_dataarray(np.asarray(root["snapshots"][name]))
            for name in root.attrs.get("snapshot_names", [])
        }
        return cls(
            spec=spec,
            observed=np.asarray(root["observed"]),
            ground_truth=ground_truth,
            snapshots=snapshots,
        )


# ---------------------------------------------------------------------------
# finalize: Scene -> Recording
# ---------------------------------------------------------------------------


def finalize(scene: Scene, spec: Spec) -> Recording:
    """Distill an exhausted ``Scene`` into a frozen, typed ``Recording``.

    Crops each cell's canvas-sized footprint and canvas-frame position to the
    sensor FOV (reference frame), drops cells left entirely in the motion margin,
    assembles the per-cell structural truth, sets ``detectable`` from the realized
    optical × illumination peak versus the sensor noise floor, reads the
    per-effect fields off ``scene.truth``, and downcasts the working movie to
    ``Output.store_dtype`` for ``observed``.
    """
    acq = scene.acq
    fov_h = acq.image_sensor.n_px_height
    fov_w = acq.image_sensor.n_px_width
    n_frames = acq.n_frames

    sensor_spec = next((s for s in spec.steps if s.kind == "sensor"), None)
    vignette = scene.truth.vignette  # FOV-sized field, or None

    planted, observed, traces, spikes = [], [], [], []
    centers, amplitudes, in_focus, detectable = [], [], [], []
    for cell in scene.cells:
        if cell.footprint_planted is None:
            continue
        margin_h = (cell.footprint_planted.shape[0] - fov_h) // 2
        margin_w = (cell.footprint_planted.shape[1] - fov_w) // 2
        p_crop = _crop(cell.footprint_planted, margin_h, margin_w, fov_h, fov_w)
        if not p_crop.any():
            continue  # entirely in the margin -> background, not a recoverable unit
        raw_obs = cell.footprint_observed
        o_crop = _crop(
            raw_obs if raw_obs is not None else cell.footprint_planted,
            margin_h, margin_w, fov_h, fov_w,
        )
        z, y_um, x_um = cell.center_um
        y_fov_um = y_um - margin_h * acq.pixel_size_um
        x_fov_um = x_um - margin_w * acq.pixel_size_um

        trace = cell.trace if cell.trace is not None else np.zeros(n_frames)
        spike = cell.spikes if cell.spikes is not None else np.zeros(n_frames)
        ifocus = cell.in_focus if cell.in_focus is not None else True

        planted.append(p_crop)
        observed.append(o_crop)
        traces.append(trace)
        spikes.append(spike)
        centers.append((z, y_fov_um, x_fov_um))
        amplitudes.append(cell.amplitude if cell.amplitude is not None else float("nan"))
        in_focus.append(ifocus)
        detectable.append(
            _is_detectable(cell, ifocus, y_fov_um, x_fov_um, vignette, sensor_spec, acq)
        )

    gt = GroundTruth(
        A_planted=_stack(planted, (0, fov_h, fov_w)),
        A_observed=_stack(observed, (0, fov_h, fov_w)),
        C=_stack(traces, (0, n_frames)),
        S=_stack(spikes, (0, n_frames)),
        centers_um=np.array(centers, dtype=float).reshape(-1, 3),
        amplitude_per_cell=np.array(amplitudes, dtype=float),
        in_focus=np.array(in_focus, dtype=bool),
        detectable=np.array(detectable, dtype=bool),
        shifts=scene.truth.shifts,
        vignette=vignette,
        leakage=scene.truth.leakage,
        bleaching=scene.truth.bleaching,
        neuropil_temporal=scene.truth.neuropil_temporal,
        neuropil_spatial=_crop_components(scene.truth.neuropil_spatial, fov_h, fov_w),
    )
    # observed is always the sensor FOV: brain_motion already crops the movie,
    # but a partial build (until= before motion) can leave it canvas-sized, so
    # crop the centered FOV here too for consistency with the cropped footprints.
    movie = scene.movie.values
    crop_h = (movie.shape[1] - fov_h) // 2
    crop_w = (movie.shape[2] - fov_w) // 2
    if crop_h > 0 or crop_w > 0:
        movie = movie[:, crop_h : crop_h + fov_h, crop_w : crop_w + fov_w]
    observed_movie = movie.astype(spec.output.store_dtype)
    return Recording(
        spec=spec, observed=observed_movie, ground_truth=gt, snapshots=scene.snapshots
    )


def _movie_dataarray(values: np.ndarray) -> xr.DataArray:
    """Rebuild a movie ``DataArray`` from stored values (the saved snapshot form).

    Snapshots are persisted as bare arrays; their coordinates are the trivial
    ``arange`` grid over ``MOVIE_DIMS``, reconstructed here so :meth:`Recording.load`
    returns the same ``(frame, height, width)`` labelling the pipeline produced.
    """
    return xr.DataArray(
        values,
        dims=list(MOVIE_DIMS),
        coords={dim: np.arange(size) for dim, size in zip(MOVIE_DIMS, values.shape)},
        name="movie",
    )


def _crop(field: np.ndarray, top: int, left: int, h: int, w: int) -> np.ndarray:
    """Crop the centered ``h×w`` sensor FOV out of a (possibly margined) canvas."""
    return field[top : top + h, left : left + w]


def _crop_components(stack: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    """Crop each ``(component, H, W)`` field to the reference FOV (or pass None)."""
    if stack is None:
        return None
    top = (stack.shape[1] - h) // 2
    left = (stack.shape[2] - w) // 2
    return stack[:, top : top + h, left : left + w]


def _stack(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    """Stack a per-unit list, or an empty array of ``empty_shape`` when there are none."""
    return np.stack(arrays) if arrays else np.zeros(empty_shape)


def _illumination_at(
    vignette: np.ndarray | None, y_fov_um: float, x_fov_um: float, pixel_size_um: float
) -> float:
    """Illumination factor at a cell's FOV position — the vignette field, or 1.0."""
    if vignette is None:
        return 1.0
    h, w = vignette.shape
    iy = int(np.clip(round(y_fov_um / pixel_size_um), 0, h - 1))
    ix = int(np.clip(round(x_fov_um / pixel_size_um), 0, w - 1))
    return float(vignette[iy, ix])


def _is_detectable(
    cell: Cell,
    in_focus: bool,
    y_fov_um: float,
    x_fov_um: float,
    vignette: np.ndarray | None,
    sensor_spec,
    acq: Acquisition,
) -> bool:
    """Whether a cell's realized peak clears the sensor noise floor (and is in focus).

    The cell's peak ΔF is dimmed by its optical brightness (depth defocus +
    scatter) and the illumination/vignette field at its position, scaled to
    detected electrons by the exposure and QE, then compared to the shot + read
    noise floor riding on its steady baseline::

        signal_e   = peak_ΔF · optical_brightness · illumination · photons_per_unit · QE
        baseline_e = baseline · optical_brightness · illumination · photons_per_unit · QE
        SNR        = signal_e / sqrt(baseline_e + read_noise_e²)

    ``detectable`` requires ``in_focus`` and ``SNR ≥ DETECT_SNR_THRESHOLD``. With
    no activity (no trace) a cell emits no transient and is not detectable; with
    no ``sensor`` step there is no noise floor to test against, so detectability
    falls back to the geometric ``in_focus`` flag.
    """
    if cell.trace is None:
        return False
    if sensor_spec is None:
        return in_focus
    if not in_focus:
        return False
    brightness = cell.optical_brightness if cell.optical_brightness is not None else 1.0
    illum = _illumination_at(vignette, y_fov_um, x_fov_um, acq.pixel_size_um)
    qe = acq.image_sensor.quantum_efficiency
    read_e = acq.image_sensor.read_noise_e
    gain = brightness * illum * sensor_spec.photons_per_unit * qe

    peak_dF = float(cell.trace.max() - cell.trace.min())
    baseline = max(float(cell.trace.min()), 0.0)
    signal_e = peak_dF * gain
    noise_e = math.sqrt(baseline * gain + read_e * read_e)
    if noise_e <= 0:
        return signal_e > 0
    return signal_e / noise_e >= DETECT_SNR_THRESHOLD
