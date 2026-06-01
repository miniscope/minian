"""Structural metrics — the shared oracle comparing estimates to ground truth.

Standalone and dependency-light: the unit tests, the parameter-matrix tests, and
both training notebooks all score CNMF output against a :class:`GroundTruth`
through this one module. It knows nothing about tests or thresholds — callers
supply those (and the fair recall denominator, ``GroundTruth.detectable_subset``).

Every function takes unit-first arrays whose dim order matches *both* sides of the
comparison: CNMF emits ``A`` as ``(unit_id, height, width)`` and ``C``/``S`` as
``(unit_id, frame)``, exactly the layout of ``GroundTruth.A_observed``/``C``/``S``.
Inputs pass through :func:`numpy.asarray`, so ``xr.DataArray`` (what minian's CNMF
returns) and plain ``ndarray`` are both accepted.

The pipeline is: :func:`hungarian_match` pairs estimated footprints to true ones
by spatial overlap, then :func:`trace_pearson` / :func:`spike_precision_recall`
score the temporal recovery of those pairs. :func:`shift_rmse` and
:func:`field_pearson` score the per-effect ground truth (motion, vignette/leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Default fraction of each footprint's energy retained when binarizing it for IoU
# (CaImAn-style): the mask is the *smallest* set of highest-value pixels whose
# summed intensity reaches this fraction of the footprint's total. 0.9 keeps the
# bright core and discards the low-intensity skirt that blur/scatter spread out.
DEFAULT_ENERGY_FRAC = 0.9


@dataclass(frozen=True)
class Match:
    """The result of pairing estimated footprints to true ones by spatial overlap.

    ``pairing`` is the optimal one-to-one assignment (maximizing total IoU) with
    pure non-overlapping pairs dropped, so it is safe to feed straight into the
    temporal metrics. The threshold-dependent quality summaries (:meth:`recall`,
    :meth:`precision`) count only pairs whose IoU clears ``iou_threshold``.

    Empty denominators (no estimated or no true cells, no matched pairs) report
    ``0.0`` rather than ``nan`` — convenient for ``assert metric >= bound`` tests.
    """

    iou_matrix: np.ndarray  # (n_est, n_true) pairwise Jaccard of binarized footprints
    pairing: tuple[tuple[int, int], ...]  # optimal (est_idx, true_idx) pairs, IoU > 0

    @property
    def n_est(self) -> int:
        return int(self.iou_matrix.shape[0])

    @property
    def n_true(self) -> int:
        return int(self.iou_matrix.shape[1])

    def matched_pairs(self, iou_threshold: float = 0.5) -> list[tuple[int, int]]:
        """The assigned pairs whose IoU is at least ``iou_threshold`` (true positives)."""
        return [(i, j) for i, j in self.pairing if self.iou_matrix[i, j] >= iou_threshold]

    def recall(self, iou_threshold: float = 0.5) -> float:
        """True positives over the number of true cells (``0.0`` if there are none)."""
        if self.n_true == 0:
            return 0.0
        return len(self.matched_pairs(iou_threshold)) / self.n_true

    def precision(self, iou_threshold: float = 0.5) -> float:
        """True positives over the number of estimated cells (``0.0`` if there are none)."""
        if self.n_est == 0:
            return 0.0
        return len(self.matched_pairs(iou_threshold)) / self.n_est

    @property
    def mean_iou(self) -> float:
        """Mean IoU over the matched (positive-overlap) pairs (``0.0`` if none)."""
        if not self.pairing:
            return 0.0
        return float(np.mean([self.iou_matrix[i, j] for i, j in self.pairing]))


class SpikeScore(NamedTuple):
    """Pooled spike-train detection score across all matched units."""

    precision: float
    recall: float


def hungarian_match(
    A_est, A_true, *, metric: str = "iou", energy_frac: float = DEFAULT_ENERGY_FRAC
) -> Match:
    """Optimally pair estimated spatial footprints to true ones by overlap.

    Each footprint is binarized to the smallest pixel set holding ``energy_frac``
    of its energy (see :data:`DEFAULT_ENERGY_FRAC`), the pairwise IoU (Jaccard)
    matrix is formed, and :func:`scipy.optimize.linear_sum_assignment` finds the
    assignment maximizing total IoU. Pairs with zero overlap are dropped from the
    returned :attr:`Match.pairing`.

    Parameters
    ----------
    A_est, A_true
        Footprint stacks ``(n, height, width)``, non-negative. Negative values
        (if any) are clipped to zero before binarizing.
    metric
        Only ``"iou"`` is supported in v1; other values raise ``ValueError``.
    energy_frac
        Fraction of each footprint's energy its binary mask retains, in ``(0, 1]``.
    """
    if metric != "iou":
        raise ValueError(f"Unsupported metric {metric!r}; only 'iou' is available in v1.")
    masks_est = _energy_masks(np.asarray(A_est, dtype=float), energy_frac)
    masks_true = _energy_masks(np.asarray(A_true, dtype=float), energy_frac)
    iou = _iou_matrix(masks_est, masks_true)

    rows, cols = linear_sum_assignment(iou, maximize=True)
    pairing = tuple((int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] > 0)
    return Match(iou_matrix=iou, pairing=pairing)


def trace_pearson(C_est, C_true, pairing) -> np.ndarray:
    """Per-matched-pair Pearson correlation between estimated and true traces.

    Returns one correlation per ``(est_idx, true_idx)`` in ``pairing`` (a constant
    trace has undefined correlation and yields ``nan``). ``C_est``/``C_true`` are
    ``(unit, frame)``.
    """
    C_est = np.asarray(C_est, dtype=float)
    C_true = np.asarray(C_true, dtype=float)
    out = []
    for i, j in pairing:
        a, b = C_est[i], C_true[j]
        if a.std() == 0 or b.std() == 0:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out, dtype=float)


def spike_precision_recall(
    S_est, S_true, pairing, *, tol_frames: int = 2, spike_thresh: float = 0.0
) -> SpikeScore:
    """Pooled spike-detection precision/recall over the matched units.

    A frame is a spike where ``S > spike_thresh``. Within each matched pair, true
    spikes are greedily matched to the nearest unused estimated spike within
    ``±tol_frames`` (a true positive); unmatched true spikes are false negatives
    and unmatched estimated spikes are false positives. Counts are pooled across
    all pairs, then reduced to ``precision = TP/(TP+FP)`` and ``recall = TP/(TP+FN)``
    (``0.0`` when a denominator is empty). ``S_est``/``S_true`` are ``(unit, frame)``.
    """
    S_est = np.asarray(S_est, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    tp = fp = fn = 0
    for i, j in pairing:
        est_t = np.flatnonzero(S_est[i] > spike_thresh)
        true_t = np.flatnonzero(S_true[j] > spike_thresh)
        used: set[int] = set()
        for t in true_t:
            candidates = [e for e in est_t if e not in used and abs(int(e) - int(t)) <= tol_frames]
            if candidates:
                used.add(min(candidates, key=lambda e: abs(int(e) - int(t))))
                tp += 1
            else:
                fn += 1
        fp += len(est_t) - len(used)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return SpikeScore(precision=precision, recall=recall)


def shift_rmse(shifts_est, shifts_true) -> float:
    """Root-mean-square error (pixels) between two ``(frame, 2)`` shift trajectories.

    Pure RMSE over all frames and both axes — the caller must put both arrays in
    the **same sign convention**. A motion-*correction* estimate is the negation
    of the applied ``GroundTruth.shifts``, so negate one before comparing.
    """
    e = np.asarray(shifts_est, dtype=float)
    t = np.asarray(shifts_true, dtype=float)
    return float(np.sqrt(np.mean((e - t) ** 2)))


def field_pearson(est, true) -> float:
    """Pearson correlation between two 2-D fields (vignette, leakage), flattened.

    Scale- and offset-invariant, so it scores the *shape* of the recovered field
    rather than its absolute level. Returns ``nan`` if either field is constant.
    """
    a = np.asarray(est, dtype=float).ravel()
    b = np.asarray(true, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _energy_masks(A: np.ndarray, energy_frac: float) -> np.ndarray:
    """Binarize each footprint to the smallest pixel set holding ``energy_frac`` energy.

    For each footprint, pixels are ranked by intensity (high to low) and the top
    ones are kept until their cumulative intensity reaches ``energy_frac`` of the
    footprint's total. An all-zero footprint yields an all-False mask.
    """
    if not 0.0 < energy_frac <= 1.0:
        raise ValueError(f"energy_frac must be in (0, 1], got {energy_frac}.")
    flat = np.clip(A, 0.0, None).reshape(A.shape[0], -1)
    masks = np.zeros(flat.shape, dtype=bool)
    for i, f in enumerate(flat):
        total = f.sum()
        if total <= 0:
            continue
        order = np.argsort(f)[::-1]  # descending intensity
        csum = np.cumsum(f[order])
        k = int(np.searchsorted(csum, energy_frac * total, side="left")) + 1
        masks[i, order[: min(k, f.size)]] = True
    return masks.reshape(A.shape)


def _iou_matrix(masks_est: np.ndarray, masks_true: np.ndarray) -> np.ndarray:
    """Pairwise IoU (Jaccard) between two stacks of boolean masks → ``(n_est, n_true)``.

    Intersections come from a single mask-vs-mask matmul (footprints flattened to
    rows); unions are ``area_est + area_true − intersection``.
    """
    e = masks_est.reshape(masks_est.shape[0], -1).astype(np.float32)
    t = masks_true.reshape(masks_true.shape[0], -1).astype(np.float32)
    intersection = e @ t.T  # (n_est, n_true)
    area_e = e.sum(axis=1)[:, None]
    area_t = t.sum(axis=1)[None, :]
    union = area_e + area_t - intersection
    return np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)
