"""Per-stage recovery demo: CNMF cell extraction (migration Step 10c).

The headline per-stage test — feed a simulated recording to minian's *real* CNMF
chain and check it recovers the known detectable footprints. This is what the
end-to-end golden-number test cannot offer: a direct, diagnosable comparison of
extracted cells against exact ground truth (``GroundTruth.detectable_subset`` —
the cells physics says are recoverable: in focus and above the noise floor).

It is ``slow``-marked and guarded by ``importorskip("pymetis")``: the CNMF chain
needs ``pymetis`` (graph partitioning), which has no Windows wheel today, so this
skips on a stock Windows dev box and activates automatically wherever pymetis is
installed (CI Linux, conda) — including once the in-flight pymetis-replacement
lands. A 60 s, 128 px, ~50-cell recording recovers ~85% of detectable cells at
IoU ~0.7 in ~80 s.

Deliberately generous bounds (well below observed, and robust to the BLAS /
numerical-library drift that the golden CNMF sums are sensitive to) — a
capability demonstration, not the calibrated replacement suite (a later PR).
"""

import os

import pytest

# The CNMF chain imports pymetis at module load; skip the whole module cleanly
# where it is unavailable, before importing anything that pulls it in.
pytest.importorskip("pymetis")

import numpy as np
import xarray as xr

from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import initA, initC, ks_refine, pnr_refine, seeds_init, seeds_merge
from minian.preprocessing import denoise, remove_background
from minian.simulation import (
    Acquisition,
    CellActivity,
    CellOptics,
    ImageSensor,
    Optics,
    PlaceSomata,
    Render,
    Sensor,
    Spec,
    hungarian_match,
    simulate,
)
from minian.utilities import save_minian

pytestmark = pytest.mark.slow

_CHK = 200  # frame chunk size for the dask pipeline


@pytest.fixture(scope="module")
def dask_client(tmp_path_factory):
    """A small local dask cluster + a scratch intermediate dir, as the pipeline expects."""
    import dask
    from dask.distributed import Client, LocalCluster

    os.environ["MINIAN_INTERMEDIATE"] = str(tmp_path_factory.mktemp("minian_intermediate"))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    # spill rather than kill workers that brush the memory cap (matches the notebook)
    dask.config.set({"distributed.worker.memory.terminate": False})
    cluster = LocalCluster(
        n_workers=2, threads_per_worker=2, memory_limit="3GB", dashboard_address=None
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def _recording():
    """A 60 s, 128 px, ~50-cell, shallow/in-focus still recording — no motion to isolate CNMF."""
    acq = Acquisition(
        fps=20.0,
        duration_s=60.0,
        optics=Optics(magnification=8.0, na=0.45, focal_plane_um=5.0, depth_of_field_um=60.0),
        image_sensor=ImageSensor(n_px_height=128, n_px_width=128, pixel_pitch_um=8.0, bit_depth=8),
    )
    spec = Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceSomata(density_per_mm2=3000.0, soma_radius_um=5.0, depth_range_um=(0.0, 15.0)),
            CellActivity(active_rate_hz=4.0, tau_decay_s=0.5),
            CellOptics(),
            Render(),
            # ~300 / NA² (NA 0.45): compensates the NA²-collection dimming so the
            # recorded movie (and thus CNMF recovery) is unchanged. See cell.py.
            Sensor(photons_per_unit=1500.0),
        ],
    )
    return simulate(spec)


def _run_cnmf(observed: np.ndarray) -> np.ndarray:
    """Run minian's real CNMF chain on an observed movie; return the extracted footprints.

    A faithful, minimal transcription of the pipeline notebook's preprocessing →
    initialization → spatial/temporal-update sequence. The ``save_minian`` calls
    between steps are load-bearing: the zarr round-trips concretize dask chunk
    shapes (skipping them surfaces a ``shape is None`` error downstream).
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    n, h, w = observed.shape
    # the real pipeline loads uint8 video; our 8-bit sensor counts cast exactly,
    # and OpenCV's median denoise requires 8-bit input
    varr = xr.DataArray(
        observed.astype(np.uint8),
        dims=["frame", "height", "width"],
        coords={"frame": np.arange(n), "height": np.arange(h), "width": np.arange(w)},
        name="varr",
    )
    varr_ref = (varr - varr.min("frame")).chunk({"frame": _CHK, "height": -1, "width": -1})
    varr_ref = denoise(varr_ref, method="median", ksize=7)
    varr_ref = remove_background(varr_ref, method="tophat", wnd=15)
    # preprocessing is 8-bit (OpenCV); the seeds/CNMF stage is float (numba)
    Y = varr_ref.astype(float)
    Y_fm_chk = Y.chunk({"frame": _CHK, "height": -1, "width": -1})
    Y_hw_chk = Y.chunk({"frame": -1, "height": 64, "width": 64})
    max_proj = Y_fm_chk.max("frame").compute()

    # --- initialization: seeds -> refine -> footprints ---
    seeds = seeds_init(Y_fm_chk, wnd_size=1000, method="rolling", stp_size=500, max_wnd=15, diff_thres=3)
    seeds, _pnr, _gmm = pnr_refine(Y_hw_chk, seeds, noise_freq=0.06, thres=1)
    seeds = ks_refine(Y_hw_chk, seeds, sig=0.05)
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, thres_dist=10, thres_corr=0.8, noise_freq=0.06)
    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], thres_corr=0.8, wnd=10, noise_freq=0.06)
    C_init = initC(Y_fm_chk, A_init)

    A, C = unit_merge(A_init, C_init, thres_corr=0.8)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": _CHK})

    # --- CNMF: one spatial then one temporal update ---
    b, _f = update_background(Y_fm_chk, A, C_chk)
    save_minian(b.rename("b"), intpath, overwrite=True)
    sn_spatial = get_noise_fft(Y_hw_chk, noise_range=(0.06, 0.5))
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, dl_wnd=10, sparse_penal=0.01, size_thres=(25, None)
    )
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    A_new = save_minian(
        A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1}
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    b_new = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f_new = save_minian(f_new.chunk({"frame": _CHK}).rename("f"), intpath, overwrite=True)

    YrA = save_minian(
        compute_trace(Y_fm_chk, A_new, b_new, C_chk_new, f_new).rename("YrA"),
        intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    update_temporal(A_new, C_new, YrA=YrA, noise_freq=0.06, sparse_penal=1, p=1, add_lag=20, jac_thres=0.2)

    return np.asarray(A_new.transpose("unit_id", "height", "width").compute())


def test_cnmf_recovers_detectable_footprints(dask_client):
    rec = _recording()
    detectable = rec.ground_truth.detectable_subset()
    assert detectable.A_observed.shape[0] > 0  # the regime must have recoverable cells

    A_est = _run_cnmf(rec.observed)
    match = hungarian_match(A_est, detectable.A_observed, metric="iou")

    # generous demonstration bounds (observed ~0.86 recall / ~0.72 mean IoU);
    # well clear of numerical-library drift, not a calibrated replacement threshold
    assert match.recall(iou_threshold=0.5) >= 0.7, (
        f"CNMF recovered only {match.recall(0.5):.2f} of detectable cells "
        f"({len(match.pairing)}/{detectable.A_observed.shape[0]})"
    )
    assert match.mean_iou >= 0.5, f"mean IoU {match.mean_iou:.2f} of matched footprints too low"
    assert match.precision(iou_threshold=0.5) >= 0.4, f"precision {match.precision(0.5):.2f} too low"
