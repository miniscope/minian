#!/usr/bin/env python3
"""Run monolithic-vs-separated parity judgement on real MiniAn demo data.

Artifacts:
- output/separation_judgement/{original,separated}/
- output/separation_judgement/metrics.csv
- output/separation_judgement/metrics.json
- output/separation_judgement/pass_fail.csv
- output/separation_judgement/timings.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

# Add module paths
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "motion_correction"))
sys.path.insert(0, str(ROOT / "source_detection"))
sys.path.insert(0, str(ROOT / "component_filtering"))

from component_filtering import FilterResult, filter_components as sep_filter
from motion_correction import correct_motion as sep_correct_motion
from preprocessing import preprocess_video as sep_preprocess
from source_detection import detect_sources as sep_detect

# Monolithic imports
sys.path.insert(0, str(REPO_ROOT))
from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import (
    initA,
    initC,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)

# Default demo data paths
DEMO_DPATH = Path("/mnt/nas02/Dataset/minian/demo_movies")
LOCAL_DPATH = REPO_ROOT / "demo_movies"

# Default parameters matching pipeline.ipynb
BASE_CONFIG = {
    "param_load_videos": {
        "pattern": r"msCam[0-9]+\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "param_denoise": {"method": "median", "ksize": 7},
    "param_background_removal": {"method": "tophat", "wnd": 15},
    "param_estimate_motion": {"dim": "frame"},
    "subset": dict(frame=slice(0, None)),
    "subset_mc": None,
    "param_seeds_init": {
        "wnd_size": 1000, "method": "rolling", "stp_size": 500,
        "max_wnd": 15, "diff_thres": 3,
    },
    "param_pnr_refine": {"noise_freq": 0.06, "thres": 1},
    "param_ks_refine": {"sig": 0.05},
    "param_seeds_merge": {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06},
    "param_initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
    "param_init_merge": {"thres_corr": 0.8},
    "param_get_noise": {"noise_range": (0.06, 0.5)},
    "param_first_spatial": {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)},
    "param_first_temporal": {
        "noise_freq": 0.06, "sparse_penal": 1, "p": 1, "add_lag": 20, "jac_thres": 0.2,
    },
    "param_first_merge": {"thres_corr": 0.8},
    "param_second_spatial": {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)},
    "param_second_temporal": {
        "noise_freq": 0.06, "sparse_penal": 1, "p": 1, "add_lag": 20, "jac_thres": 0.4,
    },
}


@dataclass
class StageOutputs:
    """Collected outputs from all pipeline stages."""
    motion: np.ndarray
    max_proj: np.ndarray
    A: np.ndarray
    C: np.ndarray
    S: np.ndarray
    b: np.ndarray
    f: np.ndarray
    b0: np.ndarray
    c0: np.ndarray
    n_units: int
    labels: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, np.ndarray]] = None


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compare_arrays(
    a: np.ndarray, b: np.ndarray,
) -> Tuple[bool, Optional[float], Optional[float], str]:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        return False, None, None, f"shape_mismatch:{arr_a.shape}!={arr_b.shape}"

    exact = bool(np.array_equal(arr_a, arr_b))
    if arr_a.size == 0:
        return exact, 0.0, None, "empty"

    a64 = arr_a.astype(np.float64, copy=False)
    b64 = arr_b.astype(np.float64, copy=False)
    max_abs = safe_float(np.max(np.abs(a64 - b64)))

    if arr_a.size < 2:
        corr = None
    else:
        std_a, std_b = float(np.std(a64)), float(np.std(b64))
        if std_a == 0.0 and std_b == 0.0:
            corr = 1.0 if exact else None
        elif std_a == 0.0 or std_b == 0.0:
            corr = None
        else:
            corr = safe_float(np.corrcoef(a64.ravel(), b64.ravel())[0, 1])

    return exact, max_abs, corr, "ok"


def compare_scalars(a: Any, b: Any) -> Tuple[bool, str]:
    exact = a == b
    return bool(exact), "ok" if exact else f"value_mismatch:{a}!={b}"


def image_ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 2 or arr_a.size == 0:
        return None
    data_min = float(min(np.min(arr_a), np.min(arr_b)))
    data_max = float(max(np.max(arr_a), np.max(arr_b)))
    data_range = data_max - data_min
    if data_range == 0.0:
        return 1.0 if np.array_equal(arr_a, arr_b) else 0.0
    return safe_float(structural_similarity(arr_a, arr_b, data_range=data_range))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_original(dpath: str, intpath: str, minian_path: str) -> StageOutputs:
    """Run monolithic pipeline programmatically (same steps as pipeline.ipynb)."""
    cfg = deepcopy(BASE_CONFIG)
    param_save_minian = {
        "dpath": minian_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    os.environ["MINIAN_INTERMEDIATE"] = intpath
    dpath = os.path.abspath(dpath)

    # Preprocessing
    varr = load_videos(dpath, **cfg["param_load_videos"])
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath, overwrite=True,
    )
    varr_ref = varr.sel(cfg["subset"])
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min
    varr_ref = denoise(varr_ref, **cfg["param_denoise"])
    varr_ref = remove_background(varr_ref, **cfg["param_background_removal"])
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    # Motion correction
    motion = estimate_motion(varr_ref, **cfg["param_estimate_motion"])
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian,
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian,
    ).compute()

    # Source detection (full CNMF)
    seeds = seeds_init(Y_fm_chk, **cfg["param_seeds_init"])
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **cfg["param_pnr_refine"])
    seeds = ks_refine(Y_hw_chk, seeds, **cfg["param_ks_refine"])
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **cfg["param_seeds_merge"])

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **cfg["param_initialize"])
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    A, C = unit_merge(A_init, C_init, **cfg["param_init_merge"])
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )

    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    sn_spatial = get_noise_fft(Y_hw_chk, **cfg["param_get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    # First spatial
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **cfg["param_first_spatial"])
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # First temporal
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **cfg["param_first_temporal"])
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # First merge
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **cfg["param_first_merge"])
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # Second spatial
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **cfg["param_second_spatial"])
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # Second temporal
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **cfg["param_second_temporal"])
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # Final saves
    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)

    return StageOutputs(
        motion=np.asarray(motion.values),
        max_proj=np.asarray(max_proj.values),
        A=np.asarray(A.compute().values),
        C=np.asarray(C.compute().values),
        S=np.asarray(S.compute().values),
        b=np.asarray(b.compute().values),
        f=np.asarray(f.compute().values),
        b0=np.asarray(b0.compute().values),
        c0=np.asarray(c0.compute().values),
        n_units=int(A.sizes["unit_id"]),
    )


def run_separated(dpath: str, intpath: str, minian_path: str) -> StageOutputs:
    """Run the 4-module separated chain."""
    param_save_minian = {
        "dpath": minian_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    config_pp = deepcopy(BASE_CONFIG)
    config_pp["intpath"] = intpath

    # Module 1
    Y_bg, chk, cfg_pp = sep_preprocess(dpath, config_pp)

    # Module 2
    cfg_mc = deepcopy(cfg_pp)
    cfg_mc["param_save_minian"] = param_save_minian
    Y_hw, Y_fm, motion, max_proj, cfg_mc_out = sep_correct_motion(Y_bg, cfg_mc)

    # Module 3
    cfg_sd = deepcopy(cfg_mc_out)
    A, C, S, b, f, b0, c0, cfg_sd_out = sep_detect(Y_hw, Y_fm, max_proj, cfg_sd)

    # Module 4
    result = sep_filter(A, C, S, b0, c0, cfg_sd_out)

    return StageOutputs(
        motion=np.asarray(motion.compute().values if hasattr(motion, "compute") else motion.values),
        max_proj=np.asarray(max_proj.values if hasattr(max_proj, "values") else max_proj),
        A=np.asarray(result.A.compute().values if hasattr(result.A, "compute") else result.A.values),
        C=np.asarray(result.C.compute().values if hasattr(result.C, "compute") else result.C.values),
        S=np.asarray(result.S.compute().values if hasattr(result.S, "compute") else result.S.values),
        b=np.asarray(b.compute().values if hasattr(b, "compute") else b.values),
        f=np.asarray(f.compute().values if hasattr(f, "compute") else f.values),
        b0=np.asarray(b0.compute().values if hasattr(b0, "compute") else b0.values),
        c0=np.asarray(c0.compute().values if hasattr(c0, "compute") else c0.values),
        n_units=int(result.A.sizes.get("unit_id", 0)),
        labels=result.labels,
        metrics=result.metrics,
    )


def add_array_row(
    rows: List[Dict[str, Any]], dataset: str, module: str, target: str,
    arr_a: np.ndarray, arr_b: np.ndarray, gate: bool,
) -> None:
    exact, max_abs, corr, note = compare_arrays(arr_a, arr_b)
    rows.append({
        "dataset": dataset, "module": module, "target": target,
        "is_gate": gate,
        "pass": bool(exact) if gate else None,
        "exact_equal": bool(exact),
        "max_abs_error": max_abs, "pearson_corr": corr,
        "ssim": None, "note": note,
    })


def add_scalar_row(
    rows: List[Dict[str, Any]], dataset: str, module: str, target: str,
    a: Any, b: Any, gate: bool,
) -> None:
    exact, note = compare_scalars(a, b)
    rows.append({
        "dataset": dataset, "module": module, "target": target,
        "is_gate": gate,
        "pass": bool(exact) if gate else None,
        "exact_equal": bool(exact),
        "max_abs_error": 0.0 if exact else None,
        "pearson_corr": None, "ssim": None, "note": note,
    })


def add_ssim_row(
    rows: List[Dict[str, Any]], dataset: str, module: str, target: str,
    img_a: np.ndarray, img_b: np.ndarray,
) -> None:
    rows.append({
        "dataset": dataset, "module": module, "target": target,
        "is_gate": False, "pass": None, "exact_equal": None,
        "max_abs_error": None, "pearson_corr": None,
        "ssim": image_ssim(img_a, img_b), "note": "ssim",
    })


def compute_metrics(
    dataset_name: str, original: StageOutputs, separated: StageOutputs,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Motion correction
    add_array_row(rows, dataset_name, "motion_correction", "motion",
                  original.motion, separated.motion, gate=True)
    add_array_row(rows, dataset_name, "motion_correction", "max_proj",
                  original.max_proj, separated.max_proj, gate=True)
    add_ssim_row(rows, dataset_name, "motion_correction", "max_proj_ssim",
                 original.max_proj, separated.max_proj)

    # Source detection
    add_array_row(rows, dataset_name, "source_detection", "A",
                  original.A, separated.A, gate=True)
    add_array_row(rows, dataset_name, "source_detection", "C",
                  original.C, separated.C, gate=True)
    add_array_row(rows, dataset_name, "source_detection", "S",
                  original.S, separated.S, gate=True)
    add_scalar_row(rows, dataset_name, "source_detection", "n_units",
                   original.n_units, separated.n_units, gate=True)

    # Component filtering
    if separated.labels is not None:
        add_scalar_row(rows, dataset_name, "component_filtering", "all_accepted",
                       True, bool(np.all(separated.labels == 1)), gate=False)

    # Ground truth checks
    add_scalar_row(rows, dataset_name, "ground_truth", "A_sum",
                   int(np.sum(original.A)), int(np.sum(separated.A)), gate=True)
    add_scalar_row(rows, dataset_name, "ground_truth", "C_sum",
                   int(np.sum(original.C)), int(np.sum(separated.C)), gate=True)
    add_scalar_row(rows, dataset_name, "ground_truth", "S_sum",
                   int(np.sum(original.S)), int(np.sum(separated.S)), gate=True)

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "dataset", "module", "target", "is_gate", "pass",
        "exact_equal", "max_abs_error", "pearson_corr", "ssim", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pass_fail(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["module"])
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (dataset, module), module_rows in sorted(grouped.items()):
        gate_rows = [r for r in module_rows if r["is_gate"]]
        if not gate_rows:
            passed = True
            failed_targets: List[str] = []
        else:
            failed_targets = [r["target"] for r in gate_rows if not r["pass"]]
            passed = len(failed_targets) == 0
        summary.append({
            "dataset": dataset, "module": module,
            "pass": passed, "failed_targets": ";".join(failed_targets),
        })
    return summary


def write_pass_fail_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["dataset", "module", "pass", "failed_targets"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_stage_outputs(base_dir: Path, outputs: StageOutputs) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    np.save(base_dir / "motion.npy", outputs.motion)
    np.save(base_dir / "max_proj.npy", outputs.max_proj)
    np.save(base_dir / "A.npy", outputs.A)
    np.save(base_dir / "C.npy", outputs.C)
    np.save(base_dir / "S.npy", outputs.S)
    np.save(base_dir / "b.npy", outputs.b)
    np.save(base_dir / "f.npy", outputs.f)
    np.save(base_dir / "b0.npy", outputs.b0)
    np.save(base_dir / "c0.npy", outputs.c0)
    if outputs.labels is not None:
        np.save(base_dir / "labels.npy", outputs.labels)
    if outputs.metrics is not None:
        for k, v in outputs.metrics.items():
            np.save(base_dir / f"metric_{k}.npy", v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniAn separation judgement parity workflow."
    )
    parser.add_argument(
        "--dpath",
        default=None,
        help="Path to demo_movies directory. Auto-detected if not set.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "output" / "separation_judgement"),
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=int(os.getenv("MINIAN_NWORKERS", 4)),
        help="Number of Dask workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine data path
    dpath = args.dpath
    if dpath is None:
        if LOCAL_DPATH.exists():
            dpath = str(LOCAL_DPATH)
        elif DEMO_DPATH.exists():
            dpath = str(DEMO_DPATH)
        else:
            raise RuntimeError(
                "No demo data found. Provide --dpath or place data at "
                f"{LOCAL_DPATH} or {DEMO_DPATH}"
            )
    print(f"Using data path: {dpath}", flush=True)

    # Set up threading limits
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Start Dask client
    from dask.distributed import Client, LocalCluster
    from minian.utilities import TaskAnnotation

    cluster = LocalCluster(
        n_workers=args.n_workers,
        memory_limit="2GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}", flush=True)

    dataset_name = "demo_movies"
    all_rows: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}

    try:
        # Run original
        intpath_orig = str(output_root / "original" / "intermediate")
        minian_orig = str(output_root / "original" / "minian")
        os.makedirs(minian_orig, exist_ok=True)

        print("\n=== Running ORIGINAL (monolithic) pipeline ===", flush=True)
        t0 = time.time()
        original = run_original(dpath, intpath_orig, minian_orig)
        timings["original_sec"] = time.time() - t0
        print(f"Original done in {timings['original_sec']:.2f}s", flush=True)
        print(f"  units={original.n_units}, A_sum={int(np.sum(original.A))}, "
              f"C_sum={int(np.sum(original.C))}, S_sum={int(np.sum(original.S))}", flush=True)

        # Run separated
        intpath_sep = str(output_root / "separated" / "intermediate")
        minian_sep = str(output_root / "separated" / "minian")
        os.makedirs(minian_sep, exist_ok=True)

        print("\n=== Running SEPARATED (4-module chain) pipeline ===", flush=True)
        t1 = time.time()
        separated = run_separated(dpath, intpath_sep, minian_sep)
        timings["separated_sec"] = time.time() - t1
        print(f"Separated done in {timings['separated_sec']:.2f}s", flush=True)
        print(f"  units={separated.n_units}, A_sum={int(np.sum(separated.A))}, "
              f"C_sum={int(np.sum(separated.C))}, S_sum={int(np.sum(separated.S))}", flush=True)

        # Save outputs
        save_stage_outputs(output_root / "original" / dataset_name, original)
        save_stage_outputs(output_root / "separated" / dataset_name, separated)

        # Compute metrics
        rows = compute_metrics(dataset_name, original, separated)
        all_rows.extend(rows)

        # Report
        pf_rows = build_pass_fail(rows)
        all_pass = all(r["pass"] for r in pf_rows)
        print(f"\nOverall PASS: {all_pass}", flush=True)
        for r in pf_rows:
            status = "PASS" if r["pass"] else f"FAIL ({r['failed_targets']})"
            print(f"  {r['module']}: {status}", flush=True)

    finally:
        client.close()
        cluster.close()

    # Write artifacts
    metrics_csv = output_root / "metrics.csv"
    metrics_json = output_root / "metrics.json"
    pass_fail_csv = output_root / "pass_fail.csv"
    timings_json = output_root / "timings.json"

    write_csv(metrics_csv, all_rows)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    pass_fail_rows = build_pass_fail(all_rows)
    write_pass_fail_csv(pass_fail_csv, pass_fail_rows)
    with timings_json.open("w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    print(f"\nWrote: {metrics_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {pass_fail_csv}")
    print(f"Wrote: {timings_json}")


if __name__ == "__main__":
    main()
