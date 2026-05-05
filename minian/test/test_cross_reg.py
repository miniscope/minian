"""Integration test for :mod:`minian.pipelines.cross_reg` (not the Jupyter notebook)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from minian.pipelines.cross_reg import (
    DEFAULT_PARAM_DIST,
    parse_cross_reg_argv,
    run_cross_reg,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DATA = _REPO_ROOT / "demo_data"


def test_parse_cross_reg_argv_defaults() -> None:
    args = parse_cross_reg_argv([])
    assert args.data == "."
    assert args.param_dist == DEFAULT_PARAM_DIST


def test_parse_cross_reg_argv_param_dist() -> None:
    args = parse_cross_reg_argv(["--param-dist", "12"])
    assert args.param_dist == 12


def test_parse_cross_reg_argv_data_override() -> None:
    args = parse_cross_reg_argv(["--data", "/tmp/cross_reg_data"])
    assert args.data == "/tmp/cross_reg_data"


@pytest.fixture
def cross_reg_input_dir(tmp_path: Path) -> Path:
    """Copy bundled multi-session ``minian.nc`` trees into an isolated working directory."""
    for sess in ("session1", "session2"):
        src_nc = _DEMO_DATA / sess / "minian.nc"
        if not src_nc.is_file():
            pytest.skip(f"missing demo inputs: {src_nc}")
        dest_dir = tmp_path / sess
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_nc, dest_dir / "minian.nc")
    return tmp_path


def test_cross_reg_pipeline_outputs(cross_reg_input_dir: Path) -> None:
    dpath = str(cross_reg_input_dir)
    run_cross_reg(dpath, r"minian.nc$", ["session"], param_dist=5)

    assert (cross_reg_input_dir / "shiftds.nc").is_file()
    assert (cross_reg_input_dir / "cents.pkl").is_file()
    assert (cross_reg_input_dir / "mappings.pkl").is_file()

    cents = pd.read_pickle(cross_reg_input_dir / "cents.pkl")
    mappings = pd.read_pickle(cross_reg_input_dir / "mappings.pkl")

    assert len(cents) == 508
    # Centroid columns are floats; summed totals are regression anchors, not integers.
    assert cents["height"].sum() == pytest.approx(99096.462, abs=0.05)
    assert cents["width"].sum() == pytest.approx(213628.121, abs=0.05)

    assert len(mappings) == 430
    assert mappings[("group", "group")].value_counts().to_dict() == {
        ("session2",): 181,
        ("session1",): 171,
        ("session1", "session2"): 78,
    }

    with xr.open_dataset(cross_reg_input_dir / "shiftds.nc") as ds:
        assert "shifts" in ds.data_vars
        assert "temps" in ds.data_vars
        assert "temps_shifted" in ds.data_vars
