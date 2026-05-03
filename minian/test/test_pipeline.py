"""Tests for :mod:`minian.pipelines.cnmf_process` and light Dask checks.

Includes tiny :mod:`dask.array` smoke tests and optional ``demo_movies`` loads
(heavy downsample + small ``.compute()``), without running the full CNMF driver.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pytest

from minian.constants import get_minian_intermediate_path, minian_folder_under
from minian.pipelines import cnmf_process
from minian.pipelines.cnmf_process import (
    PipelinePaths,
    _spatial_chunks_full_frame,
    main,
    parse_pipeline_argv,
)
from minian.utilities import ensure_ffmpeg, load_videos

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_MOVIES = _REPO_ROOT / "demo_movies"


def _demo_movies_or_skip() -> Path:
    if not _DEMO_MOVIES.is_dir():
        pytest.skip(f"missing demo videos directory: {_DEMO_MOVIES}")
    if not any(_DEMO_MOVIES.glob("msCam*.avi")):
        pytest.skip(f"no msCam*.avi under {_DEMO_MOVIES}")
    return _DEMO_MOVIES


def test_parse_pipeline_argv_defaults() -> None:
    args = parse_pipeline_argv([])
    assert args.data == "./demo_movies/"
    assert args.config is None
    assert "worker_cpu_ratio" not in vars(args)


def test_parse_pipeline_argv_overrides() -> None:
    args = parse_pipeline_argv(
        ["-d", "/data/movies", "-c", "/cfg/p.json", "--worker-cpu-ratio", "0.42"]
    )
    assert args.data == "/data/movies"
    assert args.config == "/cfg/p.json"
    assert args.worker_cpu_ratio == pytest.approx(0.42)


def test_main_calls_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        data_dir: str,
        *,
        worker_cpu_ratio: Any = None,
        config_path: Any = None,
    ) -> None:
        captured["data_dir"] = data_dir
        captured["worker_cpu_ratio"] = worker_cpu_ratio
        captured["config_path"] = config_path

    monkeypatch.setattr(cnmf_process, "run_pipeline", fake_run)
    main(
        ["-d", "/tmp/demo_movies", "--worker-cpu-ratio", "0.33", "-c", "/tmp/cfg.json"]
    )
    assert captured["data_dir"] == "/tmp/demo_movies"
    assert captured["worker_cpu_ratio"] == pytest.approx(0.33)
    assert captured["config_path"] == "/tmp/cfg.json"


def test_spatial_chunks_full_frame() -> None:
    assert _spatial_chunks_full_frame() == {
        "unit_id": 1,
        "height": -1,
        "width": -1,
    }


def test_pipeline_paths_match_run_pipeline_layout(tmp_path: Path) -> None:
    """Same ``root`` / ``dpath`` / ``intpath`` / ``param_save_minian`` layout as :func:`run_pipeline` builds."""
    demo = tmp_path / "demo_movies"
    demo.mkdir()
    dpath = str(demo.resolve())
    root = Path(dpath).parent
    intpath = get_minian_intermediate_path(str(root))
    save_kw: dict[str, Any] = {
        "meta_dict": {"session": -1, "animal": -2},
        "overwrite": True,
        "dpath": minian_folder_under(dpath),
    }
    paths = PipelinePaths(
        root=root,
        dpath=dpath,
        intpath=intpath,
        param_save_minian=save_kw,
    )
    assert paths.root == root
    assert paths.dpath == os.path.abspath(dpath)
    assert paths.intpath == intpath
    assert paths.param_save_minian["dpath"] == minian_folder_under(
        os.path.abspath(dpath)
    )


def test_dask_array_chunked_elementwise_and_sum() -> None:
    with dask.config.set(scheduler="threads"):
        x = da.ones((128, 64), chunks=(32, 32))
        y = (x * 2.0 + 1.0).sum()
        assert float(y.compute()) == pytest.approx(128 * 64 * 3.0)


def test_dask_array_rechunk_and_reduce() -> None:
    with dask.config.set(scheduler="threads"):
        x = da.arange(120, chunks=30)
        y = x.rechunk(20).reshape((10, 12)).sum(axis=0)
        out = y.compute()
        assert out.shape == (12,)
        assert int(out.sum()) == sum(range(120))


def test_load_videos_demo_movies_dask_graph_and_slice_sum() -> None:
    """Lazy concat from ``demo_movies``; tiny ``.compute()`` to exercise ffmpeg-backed dask graph."""
    dpath = str(_demo_movies_or_skip())
    with dask.config.set(scheduler="threads"):
        varr = load_videos(
            dpath,
            pattern=r"msCam[0-9]+\.avi$",
            dtype=np.uint8,
            downsample={"frame": 40, "height": 4, "width": 4},
            downsample_strategy="subset",
        )
    assert isinstance(varr.data, da.Array)
    assert varr.ndim == 3
    assert varr.sizes["frame"] > 0
    sub = varr.isel(frame=slice(0, 2), height=slice(0, 16), width=slice(0, 16))
    total = int(sub.sum().compute())
    assert 0 < total <= 2 * 16 * 16 * 255


def test_load_videos_demo_movies_rechunk_first_frame() -> None:
    """Rechunk + single-frame reduction (pattern similar to pipeline chunk tuning)."""
    dpath = str(_demo_movies_or_skip())
    with dask.config.set(scheduler="threads"):
        varr = load_videos(
            dpath,
            pattern=r"msCam[0-9]+\.avi$",
            dtype=np.uint8,
            downsample={"frame": 40, "height": 4, "width": 4},
            downsample_strategy="subset",
        )
        rch = varr.chunk({"frame": -1, "height": 32, "width": 32})
        one = rch.isel(frame=0)
        s = int(one.sum().compute())
    assert s > 0


def test_ensure_ffmpeg_smoke() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe required on PATH for video tests")
    ensure_ffmpeg()
