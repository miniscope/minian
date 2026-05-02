"""Tests for :mod:`minian.config`."""

import json
import os

import pytest

from minian.config import (
    DEFAULT_WORKER_CPU_RATIO,
    PipelineConfig,
    load_pipeline_config,
    pipeline_config_to_jsonable,
    resolve_n_workers,
)
from minian.constants import (
    MINIAN,
    MINIAN_CONFIG_FILENAME,
    MINIAN_INTERMEDIATE,
    get_minian_intermediate_path,
)


def test_resolve_n_workers_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAN_NWORKERS", "8")
    assert resolve_n_workers() == 8
    monkeypatch.setenv("MINIAN_NWORKERS", "1")
    assert resolve_n_workers() == 1


def test_resolve_n_workers_reserve_forces_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    monkeypatch.delenv("MINIAN_NWORKERS", raising=False)
    assert resolve_n_workers(reserve=10_000) == 1


def test_algorithm_param_dicts_excludes_save_minian() -> None:
    cfg = PipelineConfig()
    d = cfg.algorithm_param_dicts()
    assert "param_save_minian" not in d
    assert "param_load_videos" in d
    assert d["param_load_videos"]["pattern"] == r"msCam[0-9]+\.avi$"
    assert "param_second_temporal" in d


def test_get_minian_intermediate_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_minian_intermediate_path() == os.path.join(
        str(tmp_path), MINIAN_INTERMEDIATE
    )
    got = get_minian_intermediate_path("/tmp")
    assert got == os.path.join("/tmp", MINIAN_INTERMEDIATE)


def test_resolve_n_workers_invalid_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    monkeypatch.setenv("MINIAN_NWORKERS", "not-a-number")
    n = resolve_n_workers(reserve=0)
    assert n >= 1


def test_resolve_n_workers_explicit_ratio_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    from minian.minian_rs import thread_allocation

    monkeypatch.delenv("MINIAN_NWORKERS", raising=False)
    monkeypatch.setenv("MINIAN_WORKER_CPU_RATIO", "0.25")
    want = int(thread_allocation(1, 1.0 / 3.0).cluster_workers)
    assert resolve_n_workers(reserve=1, worker_cpu_ratio=1.0 / 3.0) == want


def test_resolve_n_workers_env_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("minian.minian_rs")
    from minian.minian_rs import thread_allocation

    monkeypatch.delenv("MINIAN_NWORKERS", raising=False)
    monkeypatch.setenv("MINIAN_WORKER_CPU_RATIO", "0.5")
    assert resolve_n_workers(reserve=1) == int(
        thread_allocation(1, 0.5).cluster_workers
    )


def test_with_paths_resolved_absolutizes_intpath_and_save_dpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    save = tmp_path / "demo" / MINIAN
    save.parent.mkdir(parents=True)
    cfg = PipelineConfig(
        intpath="./scratch",
        param_save_minian={
            "dpath": str(save),
            "meta_dict": {"session": -1},
            "overwrite": False,
        },
    )
    r = cfg.with_paths_resolved()
    assert os.path.isabs(r.intpath)
    assert r.intpath == os.path.join(str(tmp_path), "scratch")
    assert r.param_save_minian["dpath"] == str(save.resolve())


def test_pipeline_config_to_jsonable_roundtrip() -> None:
    pytest.importorskip("minian.minian_rs")
    cfg = PipelineConfig()
    d = pipeline_config_to_jsonable(cfg, include_resolved_workers=True)
    text = json.dumps(d)
    assert "param_first_temporal" in text
    assert "resolved_n_workers" in d
    assert d["resolved_n_workers"] >= 1
    assert "resolved_worker_cpu_ratio" in d
    assert abs(d["resolved_worker_cpu_ratio"] - DEFAULT_WORKER_CPU_RATIO) < 1e-9
    loads = json.loads(text)
    assert loads["subset"]["frame"]["__slice__"] == [0, None, None]


def test_load_pipeline_config_defaults_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    got = load_pipeline_config(cwd=str(tmp_path))
    assert isinstance(got, PipelineConfig)
    default = PipelineConfig()
    assert got.intpath == default.intpath


def test_load_pipeline_config_reads_cwd_filename(tmp_path) -> None:
    cfg_in = PipelineConfig(
        n_workers=7,
        param_save_minian={
            "dpath": "./custom/minian",
            "meta_dict": {"session": -1},
            "overwrite": True,
        },
    )
    dumped = pipeline_config_to_jsonable(cfg_in)
    p = tmp_path / MINIAN_CONFIG_FILENAME
    p.write_text(json.dumps(dumped), encoding="utf-8")
    got = load_pipeline_config(cwd=str(tmp_path))
    assert got.n_workers == 7
    assert got.param_save_minian["dpath"] == "./custom/minian"


def test_load_pipeline_config_explicit_path(tmp_path) -> None:
    p = tmp_path / "other.json"
    cfg_in = PipelineConfig(n_workers=3)
    p.write_text(json.dumps(pipeline_config_to_jsonable(cfg_in)), encoding="utf-8")
    assert load_pipeline_config(path=str(p)).n_workers == 3


def test_apply_environment_uses_thread_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", env)
    cfg = PipelineConfig(
        intpath="/tmp/im",
        thread_env={
            "OMP_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "CUSTOM_FLAG": "x",
        },
    )
    cfg.apply_environment()
    assert env["MINIAN_INTERMEDIATE"] == "/tmp/im"
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["CUSTOM_FLAG"] == "x"


def test_apply_environment_blas_threads_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", env)
    cfg = PipelineConfig(
        thread_env={"OMP_NUM_THREADS": "99"},
    )
    cfg.apply_environment(blas_threads=2)
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"


def test_pipeline_config_json_roundtrip_key_fields() -> None:
    from minian.config import _pipeline_config_from_json_dict

    cfg = PipelineConfig(reserve_cores_for_os=2)
    blob = pipeline_config_to_jsonable(cfg)
    cfg2 = _pipeline_config_from_json_dict(blob)
    assert cfg2.reserve_cores_for_os == 2
    assert cfg2.subset == cfg.subset
    assert cfg2.thread_env == cfg.thread_env


def test_rust_allocation_matches_resolve_n_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``thread_allocation``, ``default_cluster_workers``, and ``resolve_n_workers`` agree."""
    rs = pytest.importorskip("minian.minian_rs")
    from minian.minian_rs import default_cluster_workers, thread_allocation

    monkeypatch.delenv("MINIAN_NWORKERS", raising=False)
    monkeypatch.delenv("MINIAN_WORKER_CPU_RATIO", raising=False)
    ta = thread_allocation(1)
    assert ta.cluster_workers == int(default_cluster_workers(1))
    assert ta.logical_cpus == int(rs.logical_parallelism())
    assert abs(float(ta.worker_cpu_ratio) - DEFAULT_WORKER_CPU_RATIO) < 1e-9
    assert resolve_n_workers(reserve=1) == int(ta.cluster_workers)
