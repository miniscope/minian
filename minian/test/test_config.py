"""Tests for :mod:`minian.config`."""

import dataclasses
import json
import os
from collections.abc import Iterator

import pytest

from minian.config import (
    PipelineConfig,
    PipelineEnv,
    build_pipeline_effective_record,
    clear_active_pipeline_config,
    dask_chunk_target_mb,
    dask_threads_per_worker,
    dask_worker_memory_limit,
    get_active_pipeline_config,
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


@pytest.fixture(autouse=True)
def reset_active_pipeline_config() -> (
    Iterator[None]
):  # pyright: ignore[reportUnusedFunction]
    """Autouse: clear active :class:`~minian.config.PipelineConfig` after each test."""
    yield
    clear_active_pipeline_config()


def test_get_active_pipeline_config_raises_when_unset() -> None:
    clear_active_pipeline_config()
    with pytest.raises(RuntimeError, match="No active pipeline"):
        get_active_pipeline_config()


def test_resolve_n_workers_ignores_obsolete_minian_n_workers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    monkeypatch.setenv("MINIAN_NWORKERS", "99")
    assert resolve_n_workers(reserve=1) != 99


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
    parent = os.path.join(tmp_path, "session_root")
    os.makedirs(parent, exist_ok=True)
    got = get_minian_intermediate_path(parent)
    assert got == os.path.join(os.path.abspath(parent), MINIAN_INTERMEDIATE)


def test_resolve_n_workers_invalid_explicit_ratio_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    n = resolve_n_workers(reserve=0, worker_cpu_ratio=float("nan"))
    assert n >= 1


def test_resolve_n_workers_explicit_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    from minian.minian_rs import thread_allocation

    monkeypatch.setenv("MINIAN_WORKER_CPU_RATIO", "0.25")
    want = int(thread_allocation(1, 1.0 / 3.0).cluster_workers)
    assert resolve_n_workers(reserve=1, worker_cpu_ratio=1.0 / 3.0) == want


def test_resolve_n_workers_default_ratio_ignores_worker_cpu_ratio_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minian.minian_rs")
    from minian.minian_rs import thread_allocation

    monkeypatch.setenv("MINIAN_WORKER_CPU_RATIO", "0.5")
    assert resolve_n_workers(reserve=1) == int(
        thread_allocation(1, PipelineEnv.DEFAULT_WORKER_CPU_RATIO).cluster_workers
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
    assert (
        abs(d["resolved_worker_cpu_ratio"] - PipelineEnv.DEFAULT_WORKER_CPU_RATIO)
        < 1e-9
    )
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
    assert get_active_pipeline_config().intpath == os.path.abspath("/tmp/im")
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
    assert abs(float(ta.worker_cpu_ratio) - PipelineEnv.DEFAULT_WORKER_CPU_RATIO) < 1e-9
    assert resolve_n_workers(reserve=1) == int(ta.cluster_workers)


def test_dask_worker_memory_limit_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PipelineEnv.MINIAN_WORKER_MEMORY, raising=False)
    assert dask_worker_memory_limit() == PipelineEnv.DEFAULT_DASK_WORKER_MEMORY


def test_dask_worker_memory_limit_from_active_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig(dask_worker_memory=" 4GB ")
    cfg.apply_environment()
    assert dask_worker_memory_limit() == "4GB"


def test_dask_threads_per_worker(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert dask_threads_per_worker() == PipelineEnv.DEFAULT_DASK_THREADS_PER_WORKER
    PipelineConfig(dask_threads_per_worker=4).apply_environment()
    assert dask_threads_per_worker() == 4


def test_dask_chunk_target_mb(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert dask_chunk_target_mb() == PipelineEnv.DEFAULT_DASK_CHUNK_TARGET_MB
    PipelineConfig(dask_chunk_target_mb=512).apply_environment()
    assert dask_chunk_target_mb() == 512
    PipelineConfig(dask_chunk_target_mb=0).apply_environment()
    assert dask_chunk_target_mb() == 1


def test_build_pipeline_effective_digest_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig()
    r1 = build_pipeline_effective_record(
        cfg,
        n_workers=2,
        worker_memory_limit="2GB",
        threads_per_worker=2,
        chunk_target_mb=128,
    )
    r2 = build_pipeline_effective_record(
        cfg,
        n_workers=2,
        worker_memory_limit="2GB",
        threads_per_worker=2,
        chunk_target_mb=128,
    )
    assert r1["defaults_digest"] == r2["defaults_digest"]
    assert len(r1["defaults_digest"]) == 16


def test_build_pipeline_effective_delta_and_cli_ratio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = dataclasses.replace(PipelineConfig(), reserve_cores_for_os=3)
    rec = build_pipeline_effective_record(
        cfg,
        n_workers=1,
        worker_memory_limit="1GB",
        threads_per_worker=1,
        chunk_target_mb=200,
        cli_worker_cpu_ratio=0.41,
    )
    assert rec["delta_from_builtin_defaults"]["reserve_cores_for_os"] == 3
    assert rec["resolved_cluster"]["cli_worker_cpu_ratio"] == 0.41
    pytest.importorskip("minian.minian_rs")
    assert isinstance(rec["resolved_cluster"]["resolved_worker_cpu_ratio"], float)


def test_build_pipeline_effective_top_level_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig()
    rec = build_pipeline_effective_record(
        cfg,
        n_workers=4,
        worker_memory_limit="3GB",
        threads_per_worker=2,
        chunk_target_mb=200,
    )
    assert set(rec) == {
        "timestamp",
        "minian_version",
        "defaults_digest",
        "delta_from_builtin_defaults",
        "resolved_cluster",
    }
    assert isinstance(rec["timestamp"], str) and "T" in rec["timestamp"]
    rc = rec["resolved_cluster"]
    assert rc["n_workers"] == 4
    assert rc["worker_memory_limit"] == "3GB"
    assert "cli_worker_cpu_ratio" not in rc
