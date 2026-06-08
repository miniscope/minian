import os
import shutil
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from ._notebook import require_dataset


def pytest_sessionstart(session):
    """Set env vars for dask resource limits"""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (2**30)
    os.environ["MINIAN_NWORKERS"] = "1"
    os.environ["MINIAN_MEM_LIMIT"] = f"{total_gb * .75:.2f}GB"
    os.environ["MINIAN_INTERACTIVE"] = "False"


# Outputs each notebook writes back INTO its (shared, cached) dataset directory,
# keyed by dataset name. Data-driven so the cleanup fixture stays generic rather
# than hard-coding paths per test. These are OUTPUTS only -- never the registry
# input files (msCam*.avi / session*/minian.nc), which must survive for reuse.
DATASET_OUTPUTS: dict[str, tuple[str, ...]] = {
    "pipeline-demo": ("minian", "minian_mc.mp4", "minian.mp4"),
    "cross-reg-sessions": ("shiftds.nc", "cents.pkl", "mappings.pkl"),
}


def _remove(path: Path) -> None:
    """Delete a file or directory tree if it exists; no-op otherwise."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@pytest.fixture(scope="session")
def dataset() -> Callable[[str], Path]:
    """Factory for resolving demo datasets (download + cache, or skip).

    Returns a ``get_dataset(name) -> Path`` callable so any fixture or test can
    pluck the dataset it needs from one shared, session-scoped resolver. This is
    just dependency injection: ``require_dataset`` skips cleanly when the dataset
    is unavailable, and the underlying ``minian.data`` cache means resolving the
    same name twice is cheap.
    """
    return require_dataset


@pytest.fixture
def clean_dataset_outputs(
    dataset: Callable[[str], Path],
) -> Callable[[str], Path]:
    """Factory yielding ``get_clean(name) -> Path`` for output-writing tests.

    Datasets are cached and shared, so a stale output from a prior run can mask a
    failure (the test would read last run's file instead of this run's). This
    fixture, built on top of the ``dataset`` factory, clears the known outputs
    BEFORE the test and removes them again on teardown so the cached INPUT files
    are left clean for the next run.
    """
    cleaned: list[tuple[Path, tuple[str, ...]]] = []

    def get_clean(name: str) -> Path:
        dpath = dataset(name)
        outputs = DATASET_OUTPUTS.get(name, ())
        for rel in outputs:  # setup: ensure the output slots are empty
            _remove(dpath / rel)
        cleaned.append((dpath, outputs))
        return dpath

    yield get_clean

    for dpath, outputs in cleaned:  # teardown: leave only the input files
        for rel in outputs:
            _remove(dpath / rel)
