import os
import shutil
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from ..data import dataset_path


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
def fetch_dataset() -> Callable[[str], Path]:
    """Session-scoped resolver for demo datasets (download + cache once).

    Returns a ``get(name) -> Path`` callable so any fixture or test can pluck
    the dataset it needs from one shared resolver. Resolving the same name
    twice is cheap (the ``minian.data`` cache), and a dataset that cannot be
    resolved fails rather than skips (see :func:`minian.data.dataset_path`).
    Read-only consumers can depend on this directly; output-writing tests
    should use the ``dataset`` fixture so their outputs get cleaned up.
    """
    return dataset_path


@pytest.fixture
def dataset(request, fetch_dataset: Callable[[str], Path]) -> Path:
    """Function-scoped dataset directory with output cleanup tied to its scope.

    Parametrize indirectly with the dataset name; the known notebook outputs
    (``DATASET_OUTPUTS``) are cleared before the test and removed again on
    teardown, so cached INPUT files are reused while stale outputs can never
    mask a failure (the test would otherwise read a prior run's results)::

        @pytest.mark.parametrize("dataset", ["pipeline-demo"], indirect=True)
        def test_x(dataset):
            dpath = dataset  # ready, with its output slots cleared

    Tying cleanup to the fixture (rather than a helper the test must remember
    to call) means requesting the dataset is what schedules its cleanup.
    """
    name = request.param
    dpath = fetch_dataset(name)
    outputs = DATASET_OUTPUTS.get(name, ())
    for rel in outputs:  # setup: ensure the output slots are empty
        _remove(dpath / rel)
    yield dpath
    for rel in outputs:  # teardown: leave only the input files
        _remove(dpath / rel)
