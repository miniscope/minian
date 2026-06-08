"""Shared helpers for the notebook-execution tests.

Notebooks now live inside the package (``minian/notebooks/**``) and pull their
demo data on demand via :mod:`minian.data`. These helpers locate a notebook,
make sure its dataset is available (downloading/caching it, or skipping the
test cleanly when it cannot be resolved), and execute it with nbconvert.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from ..data import dataset_path
from ..notebooks import notebook_root

NOTEBOOKS_DIR = notebook_root()
ARTIFACT_DIR = Path("artifact").resolve()


def require_dataset(name):
    """Ensure a demo dataset is available locally, or skip the test.

    Resolving it here (download + cache, or from a prepopulated
    ``MINIAN_CACHE_DIR``) means the notebook's own ``fetch`` call hits the
    cache. Skips the test if the dataset cannot be resolved.
    """
    try:
        return dataset_path(name)
    except OSError as exc:
        # pooch raises requests/OS errors (a subclass of OSError) when it can't
        # download and the file isn't already cached; skip rather than fail.
        # An unknown dataset name (KeyError) still fails hard.
        pytest.skip(f"demo dataset {name!r} unavailable: {exc}")


def execute_notebook(relpath, output):
    """Execute ``notebooks/<relpath>`` with nbconvert, writing to ``artifact/``.

    ``output`` is the output base name (no extension); the executed notebook is
    written to ``artifact/<output>.ipynb`` (the docs build reads it from there).
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(ARTIFACT_DIR),
            "--output",
            output,
            str(NOTEBOOKS_DIR / relpath),
        ],
        check=True,
    )
    return ARTIFACT_DIR / f"{output}.ipynb"
