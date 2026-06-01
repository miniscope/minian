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

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
ARTIFACT_DIR = Path("artifact").resolve()


def discover_notebooks():
    """All bundled notebooks, as ``bundle/notebook.ipynb`` POSIX relpaths."""
    return sorted(
        p.relative_to(NOTEBOOKS_DIR).as_posix()
        for p in NOTEBOOKS_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )


def require_dataset(name):
    """Ensure a demo dataset is available locally, or skip the test.

    Resolving it here (download + cache, or via ``MINIAN_DATA_DIR``) means the
    notebook's own ``fetch`` call hits the cache. Skips the test if the dataset
    is unavailable (unpublished, with no ``MINIAN_DATA_DIR`` copy); a missing or
    checksum-mismatched ``MINIAN_DATA_DIR`` file fails hard rather than skipping.
    """
    try:
        return dataset_path(name)
    except RuntimeError as exc:
        pytest.skip(str(exc))


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
