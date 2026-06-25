"""Shared helper for the notebook-execution tests.

Notebooks now live inside the package (``minian/notebooks/**``) and pull their
demo data on demand via :mod:`minian.data`. Datasets are resolved by the
``dataset`` / ``fetch_dataset`` fixtures (see ``conftest.py``) so a notebook's
own ``fetch`` call hits the warm cache; this module just locates and executes
a notebook with nbconvert.
"""

import subprocess
import sys
from pathlib import Path

from ..notebooks import notebook_root

NOTEBOOKS_DIR = notebook_root()
ARTIFACT_DIR = Path("artifact").resolve()


def execute_notebook(relpath: str, output: str) -> Path:
    """Execute ``notebooks/<relpath>`` with nbconvert, writing to ``artifact/``.

    ``output`` is the output base name (no extension); the executed notebook is
    written to ``artifact/<output>.ipynb`` (the docs build reads it from there).
    ``check=True`` turns a non-zero nbconvert exit (a failed notebook) into a
    test failure. The per-cell ``--ExecutePreprocessor.timeout`` bounds a hung
    cell (e.g. a deadlocked dask cluster) so it surfaces as a failing test with
    a traceback rather than blocking until CI's wall-clock ceiling.
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
            "--ExecutePreprocessor.timeout=600",
            "--output-dir",
            str(ARTIFACT_DIR),
            "--output",
            output,
            str(NOTEBOOKS_DIR / relpath),
        ],
        check=True,
    )
    return ARTIFACT_DIR / f"{output}.ipynb"
