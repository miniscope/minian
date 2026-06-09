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


def execute_notebook(relpath, output):
    """Execute ``notebooks/<relpath>`` with nbconvert, writing to ``artifact/``.

    ``output`` is the output base name (no extension); the executed notebook is
    written to ``artifact/<output>.ipynb`` (the docs build reads it from there).
    ``check=True`` turns a non-zero nbconvert exit (a failed notebook) into a
    test failure.
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
