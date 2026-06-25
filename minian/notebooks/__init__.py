"""Jupyter notebooks shipped with minian.

The notebooks live inside the installed package so that ``pip install minian``
gives you everything needed to read and edit them offline. Copy one into your
working directory with the ``minian notebooks`` CLI::

    minian notebooks list
    minian notebooks copy pipeline

A "notebook" here is just a top-level entry under this package (the
``pipeline/`` and ``cross_registration/`` folders); copying one copies the
folder and its assets so every relative reference keeps working. Large demo
datasets are *not* shipped here; they are fetched on demand via
:mod:`minian.data`. Full usage guides live in the MiniAn docs, not in this
package.
"""

import importlib.resources as ir
import shutil
from pathlib import Path

__all__ = ["NOTEBOOKS", "notebook_root", "notebook_files", "copy"]

# Canonical one-line descriptions, keyed by notebook name. The prose guides
# live in the docs site (docs/source/pipeline, docs/source/cross_reg); the CLI
# only needs a short label, so we don't ship a README per notebook.
NOTEBOOKS: dict[str, str] = {
    "pipeline": (
        "End-to-end calcium-imaging pipeline: load videos, preprocess, "
        "motion-correct, then CNMF (spatial/temporal)."
    ),
    "cross_registration": ("Align and match cells across recording sessions of the same animal."),
    "pipeline_groundtruth": (
        "Run the full pipeline on a minisim synthetic recording and score every "
        "stage against ground truth (needs the 'training' extra)."
    ),
}

# Never copy/list build junk or notebook execution output.
_SKIP = {"__pycache__", ".ipynb_checkpoints", "minian_intermediate"}
_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".ipynb_checkpoints",
    "*.nbi",
    "*.nbc",
    "minian_intermediate",
    "*.zarr",
    "*.mp4",
    "*.nc",
)


def notebook_root() -> Path:
    """Filesystem directory holding the bundled notebooks."""
    return Path(ir.files(__name__))


def notebook_files() -> list[str]:
    """Every shipped notebook, as ``name/notebook.ipynb`` POSIX relpaths."""
    root = notebook_root()
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.ipynb")
        if not _SKIP.intersection(p.parts)
    )


def copy(name: str, dest: Path) -> list[Path]:
    """Copy notebook entries matching ``name`` into ``dest``.

    A name is just a prefix: ``copy("pipeline", dest)`` copies everything
    matching ``pipeline*`` at the notebooks root (the ``pipeline/`` folder and
    its assets), which is why there is no separate notion of a "bundle". A
    fully-qualified ``name/notebook.ipynb`` works too.
    """
    root = notebook_root()
    matches = [p for p in sorted(root.glob(f"{name}*")) if p.name not in _SKIP]
    if not matches:
        raise KeyError(f"No notebook matching {name!r}. Available: {', '.join(NOTEBOOKS)}")
    dest = Path(dest)
    copied = []
    for src in matches:
        target = dest / src.name
        if src.is_dir():
            shutil.copytree(src, target, ignore=_IGNORE, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        copied.append(target)
    return copied
