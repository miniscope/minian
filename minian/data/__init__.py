"""On-demand fetching of MiniAn demo datasets.

Demo data lives outside the git repository (see :mod:`minian.data._registry`).
:func:`fetch` downloads a named dataset once, caches it to the OS cache dir,
verifies it against recorded SHA256 checksums, and returns the local directory::

    from minian.data import fetch
    dpath = fetch("pipeline-demo")   # pathlib.Path to a dir of msCam*.avi

The cache location follows ``pooch.os_cache`` (e.g. ``~/.cache/minian`` on
Linux, ``%LOCALAPPDATA%\\minian\\Cache`` on Windows) and can be overridden with
the ``MINIAN_CACHE_DIR`` environment variable (handled natively by pooch's
``env=`` argument).

Offline escape hatch: set ``MINIAN_DATA_DIR`` to a directory that already
contains the datasets as ``<MINIAN_DATA_DIR>/<dataset-name>/...`` and
:func:`fetch` returns from there (verifying checksums) without any network
access. This is deliberately *not* delegated to pooch: pooch attempts a
download whenever a registered file is missing (regardless of its
``allow_updates`` setting) and conflates the writable cache dir with the
read-only data dir, neither of which works for an air-gapped node.
"""

import hashlib
import os
from pathlib import Path

import pooch

from . import _registry

__all__ = ["fetch", "fetch_all", "datasets", "dataset_path", "cache_dir"]


def cache_dir() -> Path:
    """Root cache directory used for downloaded datasets.

    Tracks where pooch stores downloads, including the ``MINIAN_CACHE_DIR``
    override.
    """
    return Path(POOCH.path)


def datasets() -> dict[str, str]:
    """Mapping of dataset name -> human-readable description."""
    return {name: meta["description"] for name, meta in _registry.DATASETS.items()}


def _check_name(name: str) -> _registry.ZenodoDataset:
    try:
        return _registry.DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {', '.join(_registry.DATASETS)}"
        ) from None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_pooch() -> pooch.Pooch:
    """Build the pooch instance from the registry (one key per file).

    The registry is hardcoded, so a dataset missing its ``zenodo_record`` is a
    bug in :mod:`minian.data._registry`, not a runtime condition: indexing
    ``meta["zenodo_record"]`` raises ``KeyError`` loudly rather than silently
    skipping the file.
    """
    registry, urls = {}, {}
    for name, meta in _registry.DATASETS.items():
        record = meta["zenodo_record"]
        for relpath, info in meta["files"].items():
            key = f"{name}/{relpath}"
            registry[key] = f"sha256:{info['sha256']}"
            urls[key] = _registry.zenodo_url(record, info["zenodo"])
    return pooch.create(
        path=pooch.os_cache("minian"),
        base_url="",  # every file has an explicit per-key URL
        env="MINIAN_CACHE_DIR",  # pooch reads the cache-dir override itself
        registry=registry,
        urls=urls,
    )


POOCH = _make_pooch()


def _local_dir(name: str) -> Path | None:
    """Return a verified local dataset dir if ``MINIAN_DATA_DIR`` provides one."""
    root = os.environ.get("MINIAN_DATA_DIR")
    if not root:
        return None
    ddir = Path(root) / name
    files = _check_name(name)["files"]
    for relpath, info in files.items():
        fpath = ddir / relpath
        if not fpath.is_file():
            raise FileNotFoundError(
                f"MINIAN_DATA_DIR is set but {fpath} is missing for dataset {name!r}."
            )
        actual = _sha256(fpath)
        if actual != info["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {fpath}:\n  expected {info['sha256']}\n  got      {actual}"
            )
    return ddir


def fetch(name: str, *, progressbar: bool = True) -> Path:
    """Download (if needed), verify, and return the local path of a dataset.

    Parameters
    ----------
    name
        Dataset name, one of :func:`datasets`.
    progressbar
        Show a download progress bar (requires ``tqdm``; ignored if absent).

    Returns
    -------
    pathlib.Path
        Directory containing the dataset files, laid out exactly as recorded
        in the registry (subdirectories preserved).
    """
    meta = _check_name(name)

    local = _local_dir(name)
    if local is not None:
        return local

    for relpath in meta["files"]:
        POOCH.fetch(f"{name}/{relpath}", progressbar=progressbar)
    return cache_dir() / name


def fetch_all(*, progressbar: bool = True) -> list[Path]:
    """Fetch every registered dataset (used by ``minian data download --all``)."""
    return [fetch(name, progressbar=progressbar) for name in _registry.DATASETS]


def dataset_path(name: str) -> Path:
    """Local directory for a dataset (fetching it first if necessary)."""
    return fetch(name, progressbar=False)
