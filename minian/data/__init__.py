"""On-demand fetching of MiniAn demo datasets.

Demo data lives outside the git repository (see :mod:`minian.data._registry`).
:func:`fetch` downloads a named dataset once, caches it to the OS cache dir,
verifies it against recorded SHA256 checksums, and returns the local directory::

    from minian.data import fetch
    dpath = fetch("pipeline-demo")   # pathlib.Path to a dir of msCam*.avi

The cache location follows ``pooch.os_cache`` (e.g. ``~/.cache/minian`` on
Linux, ``%LOCALAPPDATA%\\minian\\Cache`` on Windows) and can be overridden with
the ``MINIAN_CACHE_DIR`` environment variable.

Offline / air-gapped use: pooch verifies every file against its recorded
SHA256 on fetch and only reaches the network when a file is absent or fails
that check. So to run without network access, point ``MINIAN_CACHE_DIR`` at a
directory that already holds the data in the cache layout
(``<MINIAN_CACHE_DIR>/<dataset-name>/<relpath>``, which is exactly how the
registry keys are named); :func:`fetch` then verifies and returns those files
without touching the network.
"""

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


def fetch(name: str, *, progressbar: bool = True) -> Path:
    """Download (if needed), verify, and return the local path of a dataset.

    pooch fetches each file once, verifies it against the registry's SHA256,
    and re-downloads only if it is missing or fails that check.

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
    for relpath in meta["files"]:
        POOCH.fetch(f"{name}/{relpath}", progressbar=progressbar)
    return cache_dir() / name


def fetch_all(*, progressbar: bool = True) -> list[Path]:
    """Fetch every registered dataset; convenience wrapper over :func:`fetch`."""
    return [fetch(name, progressbar=progressbar) for name in _registry.DATASETS]


def dataset_path(name: str) -> Path:
    """Local directory for a dataset (fetching it first if necessary)."""
    return fetch(name, progressbar=False)
