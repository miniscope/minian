"""Spec-hash → zarr caching, kept as a thin library concern over :func:`simulate`.

A realistic spec takes tens of seconds to simulate, and the Step 9–10 parameter
sweeps re-request the same recordings many times. :func:`simulate_cached` keys a
recording by its :attr:`~minian.simulation.spec.Spec.cache_key` (a hash of the
canonical spec JSON): a cache hit is loaded from disk, a miss simulates once and
saves the result. ``simulate()`` itself stays pure — it has no I/O and no disk
side effects — so caching composes on top rather than being baked in.

The cache directory defaults to ``~/.cache/minian/sim`` and is overridable with
the ``MINIAN_SIM_CACHE`` environment variable. Because ``output.save_intermediates``
(and every other knob) is part of the spec, it folds into the cache key — a
recording cached without snapshots can never falsely satisfy a request that wants
them; the keys simply differ.
"""

from __future__ import annotations

import os
from pathlib import Path

from minian.simulation.recording import Recording
from minian.simulation.simulate import simulate
from minian.simulation.spec import Spec

#: Default cache root, used when ``$MINIAN_SIM_CACHE`` is unset.
DEFAULT_CACHE_DIR = "~/.cache/minian/sim"


def cache_dir() -> Path:
    """The resolved cache root: ``$MINIAN_SIM_CACHE`` if set, else :data:`DEFAULT_CACHE_DIR`.

    The returned path is user-expanded (``~``) but not created — :func:`simulate_cached`
    makes it on first write.
    """
    return Path(os.environ.get("MINIAN_SIM_CACHE", DEFAULT_CACHE_DIR)).expanduser()


def cache_path(spec: Spec, root: str | Path | None = None) -> Path:
    """The on-disk path a recording for ``spec`` is cached at: ``{root}/{cache_key}.zarr``."""
    base = Path(root).expanduser() if root is not None else cache_dir()
    return base / f"{spec.cache_key()}.zarr"


def simulate_cached(spec: Spec, *, root: str | Path | None = None) -> Recording:
    """Return the recording for ``spec``, loading from cache or simulating on a miss.

    Parameters
    ----------
    spec
        The recording spec; its :attr:`~minian.simulation.spec.Spec.cache_key`
        is the cache key.
    root
        Cache directory. Defaults to :func:`cache_dir` (``$MINIAN_SIM_CACHE`` or
        ``~/.cache/minian/sim``).

    Notes
    -----
    A hit is served by :meth:`Recording.load`, which re-verifies the stored spec
    hash; a miss runs :func:`simulate` and persists the result with
    :meth:`Recording.save` before returning it.
    """
    path = cache_path(spec, root)
    if path.exists():
        return Recording.load(path)
    rec = simulate(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec.save(path)
    return rec
