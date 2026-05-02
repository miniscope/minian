"""Compute xarray/dask data on a local scheduler (for notebooks with a distributed Client)."""

from __future__ import annotations

import logging
from typing import Any

import dask

log = logging.getLogger(__name__)


def materialize_local(x: Any, scheduler: str = "threads") -> Any:
    """
    Materialize lazy data, preferring a **local** thread pool over distributed workers.

    With an active :class:`distributed.Client`, a plain ``.compute()`` can run
    heavy tasks on workers and trigger OOM / ``KilledWorker``. This helper first
    tries ``scheduler=\"threads\"`` so small arrays are computed in the notebook
    process.

    If the graph still references **cluster-only keys** (common after ``persist()``
    on workers), a thread-local run can raise ``ValueError: Missing dependency …``.
    In that case we **retry once** with the **default** scheduler (usually the
    active Client) so a gather/recompute can complete.

    Parameters
    ----------
    x
        ``xarray.DataArray`` / ``Dataset``, ``dask.array.Array``, or any object
        with ``compute()``. Non-lazy values are returned unchanged.
    scheduler
        Passed to :func:`dask.config.set` for the first attempt (default
        ``\"threads\"``).

    Returns
    -------
    The computed or unchanged value.
    """
    compute = getattr(x, "compute", None)
    if compute is None:
        return x
    try:
        with dask.config.set(scheduler=scheduler):
            return compute()
    except ValueError as err:
        if "missing dependency" not in str(err).lower():
            raise
        log.warning(
            "materialize_local: %s scheduler failed (%s); retrying with default scheduler",
            scheduler,
            err,
        )
        return compute()
