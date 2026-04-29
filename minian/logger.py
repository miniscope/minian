"""Central logging helpers for Minian."""

from __future__ import annotations

import logging
import sys
from typing import Any

from .constants import MINIAN


def configure_logging(
    level: int | str = logging.INFO,
    *,
    force: bool = False,
    stream: Any | None = None,
) -> None:
    """Attach a :class:`~logging.StreamHandler` to the ``minian`` logger.

    Call once at process start (e.g. notebook first cell or CLI entry).

    Without calling this, child loggers under ``minian`` propagate to the root
    logger unless you attach a NullHandler elsewhere. Prefer calling this once
    for consistent formatting.
    """
    lg = logging.getLogger(MINIAN)

    def _non_null_handlers() -> list:
        return [h for h in lg.handlers if not isinstance(h, logging.NullHandler)]

    if _non_null_handlers() and not force:
        lg.setLevel(level)
        return

    if force:
        lg.handlers.clear()
    else:
        for h in list(lg.handlers):
            if isinstance(h, logging.NullHandler):
                lg.removeHandler(h)

    fmt = logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s",
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(fmt)
    lg.addHandler(handler)
    lg.setLevel(level)
    lg.propagate = False
