"""Central logging helpers and shared wall-clock banners for CLIs."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Iterator

from ..constants import MINIAN


class ANSIColor(StrEnum):
    """ANSI SGR foreground codes (parameter after ``ESC[``, before ``m``)."""

    BRIGHT_RED = "91"
    BRIGHT_CYAN = "96"


def _sgr(color: ANSIColor | str | None) -> str | None:
    if color is None:
        return None
    if isinstance(color, ANSIColor):
        return color.value
    return color


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


def configure_cli_logging() -> None:
    """CLI entry: honor ``MINIAN_LOG_LEVEL`` and attach a handler (``force=True``)."""
    configure_logging(os.getenv("MINIAN_LOG_LEVEL", "INFO"), force=True)


def format_wall_duration(elapsed: float) -> str:
    """Human-readable span: ``12.345s``, ``3m 12.345s``, ``1h 4m 2.345s``."""
    if elapsed < 60:
        return f"{elapsed:.3f}s"
    if elapsed < 3600:
        m = int(elapsed // 60)
        s = elapsed - m * 60
        return f"{m}m {s:.3f}s"
    h = int(elapsed // 3600)
    rem = elapsed - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    return f"{h}h {m}m {s:.3f}s"


def _colorize(msg: str, sgr: str | None) -> str:
    if sgr is None or not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return msg
    return f"\033[{sgr}m{msg}\033[0m"


def print_wall_elapsed(
    prefix: str,
    label: str,
    elapsed: float,
    *,
    color: ANSIColor | str | None = None,
) -> None:
    """Print one optional-colored line: ``prefix label: <duration>``."""
    body = f"{prefix} {label}: {format_wall_duration(elapsed)}"
    print(_colorize(body, _sgr(color)))


def print_wall_since(
    prefix: str,
    label: str,
    t0: float,
    *,
    color: ANSIColor | str | None = None,
) -> None:
    """Print wall time since perf-counter anchor ``t0``."""
    print_wall_elapsed(prefix, label, time.perf_counter() - t0, color=color)


@contextmanager
def wall_section(
    prefix: str,
    label: str,
    *,
    color: ANSIColor | str | None = None,
) -> Iterator[None]:
    """On exit, print a wall-clock line for ``label`` (success or failure)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print_wall_elapsed(prefix, label, time.perf_counter() - t0, color=color)
