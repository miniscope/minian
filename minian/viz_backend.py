"""Visualization backend adapter.

This module centralizes HoloViews imports so we can incrementally migrate away
from HoloViews without touching every visualization call site at once.
"""

try:
    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread
    from holoviews.streams import (
        BoxEdit,
        DoubleTap,
        Pipe,
        RangeXY,
        Selection1D,
        Stream,
        Tap,
    )
    from holoviews.util import Dynamic
except Exception as exc:
    raise ImportError(
        "Visualization backend dependencies are missing. "
        "Install Minian visualization extras (for now this includes HoloViews)."
    ) from exc


__all__ = [
    "hv",
    "datashade",
    "dynspread",
    "BoxEdit",
    "DoubleTap",
    "Pipe",
    "RangeXY",
    "Selection1D",
    "Stream",
    "Tap",
    "Dynamic",
]
