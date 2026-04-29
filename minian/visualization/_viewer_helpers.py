"""Small shared helpers for Panel metadata widgets and frame playback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from panel import widgets as pnwgt

from ._viz_constants import ImagePalette, MetaSelect


def build_meta_select_widgets(
    meta_dicts: Mapping[str, Sequence[Any]],
    *,
    height: int = MetaSelect.WIDGET_HEIGHT,
    width: int = MetaSelect.WIDGET_WIDTH,
) -> dict[str, pnwgt.Select]:
    """One ``Select`` per metadata dimension (name + options from ``meta_dicts``)."""
    return {
        str(d): pnwgt.Select(name=str(d), options=list(v), height=height, width=width)
        for d, v in meta_dicts.items()
    }


def wire_meta_select_widgets(
    widgets: Mapping[str, pnwgt.Select],
    state: MutableMapping[str, Any],
    on_change: Callable[[], None],
) -> None:
    """Update ``state[dim]`` from each select and call ``on_change()`` when value changes."""
    for dim, wgt in widgets.items():

        def _watch(evt, *, _dim: str = dim):
            state[_dim] = evt.new
            on_change()

        wgt.param.watch(_watch, "value")


def wire_frame_player_index(
    player: pnwgt.Player,
    emit_frame: Callable[[int], None],
) -> None:
    """When the player index changes, call ``emit_frame(new_index)`` (skips no-op updates)."""
    def _play(evt):
        if evt.old != evt.new:
            emit_frame(int(evt.new))

    player.param.watch(_play, "value")


def footprint_image_opts(height_coords: Any, width_coords: Any) -> dict[str, Any]:
    """HoloViews flat opts for spatial footprint ``hv.Image`` (Viridis, size from coords)."""
    return {
        "frame_height": len(height_coords),
        "frame_width": len(width_coords),
        "cmap": ImagePalette.VIRIDIS_DISPLAY,
    }
