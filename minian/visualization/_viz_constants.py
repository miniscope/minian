"""HoloViews / Panel literals: sizes, palettes, subplot labels, widget geometry."""

from __future__ import annotations

# ``Preprocess.{IMAGE_TITLE, …}`` feed ``hv.relabel()`` (subplot label), not
# ``hv.opts`` ``title`` / Bokeh ``Figure.title`` strings (see ``visualization`` package doc).

from enum import StrEnum


class ImagePalette(StrEnum):
    """Colormap names for HoloViews / Bokeh-style ``cmap=`` arguments."""

    VIRIDIS_DISPLAY = "Viridis"
    VIRIDIS_BOKEH = "viridis"


class SummaryStat(StrEnum):
    """Allowed ``VArrayViewer`` summary curve names."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    DIFF = "diff"


class Preprocess:
    """Sizes and subplot label templates for :func:`pipeline_plots.visualize_preprocess`."""

    FRAME_WIDTH = 500
    IMAGE_TITLE = "Image {label} {group} {dimensions}"
    CONTOURS_TITLE = "Contours {label} {group} {dimensions}"


class Datashade:
    """Datashader overlay options."""

    NDCURVE_MIN_ALPHA = 200


class Seeds:
    """``visualize_seeds`` layout and colors."""

    FRAME_WIDTH = 600
    POINTS_UNMASKED_COLOR = "white"
    MASK_FALSE_COLOR = "red"


class Gmm:
    """``visualize_gmm_fit`` figure geometry and histogram styling."""

    FIG_WIDTH = 500
    FIG_HEIGHT = 350
    HIST_FILL_ALPHA = 0.6
    HIST_FILL_COLOR = "gray"


class Spatial:
    """Spatial / CNMF footprint plots."""

    IMAGE_FRAME_WIDTH = 400
    TEMPORAL_CURVE_FRAME_WIDTH = 500
    TEMPORAL_CURVE_FRAME_HEIGHT = 50
    POINTS_SIZE = 8
    POINTS_FILL_ALPHA = 0.2


class Temporal:
    """Temporal-update and pulse-preview plots."""

    UNIT_MAP_FRAME_HEIGHT = 400
    UNIT_MAP_FRAME_WIDTH = 1000
    SPATIAL_FOOTPRINT_FRAME_WIDTH = 500
    PULSE_PREVIEW_LEN = 500


class Motion:
    """``visualize_motion`` frame sizes and colormap."""

    IMAGE_FRAME_WIDTH = 500
    IMAGE_ASPECT = 3
    CURVE_FRAME_WIDTH = 500
    CURVE_ASPECT = 2
    DIVERGING_CMAP = "RdBu"


class VArray:
    """``VArrayViewer`` HoloViews opts and histogram labels."""

    FRAME_WIDTH = 500
    HIST_SIDE_WIDTH = 150
    HIST_NUM_BINS = 50
    HIST_XLABEL = "fluorescence"
    HIST_YLABEL = "freq"
    SUMMARY_ASPECT = 3
    SUMMARY_RGB_HEIGHT_FLOOR = 120
    POLYGON_FILL_ALPHA = 0.3
    POLYGON_LINE_COLOR = "white"
    VLINE_COLOR = "red"


class Player:
    """Panel playback toolbar (``pnwgt.Player``) and Update Mask button."""

    WIDTH = 650
    HEIGHT = 90
    INTERVAL_MS = 10
    UPDATE_MASK_BUTTON_WIDTH = 100
    UPDATE_MASK_BUTTON_HEIGHT = 30


class MetaSelect:
    """Metadata dimension ``Select`` widgets (``_viewer_helpers``)."""

    WIDGET_HEIGHT = 45
    WIDGET_WIDTH = 120


class PanelLayout:
    """General Panel sizing modes."""

    SIZING_STRETCH_WIDTH = "stretch_width"


__all__ = [
    "Datashade",
    "Gmm",
    "ImagePalette",
    "MetaSelect",
    "Motion",
    "PanelLayout",
    "Player",
    "Preprocess",
    "Seeds",
    "Spatial",
    "SummaryStat",
    "Temporal",
    "VArray",
]
