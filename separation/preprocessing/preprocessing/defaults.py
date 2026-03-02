"""Default parameters for the preprocessing module.

Extracted from pipeline.ipynb parameter cells.
"""
from __future__ import annotations

import numpy as np


def get_defaults() -> dict:
    """Return the default preprocessing config matching pipeline.ipynb."""
    return {
        "param_load_videos": {
            "pattern": r"msCam[0-9]+\.avi$",
            "dtype": np.uint8,
            "downsample": dict(frame=1, height=1, width=1),
            "downsample_strategy": "subset",
        },
        "param_denoise": {"method": "median", "ksize": 7},
        "param_background_removal": {"method": "tophat", "wnd": 15},
        "subset": dict(frame=slice(0, None)),
        "intpath": "./minian_intermediate",
    }
