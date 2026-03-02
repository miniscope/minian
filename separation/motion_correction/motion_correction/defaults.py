"""Default parameters for the motion correction module.

Extracted from pipeline.ipynb parameter cells.
"""
from __future__ import annotations


def get_defaults() -> dict:
    """Return the default motion correction config matching pipeline.ipynb."""
    return {
        "param_estimate_motion": {"dim": "frame"},
        "subset_mc": None,
    }
