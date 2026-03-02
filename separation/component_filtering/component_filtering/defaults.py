"""Default parameters for the component filtering module.

The original pipeline.ipynb does not have an explicit filtering step;
update_temporal implicitly drops zero-trace units. This module adds
an explicit post-processing layer with quality metrics.
"""
from __future__ import annotations


def get_defaults() -> dict:
    """Return the default component filtering config."""
    return {
        # Optional final merge pass (disabled by default for exact parity)
        "param_final_merge": None,
        # Quality thresholds (informational — set to None to accept all)
        "snr_threshold": None,
        "spatial_contiguity_threshold": None,
        "temporal_stability_threshold": None,
    }
