"""Unit tests for discrete functions in ``minian.visualization`` on synthetic input."""

import numpy as np
import pytest
import scipy.sparse
import xarray as xr


class TestVisualizeSpatialPartition:
    """The visualization layer is mostly covered by notebook execution in
    test_pipeline.py; here we just pin the explicit contract violation
    raise so that misaligned inputs fail loudly instead of silently
    misrendering every point."""

    def test_rejects_misaligned_positions_membership(self):
        # Deferred import: the visualization module pulls in panel/bokeh
        # at import time, which is heavier than the rest of the test
        # file needs.
        from ..visualization import visualize_spatial_partition

        max_proj = xr.DataArray(
            np.zeros((10, 10), dtype="float32"),
            dims=("height", "width"),
            coords={"height": np.arange(10), "width": np.arange(10)},
        )
        positions = np.zeros((5, 2), dtype=float)
        membership = np.zeros(4, dtype=int)  # length mismatch
        adj = scipy.sparse.csr_matrix((5, 5))
        with pytest.raises(
            ValueError,
            match="positions has 5 rows but membership has 4",
        ):
            visualize_spatial_partition(
                max_proj, positions, membership, adj=adj, n_frames=100
            )
