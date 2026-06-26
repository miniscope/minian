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
            visualize_spatial_partition(max_proj, positions, membership, adj=adj, n_frames=100)


class TestVisualizeSeeds:
    """Pin the seed-overlay z-order: kept (True) seeds must render on top of
    filtered-out (False) seeds so a good seed is never hidden behind a rejected
    one, regardless of the row order of the seeds dataframe."""

    def _max_proj(self):
        return xr.DataArray(
            np.zeros((10, 10), dtype="float32"),
            dims=("height", "width"),
            coords={"height": np.arange(10), "width": np.arange(10)},
        )

    def test_true_seeds_render_on_top_of_false(self):
        import holoviews as hv
        import pandas as pd

        from ..visualization import visualize_seeds

        hv.extension("bokeh")  # .options() resolves against a loaded backend
        # Interleave True/False so a single-layer plot would draw them in mixed
        # order; the overlay split must still put every True seed on top.
        seeds = pd.DataFrame(
            {
                "height": [1, 2, 3, 4],
                "width": [1, 2, 3, 4],
                "seeds": [5, 6, 7, 8],
                "mask_good": [True, False, True, False],
            }
        )
        ov = visualize_seeds(self._max_proj(), seeds, mask="mask_good")
        points = ov.traverse(specs=[hv.Points])
        assert len(points) == 2
        # traverse preserves overlay (z) order; the last layer is on top.
        bottom, top = points
        assert top.dimension_values("mask_good").size == 2
        assert bool(top.dimension_values("mask_good").all())  # top = kept seeds
        assert not bool(bottom.dimension_values("mask_good").any())  # bottom = rejected

    def test_unmasked_returns_single_points_layer(self):
        import holoviews as hv
        import pandas as pd

        from ..visualization import visualize_seeds

        hv.extension("bokeh")  # .options() resolves against a loaded backend
        seeds = pd.DataFrame(
            {"height": [1, 2], "width": [1, 2], "seeds": [5, 6]}
        )
        ov = visualize_seeds(self._max_proj(), seeds)
        assert len(ov.traverse(specs=[hv.Points])) == 1
