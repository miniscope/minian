"""Unit tests for discrete functions in ``minian.cnmf`` on synthetic input."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.neighbors import radius_neighbors_graph

from ..cnmf import adj_corr


class TestAxisOrientation:
    """Regression for #302."""

    def test_adj_corr_returns_true_pearson(self):
        # adj_corr's output must match np.corrcoef on the underlying pixel
        # traces. Pre-fix, the multi-dim-indexed path silently transposed
        # vsub and idx_corr reduced over the wrong axis, returning values
        # uncorrelated with the actual traces.
        rng = np.random.RandomState(11)
        n, t, h, w = 16, 80, 25, 25
        varr = xr.DataArray(
            rng.standard_normal((h, w, t)).astype("float32"),
            dims=("height", "width", "frame"),
            coords={
                "height": np.arange(h), "width": np.arange(w),
                "frame": np.arange(t),
            },
        )
        hs = rng.randint(0, h, size=n)
        ws = rng.randint(0, w, size=n)
        nod_df = pd.DataFrame({"height": hs, "width": ws})
        adj = radius_neighbors_graph(nod_df.values, radius=8).astype(bool)
        out = adj_corr(varr, adj, nod_df, freq=None).toarray()

        traces = np.stack(
            [varr.isel(height=hs[k], width=ws[k]).values for k in range(n)]
        )
        for i, j in zip(*out.nonzero()):
            truth = np.corrcoef(traces[i], traces[j])[0, 1]
            assert out[i, j] == pytest.approx(truth, abs=1e-5)
