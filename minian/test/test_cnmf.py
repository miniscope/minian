"""Unit tests for discrete functions in ``minian.cnmf`` on synthetic input."""

import inspect
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xarray as xr
from sklearn.neighbors import radius_neighbors_graph

from ..cnmf import (
    adj_corr,
    graph_optimize_corr,
    partition_diagnostics,
    spatial_partition,
    unit_merge,
)
from ..initialization import initA, seeds_merge


# ---------------------------------------------------------------------------
# spatial_partition
# ---------------------------------------------------------------------------


class TestSpatialPartition:
    """k-d tree median-split partitioner properties."""

    def test_returns_one_label_per_node(self):
        positions = np.random.RandomState(0).uniform(0, 100, size=(50, 2))
        membership = spatial_partition(positions, target_chunk=10)
        assert membership.shape == (50,)
        assert membership.dtype.kind == "i"

    def test_labels_are_dense_from_zero(self):
        # Gaps in the label space would silently create empty groupby groups
        # downstream in graph_optimize_corr.
        positions = np.random.RandomState(1).uniform(0, 100, size=(200, 2))
        membership = spatial_partition(positions, target_chunk=32)
        unique = np.unique(membership)
        assert (unique == np.arange(len(unique))).all()

    def test_every_partition_within_target_chunk(self):
        positions = np.random.RandomState(2).uniform(0, 100, size=(317, 2))
        membership = spatial_partition(positions, target_chunk=50)
        assert np.bincount(membership).max() <= 50

    def test_balance_under_uniform_density(self):
        # 1024 / 128 = 8 leaves of exactly 128 under median split.
        positions = np.random.RandomState(3).uniform(0, 100, size=(1024, 2))
        membership = spatial_partition(positions, target_chunk=128)
        assert (np.bincount(membership) == 128).all()

    def test_small_input_is_single_partition(self):
        membership = spatial_partition(
            np.array([[1.0, 2.0], [3.0, 4.0]]), target_chunk=10
        )
        assert membership.tolist() == [0, 0]

    def test_single_point(self):
        positions = np.array([[5.0, 7.0]])
        membership = spatial_partition(positions, target_chunk=4)
        assert membership.tolist() == [0]

    def test_empty_input(self):
        membership = spatial_partition(np.empty((0, 2)), target_chunk=4)
        assert membership.shape == (0,)

    def test_deterministic(self):
        positions = np.random.RandomState(4).uniform(0, 100, size=(200, 2))
        a = spatial_partition(positions, target_chunk=32)
        b = spatial_partition(positions, target_chunk=32)
        assert (a == b).all()

    def test_splits_on_longer_axis(self):
        # Points stretched along height; the top and bottom 25 must not
        # share any partition label.
        positions = np.column_stack([np.linspace(0, 100, 100), np.zeros(100)])
        membership = spatial_partition(positions, target_chunk=25)
        assert set(membership[-25:]).isdisjoint(set(membership[:25]))

    def test_rejects_wrong_position_shape(self):
        with pytest.raises(ValueError, match="positions must be"):
            spatial_partition(np.array([1.0, 2.0, 3.0]), target_chunk=4)
        with pytest.raises(ValueError, match="positions must be"):
            spatial_partition(np.zeros((5, 3)), target_chunk=4)

    def test_rejects_invalid_target_chunk(self):
        with pytest.raises(ValueError, match="target_chunk"):
            spatial_partition(np.zeros((4, 2)), target_chunk=0)
        with pytest.raises(ValueError, match="target_chunk"):
            spatial_partition(np.zeros((4, 2)), target_chunk=-3)


# ---------------------------------------------------------------------------
# adj_corr + graph_optimize_corr position plumbing
# ---------------------------------------------------------------------------


def _synthetic_varr(height: int, width: int, frames: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    return xr.DataArray(
        rng.standard_normal((height, width, frames)).astype("float32"),
        dims=("height", "width", "frame"),
        coords={
            "height": np.arange(height),
            "width": np.arange(width),
            "frame": np.arange(frames),
        },
    )


class TestAdjCorrPositionPlumbing:
    """The `positions` kwarg affects chunking only, not correlation values."""

    def _setup(self, n=80, h=15, w=15, t=300, radius=4, seed=0):
        rng = np.random.RandomState(seed)
        varr = _synthetic_varr(h, w, t, seed=seed + 1)
        hs = rng.randint(0, h, size=n)
        ws = rng.randint(0, w, size=n)
        nod_df = pd.DataFrame({"height": hs, "width": ws})
        adj = radius_neighbors_graph(
            nod_df[["height", "width"]].values, radius=radius
        ).astype(bool)
        return varr, adj, nod_df

    def test_fallback_matches_explicit_positions(self):
        varr, adj, nod_df = self._setup()
        result_explicit = adj_corr(
            varr, adj, nod_df, freq=None,
            positions=nod_df[["height", "width"]].values,
        )
        result_fallback = adj_corr(varr, adj, nod_df, freq=None, positions=None)
        assert np.allclose(
            result_explicit.toarray(), result_fallback.toarray()
        )

    def test_correlations_invariant_to_partition_choice(self):
        # Scrambled positions force a different partition tree but every
        # (i, j) correlation must remain the same scalar value.
        varr, adj, nod_df = self._setup(seed=7)
        ref = adj_corr(varr, adj, nod_df, freq=None).toarray()
        bogus = np.random.RandomState(99).uniform(0, 1000, size=(len(nod_df), 2))
        scrambled = adj_corr(
            varr, adj, nod_df, freq=None, positions=bogus
        ).toarray()
        assert np.allclose(ref, scrambled)

    def test_output_shape_matches_input_adj(self):
        # One entry per undirected edge; callers mirror via `adj + adj.T`.
        varr, adj, nod_df = self._setup(n=40)
        out = adj_corr(varr, adj, nod_df, freq=None)
        assert isinstance(out, scipy.sparse.csr_matrix)
        assert out.shape == adj.shape
        out_pairs = set(zip(*out.nonzero()))
        adj_pairs = set(zip(*adj.nonzero()))
        assert out_pairs.issubset(adj_pairs)
        assert out.nnz == adj.nnz // 2

    def test_with_smoothing(self):
        # `freq` triggers the FFT lowpass branch; partition-invariance still
        # holds.
        varr, adj, nod_df = self._setup(t=400)
        a = adj_corr(varr, adj, nod_df, freq=0.1).toarray()
        b = adj_corr(
            varr, adj, nod_df, freq=0.1,
            positions=nod_df[["height", "width"]].values,
        ).toarray()
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# graph_optimize_corr: contract violations should fail loudly
# ---------------------------------------------------------------------------


class TestGraphOptimizeCorrContract:
    """Validate the error paths around the new positions/membership kwargs.

    Partitioning is a chunking hint, so silent misuse doesn't corrupt the
    correlation values --
    only memory balance and the printed pixel_recompute_ratio. Loud errors
    let the upstream bug surface immediately.
    """

    @staticmethod
    def _build_graph(n: int, with_pos_attrs: bool = True) -> nx.Graph:
        G = nx.Graph()
        for i in range(n):
            attrs = {"height": float(i), "width": float(i)} if with_pos_attrs else {}
            G.add_node(i, **attrs)
        for i in range(n - 1):
            G.add_edge(i, i + 1)
        return G

    def _tiny_varr(self):
        rng = np.random.RandomState(0)
        return xr.DataArray(
            rng.standard_normal((32, 32, 50)).astype("float32"),
            dims=("height", "width", "frame"),
            coords={
                "height": np.arange(32),
                "width": np.arange(32),
                "frame": np.arange(50),
            },
        )

    def test_rejects_positions_length_mismatch(self):
        G = self._build_graph(5)
        bad_positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="positions has 2 rows but G has 5"):
            graph_optimize_corr(
                self._tiny_varr(), G, freq=None, positions=bad_positions
            )

    def test_rejects_missing_position_attrs(self):
        G = self._build_graph(4, with_pos_attrs=False)
        with pytest.raises(ValueError, match="positions=None fallback requires"):
            graph_optimize_corr(self._tiny_varr(), G, freq=None, positions=None)

    def test_membership_attrs_are_plain_int(self):
        # JSON-serializable; spatial_partition returns np.int64 and we cast.
        G = self._build_graph(4)
        membership = spatial_partition(
            np.array([[float(i), float(i)] for i in range(4)]), target_chunk=2
        )
        nx.set_node_attributes(
            G, {k: {"part": int(v)} for k, v in zip(sorted(G.nodes), membership)}
        )
        payload = json.dumps(nx.node_link_data(G, edges="links"))
        parts = {n["id"]: n["part"] for n in json.loads(payload)["nodes"]}
        assert all(isinstance(p, int) for p in parts.values())


# ---------------------------------------------------------------------------
# unit_merge centroid handling (NaN-safe + single fused compute)
# ---------------------------------------------------------------------------


class TestUnitMergeCentroidPathway:
    """Behaviour of spatial_partition under the NaN / FOV-centre fallback
    that unit_merge installs when a footprint has zero mass."""

    def test_nan_positions_do_not_crash_partition(self):
        positions = np.array([
            [0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [2.0, 2.0], [3.0, 3.0],
        ])
        membership = spatial_partition(positions, target_chunk=2)
        assert membership.shape == (5,)
        assert (membership >= 0).all()

    def test_finite_fallback_gives_balanced_partition(self):
        # 5 zero-mass units parked at the FOV centre, mixed with 16 spread
        # units. The centred rows occupy at most 2 adjacent labels.
        centre = [16.0, 16.0]
        real_positions = np.array(
            [[i * 4.0, j * 4.0] for i in range(4) for j in range(4)]
        )
        positions = np.vstack([real_positions, np.tile(centre, (5, 1))])
        membership = spatial_partition(positions, target_chunk=5)
        assert np.bincount(membership).max() <= 5
        centred_labels = set(membership[-5:].tolist())
        assert len(centred_labels) <= 2
        if len(centred_labels) == 2:
            lo, hi = sorted(centred_labels)
            assert hi - lo == 1


# ---------------------------------------------------------------------------
# partition_diagnostics
# ---------------------------------------------------------------------------


class TestPartitionDiagnostics:
    """Arithmetic of the diagnostic helper, pinned on hand-built graphs."""

    def _line_graph_membership(self, n: int, chunk: int):
        # n nodes on a 1D line; stable argsort pairs consecutive nodes.
        positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        membership = spatial_partition(positions, target_chunk=chunk)
        adj = radius_neighbors_graph(positions, radius=1.5).astype(bool)
        return positions, membership, adj

    def test_sizes_and_n_parts_without_adj(self):
        _, membership, _ = self._line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership)
        assert diag["n_parts"] == 4
        assert diag["sizes"].tolist() == [2, 2, 2, 2]
        for absent in ("edges_per_partition", "cross_fraction", "mem_mb"):
            assert absent not in diag

    def test_edge_counts_on_line_graph(self):
        # 8 nodes on a line, radius=1.5 -> 7 consecutive-pair edges, chunk=2
        # pairs them into (0,1) (2,3) (4,5) (6,7) -> 4 intra, 3 cross.
        _, membership, adj = self._line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership, adj=adj)
        assert diag["total_edges"] == 7
        assert diag["edges_per_partition"].tolist() == [1, 1, 1, 1]
        assert diag["cross_edges"] == 3
        assert diag["cross_fraction"] == pytest.approx(3 / 7)

    def test_symmetric_and_triangular_adj_give_same_result(self):
        _, membership, adj = self._line_graph_membership(10, chunk=2)
        from scipy.sparse import tril
        diag_sym = partition_diagnostics(membership, adj=adj)
        diag_tri = partition_diagnostics(membership, adj=tril(adj, k=-1))
        for key in ("total_edges", "cross_edges", "cross_fraction"):
            assert diag_sym[key] == diag_tri[key]
        assert (diag_sym["edges_per_partition"] == diag_tri["edges_per_partition"]).all()

    def test_self_loops_are_dropped(self):
        # A node connected only to itself should NOT contribute to any
        # intra-partition or total-edge count.
        membership = np.array([0, 0, 1, 1])
        adj = scipy.sparse.csr_matrix(np.eye(4))  # only self-loops
        diag = partition_diagnostics(membership, adj=adj)
        assert diag["total_edges"] == 0
        assert diag["cross_fraction"] == 0.0
        assert diag["edges_per_partition"].tolist() == [0, 0]

    def test_memory_estimate(self):
        # n_frames=1000, default bytes_per_sample=4 -> 4000 B/node = ~3.9 kB.
        # sizes=[2,2,2,2] -> 8000 B/partition = 0.00763 MiB.
        _, membership, _ = self._line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership, n_frames=1000)
        expected_mib = 2 * 1000 * 4 / (1024 ** 2)
        assert diag["mem_mb"].tolist() == pytest.approx([expected_mib] * 4)

    def test_empty_membership_returns_empty_diag(self):
        diag = partition_diagnostics(
            np.empty(0, dtype=int),
            adj=scipy.sparse.csr_matrix((0, 0)),
            n_frames=100,
        )
        assert diag["n_parts"] == 0
        assert diag["sizes"].shape == (0,)
        assert diag["total_edges"] == 0
        assert diag["mem_mb"].shape == (0,)


# ---------------------------------------------------------------------------
# chunk-kwarg plumbing
# ---------------------------------------------------------------------------


class TestChunkKwargPlumbing:
    """Pipeline entry points expose `chunk` and forward it correctly."""

    @pytest.mark.parametrize(
        "func", [adj_corr, unit_merge, seeds_merge, initA]
    )
    def test_chunk_default_matches_constant(self, func):
        sig = inspect.signature(func)
        assert "chunk" in sig.parameters
        assert sig.parameters["chunk"].default == 600

    def test_adj_corr_returns_true_pearson(self):
        # Regression for the vsub.T axis bug: adj_corr must match
        # np.corrcoef on the underlying pixel traces and be chunk-invariant.
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

        small = adj_corr(varr, adj, nod_df, freq=None, chunk=4).toarray()
        big = adj_corr(varr, adj, nod_df, freq=None, chunk=64).toarray()
        assert np.allclose(small, big, atol=1e-5)

    def test_chunk_kwarg_is_forwarded_to_graph_optimize_corr(self):
        from unittest.mock import patch
        from .. import cnmf

        rng = np.random.RandomState(0)
        n, t, h, w = 20, 50, 30, 30
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
        adj = radius_neighbors_graph(
            nod_df[["height", "width"]].values, radius=5
        ).astype(bool)

        seen_chunks = []
        original = cnmf.graph_optimize_corr

        def spy(*args, **kwargs):
            seen_chunks.append(kwargs.get("chunk"))
            return original(*args, **kwargs)

        with patch.object(cnmf, "graph_optimize_corr", spy):
            adj_corr(varr, adj, nod_df, freq=None, chunk=20)
            adj_corr(varr, adj, nod_df, freq=None, chunk=5)
        assert seen_chunks == [20, 5]
