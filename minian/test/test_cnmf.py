"""Unit tests for discrete functions in ``minian.cnmf`` on synthetic input."""

from collections.abc import Callable

import dask.array as da
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xarray as xr
from sklearn.neighbors import radius_neighbors_graph

from ..cnmf import (
    adj_corr,
    filt_fft_vec,
    graph_optimize_corr,
    partition_diagnostics,
    spatial_partition,
    unit_merge,
)

# ---------------------------------------------------------------------------
# spatial_partition
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_4x4():
    """16 points at integer coordinates on a 4x4 lattice, row-major.

    Index layout:
        row h=0:  0  1  2  3      (w=0..3)
        row h=1:  4  5  6  7
        row h=2:  8  9 10 11
        row h=3: 12 13 14 15

    With ``target_chunk=4`` the partitioner splits h first (tied extents
    => ``argmax`` returns axis 0), then w inside each half, producing
    the four 2x2 corner tiles. The labels are emitted depth-first
    left-then-right, so the expected mapping is hand-verifiable below.
    """
    h, w = np.divmod(np.arange(16), 4)
    return np.column_stack([h.astype(float), w.astype(float)])


class TestSpatialPartitionBaseCase:
    """Pin the partition output for an explicit hand-verifiable input.

    These tests double as worked examples in the documentation: anyone
    reading the algorithm can re-derive the expected labels and confirm
    the implementation matches.
    """

    def test_grid_4x4_chunk_4_gives_four_corner_tiles(self, grid_4x4):
        # The four expected partitions are the 2x2 corners of the grid:
        # top-left {0,1,4,5}, top-right {2,3,6,7},
        # bottom-left {8,9,12,13}, bottom-right {10,11,14,15}.
        membership = spatial_partition(grid_4x4, target_chunk=4)

        expected_tiles = [
            {0, 1, 4, 5},  # top-left  (h in 0..1, w in 0..1)
            {2, 3, 6, 7},  # top-right (h in 0..1, w in 2..3)
            {8, 9, 12, 13},  # bot-left  (h in 2..3, w in 0..1)
            {10, 11, 14, 15},  # bot-right (h in 2..3, w in 2..3)
        ]
        # Every node assigned to exactly one of the four tiles.
        for tile in expected_tiles:
            label = membership[next(iter(tile))]
            assert set(np.where(membership == label)[0]) == tile

    def test_grid_4x4_left_half_labels_precede_right_half(self, grid_4x4):
        # Depth-first left-then-right emission means lower-h tiles get
        # lower label numbers; the docstring guarantees consecutive
        # labels are spatially adjacent. This is the test that
        # protects that guarantee from a future "shuffle labels"
        # refactor.
        membership = spatial_partition(grid_4x4, target_chunk=4)
        top_label = membership[0]  # tile containing (0,0)
        bot_label = membership[12]  # tile containing (3,0)
        assert top_label < bot_label

    def test_stretched_grid_splits_along_longer_axis(self):
        # 8 points along x, only 1 along y -> first split must be on x.
        # With target_chunk=4 the result is left half {0..3}, right
        # half {4..7}; the y-coordinate is irrelevant.
        positions = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
        membership = spatial_partition(positions, target_chunk=4)
        assert set(membership[:4]) == {membership[0]}
        assert set(membership[4:]) == {membership[4]}
        assert membership[0] != membership[4]


class TestSpatialPartitionTargetChunkSweep:
    """``target_chunk`` is the only knob users touch; pin its semantics.

    On 16 grid points, partition count = 16 / (largest power of 2 <=
    target_chunk that the recursion can land on). The recursion halves
    until size <= target_chunk, so:

        tc >= 16     -> 1 partition  (no split at root)
        8 <= tc < 16 -> 2 partitions (one split)
        4 <= tc < 8  -> 4 partitions
        2 <= tc < 4  -> 8 partitions
        tc == 1      -> 16 partitions
    """

    @pytest.mark.parametrize(
        "target_chunk, expected_n_parts, expected_part_size",
        [
            (16, 1, 16),
            (10, 2, 8),  # 16 > 10 -> split; 8 <= 10 -> stop
            (8, 2, 8),
            (5, 4, 4),
            (4, 4, 4),
            (3, 8, 2),
            (2, 8, 2),
            (1, 16, 1),
        ],
    )
    def test_partition_count_and_size_on_grid(
        self, grid_4x4, target_chunk, expected_n_parts, expected_part_size
    ):
        membership = spatial_partition(grid_4x4, target_chunk=target_chunk)
        counts = np.bincount(membership)
        assert len(counts) == expected_n_parts
        assert (counts == expected_part_size).all()

    @pytest.mark.parametrize("n", (1, 2, 3, 7, 49, 199, 200, 201))
    @pytest.mark.parametrize("chunk", (1, 5, 17, 50, 200, 500))
    def test_max_part_size_never_exceeds_target_chunk(self, n: int, chunk: int):
        # Uniform-grid points, sweeping both the point count -- including odd
        # and prime counts not divisible by 2, where the median split can't
        # halve evenly -- and target_chunk. No partition may exceed the
        # requested size, for any combination.
        rng = np.random.RandomState(0)
        positions = rng.uniform(0, 100, size=(n, 2))
        membership = spatial_partition(positions, target_chunk=chunk)
        assert np.bincount(membership).max() <= chunk, f"n={n} chunk={chunk}"

    def test_partition_count_is_monotone_in_target_chunk(self):
        # Smaller target_chunk -> at least as many partitions. Holds
        # because the recursion can only split, never merge.
        rng = np.random.RandomState(1)
        positions = rng.uniform(0, 100, size=(300, 2))
        n_parts_prev = None
        for tc in (1, 2, 5, 10, 25, 50, 100, 300):
            membership = spatial_partition(positions, target_chunk=tc)
            n_parts = membership.max() + 1
            if n_parts_prev is not None:
                assert n_parts <= n_parts_prev, f"non-monotone at tc={tc}"
            n_parts_prev = n_parts

    def test_labels_are_dense_from_zero(self):
        # Gaps in the label space would silently create empty groupby
        # groups downstream in graph_optimize_corr.
        rng = np.random.RandomState(2)
        positions = rng.uniform(0, 100, size=(200, 2))
        membership = spatial_partition(positions, target_chunk=32)
        unique = np.unique(membership)
        assert (unique == np.arange(len(unique))).all()


class TestSpatialPartitionCompactness:
    """The whole point of this function is to produce *spatially*
    compact partitions -- a random label assignment would also satisfy
    size constraints but cross many more edges in `graph_optimize_corr`.
    """

    def test_partition_diameter_is_bounded(self):
        # On uniform 2D data, each partition should be at most
        # ~sqrt(target_chunk) wide. Compare against a random label
        # baseline: bounding-box diagonals should be much tighter.
        rng = np.random.RandomState(0)
        positions = rng.uniform(0, 100, size=(256, 2))
        membership = spatial_partition(positions, target_chunk=16)

        def max_diameter(labels: np.ndarray) -> float:
            d = 0.0
            for lab in np.unique(labels):
                pts = positions[labels == lab]
                diag = np.linalg.norm(pts.max(0) - pts.min(0))
                d = max(d, diag)
            return d

        random_labels = rng.permutation(len(positions)) % (membership.max() + 1)
        spatial_diam = max_diameter(membership)
        random_diam = max_diameter(random_labels)
        # Spatial partition packs each leaf into a tile of side
        # ~sqrt(16/256)*100 = 25, diagonal ~35. Random labels spread
        # each label across the full 100x100 FOV, diagonal ~140.
        # Order-of-magnitude assertion: spatial should be < half the
        # random baseline (typical ratio ~4x).
        assert spatial_diam < 0.5 * random_diam

    def test_neighbouring_points_share_partition_more_often(self):
        # Construct two clusters, far apart. The partitioner must keep
        # each cluster intact at target_chunk = cluster size.
        cluster_a = np.column_stack([np.zeros(8), np.linspace(0, 1, 8)])
        cluster_b = np.column_stack([np.full(8, 100.0), np.linspace(0, 1, 8)])
        positions = np.vstack([cluster_a, cluster_b])
        membership = spatial_partition(positions, target_chunk=8)
        assert set(membership[:8]) == {membership[0]}
        assert set(membership[8:]) == {membership[8]}
        assert membership[0] != membership[8]


class TestSpatialPartitionEdgeCases:
    def test_single_point(self):
        membership = spatial_partition(np.array([[5.0, 7.0]]), target_chunk=4)
        assert membership.tolist() == [0]

    def test_empty_input(self):
        membership = spatial_partition(np.empty((0, 2)), target_chunk=4)
        assert membership.shape == (0,)

    def test_n_equals_target_chunk(self):
        # Recursion terminates immediately when len(idx) <= target_chunk;
        # this is the boundary where it just-barely doesn't split.
        positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        membership = spatial_partition(positions, target_chunk=4)
        assert membership.tolist() == [0, 0, 0, 0]

    def test_n_exceeds_target_chunk_by_one(self):
        # 5 points, target_chunk=4: one split into 2 + 3.
        positions = np.column_stack([np.arange(5, dtype=float), np.zeros(5)])
        membership = spatial_partition(positions, target_chunk=4)
        counts = np.bincount(membership)
        assert sorted(counts.tolist()) == [2, 3]

    def test_deterministic_across_repeated_calls(self):
        # spatial_partition must be deterministic: the optional visualization
        # step recomputes the partition independently of the compute path, so a
        # method that returned different-but-equally-valid partitions run to run
        # would make the visualization disagree with what was actually computed.
        # Use heavily tied coordinates (many points share a split-axis value) so
        # a non-stable tiebreak -- the most likely way a future change to the
        # method reintroduces nondeterminism -- surfaces here rather than slips
        # through on the ambiguity-free uniform-random case.
        rng = np.random.RandomState(4)
        positions = rng.randint(0, 5, size=(200, 2)).astype(float)  # ~8 dupes/cell
        first = spatial_partition(positions, target_chunk=32)
        for _ in range(5):
            again = spatial_partition(positions.copy(), target_chunk=32)
            assert (again == first).all()


class TestSpatialPartitionContractViolations:
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

    def test_rejects_non_finite_positions(self):
        # NaN/inf rows would silently cluster via argsort's NaN handling,
        # which would corrupt every downstream chunk assignment.
        bad_nan = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan]])
        with pytest.raises(ValueError, match="positions must be finite"):
            spatial_partition(bad_nan, target_chunk=2)
        bad_inf = np.array([[0.0, 0.0], [np.inf, 1.0]])
        with pytest.raises(ValueError, match="positions must be finite"):
            spatial_partition(bad_inf, target_chunk=2)


# ---------------------------------------------------------------------------
# adj_corr + graph_optimize_corr position plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_varr() -> Callable[[int, int, int, int], xr.DataArray]:
    """Factory: build a (frame, height, width) random xr.DataArray.

    Dim order matches what ``minian.utilities.load_videos`` produces and
    what ``apply_transform`` / ``save_minian`` preserve through the
    pipeline. xarray's vectorized ``sel`` is dim-order sensitive (it
    places the new "pixels" dim at the position of the first replaced
    dim), so testing against a fixture in the opposite order would let
    a transpose bug in ``construct_comput`` slip past every assertion
    in this file.
    """

    def _build(height: int, width: int, frames: int, seed: int = 0) -> xr.DataArray:
        rng = np.random.RandomState(seed)
        return xr.DataArray(
            rng.standard_normal((frames, height, width)).astype("float32"),
            dims=("frame", "height", "width"),
            coords={
                "frame": np.arange(frames),
                "height": np.arange(height),
                "width": np.arange(width),
            },
        )

    return _build


@pytest.fixture
def adj_corr_setup(
    synthetic_varr,
) -> Callable[[int, int, int, int, int, int], tuple[xr.DataArray, np.ndarray, pd.DataFrame]]:
    """Factory: ``(varr, adj, nod_df)`` for a radius-neighbour graph on
    randomly placed nodes. Defaults match the smaller-FOV regression
    case; bump ``n``/``t``/``radius`` per test as needed.
    """

    def _build(
        n=80, h=15, w=15, t=300, radius=4, seed=0
    ) -> tuple[xr.DataArray, np.ndarray, pd.DataFrame]:
        rng = np.random.RandomState(seed)
        varr = synthetic_varr(h, w, t, seed=seed + 1)
        hs = rng.randint(0, h, size=n)
        ws = rng.randint(0, w, size=n)
        nod_df = pd.DataFrame({"height": hs, "width": ws})
        adj = radius_neighbors_graph(nod_df[["height", "width"]].values, radius=radius).astype(bool)
        return varr, adj, nod_df

    return _build


class TestAdjCorr:
    """End-to-end behaviour of :func:`adj_corr`.

    ``test_returns_true_pearson`` pins the output against ``np.corrcoef``
    on the underlying traces (and, with ``freq``, against the same after
    :func:`filt_fft_vec` lowpass smoothing). The chunk parametrization
    in that test also covers chunk-invariance directly: matching
    np.corrcoef at chunk=4 AND chunk=64 makes it impossible for the two
    to disagree with each other. The other tests in this class are
    surface properties (shape, partition invariance) that only become
    meaningful because that test anchors the values.
    """

    @pytest.mark.parametrize("chunk", [4, 64])
    @pytest.mark.parametrize("freq", [None, 0.05])
    def test_returns_true_pearson(self, freq, chunk, adj_corr_setup):
        # Regression for the vsub.T axis bug (issue #302): adj_corr must
        # match np.corrcoef on the underlying pixel traces. Parametrized
        # on freq because the unsmoothed and smoothed branches of
        # construct_comput go through different gufunc paths -- both had
        # the bug. Parametrized on chunk because that bug made idx_corr
        # compute correlations between frame slices across the chunk's
        # pixel subset, so its output depended on which other pixels
        # happened to share the chunk -- matching np.corrcoef at
        # multiple chunk sizes is a direct fingerprint of the fix.
        varr, adj, nod_df = adj_corr_setup(n=16, t=80, h=25, w=25, radius=8, seed=11)
        hs = nod_df["height"].values
        ws = nod_df["width"].values
        out = adj_corr(varr, adj, nod_df, freq=freq, chunk=chunk).toarray()

        traces = np.stack(
            [varr.isel(height=hs[k], width=ws[k]).values for k in range(len(nod_df))]
        ).astype("float32")
        # filt_fft_vec mutates in place; copy to keep `traces` clean for
        # later inspection if this assertion fails.
        traces_truth = filt_fft_vec(traces.copy(), freq, "low") if freq is not None else traces
        for i, j in zip(*out.nonzero()):
            truth = np.corrcoef(traces_truth[i], traces_truth[j])[0, 1]
            assert out[i, j] == pytest.approx(truth, abs=1e-5)

    def test_correlations_invariant_to_partition_choice(self, adj_corr_setup):
        # Partitioning only controls how the correlation work is chunked,
        # never the values. A small `chunk` forces the k-d tree to split into
        # several partitions; `chunk >= n` keeps everyone in one partition.
        # Every (i, j) correlation must be identical across the two. Anchored
        # against np.corrcoef by test_returns_true_pearson.
        varr, adj, nod_df = adj_corr_setup(n=80, seed=7)
        multi = adj_corr(varr, adj, nod_df, freq=None, chunk=8).toarray()
        single = adj_corr(varr, adj, nod_df, freq=None, chunk=1000).toarray()
        assert np.allclose(multi, single)

    def test_output_shape_matches_input_adj(self, adj_corr_setup):
        # One entry per undirected edge; callers mirror via `adj + adj.T`.
        varr, adj, nod_df = adj_corr_setup(n=40)
        out = adj_corr(varr, adj, nod_df, freq=None)
        assert isinstance(out, scipy.sparse.csr_matrix)
        assert out.shape == adj.shape
        out_pairs = set(zip(*out.nonzero()))
        adj_pairs = set(zip(*adj.nonzero()))
        assert out_pairs.issubset(adj_pairs)
        assert out.nnz == adj.nnz // 2


# ---------------------------------------------------------------------------
# graph_optimize_corr: contract violations should fail loudly
# ---------------------------------------------------------------------------


@pytest.fixture
def line_graph_with_pos_attrs():
    """Factory: n-node line graph with optional ``height``/``width`` node attrs."""

    def _build(n: int, with_pos_attrs: bool = True) -> nx.Graph:
        G = nx.Graph()
        for i in range(n):
            attrs = {"height": float(i), "width": float(i)} if with_pos_attrs else {}
            G.add_node(i, **attrs)
        for i in range(n - 1):
            G.add_edge(i, i + 1)
        return G

    return _build


@pytest.fixture
def tiny_varr(synthetic_varr):
    """32x32x50 random varr; cheap enough to use in every contract test."""
    return synthetic_varr(32, 32, 50, seed=0)


class TestGraphOptimizeCorr:
    """Functional behaviour of the chunked-correlation primitive.

    ``test_edge_correlations_match_corrcoef`` pins each returned edge
    correlation against ``np.corrcoef`` on the underlying pixel traces.
    The serializability test is a side-effect contract downstream code
    (notebook checkpoints, JSON inspection) depends on.
    """

    @pytest.mark.parametrize("freq", [None, 0.05])
    def test_edge_correlations_match_corrcoef(self, freq, synthetic_varr):
        # Build a small graph + adjacency directly (no adj_corr wrapper).
        # Each edge's returned corr must equal np.corrcoef on the two
        # pixel traces (with lowpass smoothing applied first when freq).
        rng = np.random.RandomState(3)
        n, h, w, t = 12, 20, 20, 80
        varr = synthetic_varr(h, w, t, seed=1)
        hs = rng.randint(0, h, size=n)
        ws = rng.randint(0, w, size=n)

        G = nx.Graph()
        G.add_nodes_from([(i, {"height": int(hs[i]), "width": int(ws[i])}) for i in range(n)])
        # Sparse edges so we have a few intra- and cross-partition pairs.
        for src, tgt in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 6), (3, 9), (5, 11)]:
            G.add_edge(src, tgt)

        corr_df = graph_optimize_corr(varr, G, freq, chunk=4)

        traces = np.stack([varr.isel(height=hs[k], width=ws[k]).values for k in range(n)]).astype(
            "float32"
        )
        if freq is not None:
            traces = filt_fft_vec(traces.copy(), freq, "low")

        for _, row in corr_df.iterrows():
            i, j = int(row["source"]), int(row["target"])
            truth = np.corrcoef(traces[i], traces[j])[0, 1]
            assert row["corr"] == pytest.approx(truth, abs=1e-5)

    def test_rejects_nodes_without_position_attrs(self, tiny_varr, line_graph_with_pos_attrs):
        # graph_optimize_corr derives spatial positions from each node's
        # height/width attributes. A node missing them is a caller bug that
        # would otherwise corrupt every partition assignment, so it must
        # fail loudly rather than silently mis-chunk the correlation work.
        G = line_graph_with_pos_attrs(4, with_pos_attrs=False)
        with pytest.raises(ValueError, match="must carry 'height' and 'width'"):
            graph_optimize_corr(tiny_varr, G, freq=None, chunk=2)


# ---------------------------------------------------------------------------
# unit_merge
# ---------------------------------------------------------------------------


class TestUnitMerge:
    """Contract violations that should fail loudly rather than silently
    corrupt downstream centroid computation."""

    def test_rejects_zero_mass_footprints(self):
        # The centroid formula `mom / mass` produces NaN when mass == 0,
        # which would in turn make spatial_partition reject the centroids
        # at a confusing call site downstream. unit_merge catches the bad
        # input at the source and names the offending unit_id.
        A_data = np.zeros((2, 4, 4), dtype=float)
        A_data[0, 1, 1] = 1.0  # unit 10 has mass; unit 20 is all-zero
        A = xr.DataArray(
            da.from_array(A_data, chunks=(1, 4, 4)),
            dims=("unit_id", "height", "width"),
            coords={
                "unit_id": [10, 20],
                "height": np.arange(4),
                "width": np.arange(4),
            },
        )
        # C is unused before the raise; pass a dummy with matching unit_id.
        C = xr.DataArray(
            np.zeros((2, 5), dtype=float),
            dims=("unit_id", "frame"),
            coords={"unit_id": [10, 20], "frame": np.arange(5)},
        )
        with pytest.raises(ValueError, match="zero-mass footprints.*unit_id=\\[20\\]"):
            unit_merge(A, C)


# ---------------------------------------------------------------------------
# partition_diagnostics
# ---------------------------------------------------------------------------


@pytest.fixture
def line_graph_membership():
    """Factory: ``(positions, membership, adj)`` for n nodes on a 1D line.

    Stable argsort pairs consecutive nodes, so partition labels follow
    node index in stride ``chunk``; radius=1.5 -> exactly the
    consecutive-pair edges.
    """

    def _build(n: int, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        membership = spatial_partition(positions, target_chunk=chunk)
        adj = radius_neighbors_graph(positions, radius=1.5).astype(bool)
        return positions, membership, adj

    return _build


class TestPartitionDiagnostics:
    """Verifies partition_diagnostics returns correct sizes, edge counts,
    and memory estimates for hand-built inputs with known expected values."""

    def test_sizes_and_n_parts_without_adj(self, line_graph_membership):
        # Protects the conditional-key contract: callers (visualize_spatial_
        # partition uses `if adj is not None` before reading edge keys) rely
        # on edge/memory keys being absent when their inputs aren't supplied.
        # If the function always populated those keys with None or 0, downstream
        # presence checks would silently misrender.
        _, membership, _ = line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership)
        assert diag["n_parts"] == 4
        assert diag["sizes"].tolist() == [2, 2, 2, 2]
        for absent in (
            "edges_per_partition",
            "cross_fraction",
            "cross_edges",
            "total_edges",
            "mem_mb",
        ):
            assert absent not in diag

    def test_edge_counts_on_line_graph(self, line_graph_membership):
        # Protects the core counting math: that intra-partition edges are
        # tallied per partition correctly and cross-partition edges aren't
        # double-counted. 8 nodes on a line, radius=1.5 -> 7 consecutive-pair
        # edges; chunk=2 pairs them into (0,1) (2,3) (4,5) (6,7) -> 4 intra,
        # 3 cross. A bug that compared partition labels with the wrong indices
        # (or counted both directions of an edge) would change these numbers.
        _, membership, adj = line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership, adj=adj)
        assert diag["total_edges"] == 7
        assert diag["edges_per_partition"].tolist() == [1, 1, 1, 1]
        assert diag["cross_edges"] == 3
        assert diag["cross_fraction"] == pytest.approx(3 / 7)

    def test_symmetric_and_triangular_adj_give_same_result(self, line_graph_membership):
        # Callers pass either symmetric (radius_neighbors_graph default) or
        # triangular (sparse.tril, used by unit_merge for A_inter) adjacency.
        # _canonicalize_edge_pairs collapses both to unique (i<j) pairs; without
        # that, symmetric adj would double cross_edges and edges_per_partition.
        _, membership, adj = line_graph_membership(10, chunk=2)
        diag_sym = partition_diagnostics(membership, adj=adj)
        diag_tri = partition_diagnostics(membership, adj=scipy.sparse.tril(adj, k=-1))
        for key in ("total_edges", "cross_edges", "cross_fraction"):
            assert diag_sym[key] == diag_tri[key]
        assert (diag_sym["edges_per_partition"] == diag_tri["edges_per_partition"]).all()

    def test_self_loops_are_dropped(self):
        # Some sparse-graph constructors (e.g. radius_neighbors_graph with
        # include_self=True) include i==i entries. Counting those would
        # inflate total_edges and put spurious entries into
        # edges_per_partition; this test pins the filter that drops them.
        membership = np.array([0, 0, 1, 1])
        adj = scipy.sparse.csr_matrix(np.eye(4))  # only self-loops
        diag = partition_diagnostics(membership, adj=adj)
        assert diag["total_edges"] == 0
        assert diag["cross_fraction"] == 0.0
        assert diag["edges_per_partition"].tolist() == [0, 0]

    def test_memory_estimate(self, line_graph_membership):
        # Pins the byte-to-MiB unit math (bytes_per_sample * n_frames * size
        # / 1024**2). A typo like `/ 1024` or `* 1024**2` would silently
        # misreport the partition memory cost, which drives target_chunk
        # tuning in the notebook preview.
        _, membership, _ = line_graph_membership(8, chunk=2)
        diag = partition_diagnostics(membership, n_frames=1000)
        expected_mib = 2 * 1000 * 4 / (1024**2)
        assert diag["mem_mb"].tolist() == pytest.approx([expected_mib] * 4)

    def test_empty_membership_returns_empty_diag(self):
        # Edge case: bincount on an empty array and iterating zero edge
        # pairs must not raise. visualize_spatial_partition can hit this
        # when seed-finding returns no seeds.
        diag = partition_diagnostics(
            np.empty(0, dtype=int),
            adj=scipy.sparse.csr_matrix((0, 0)),
            n_frames=100,
        )
        assert diag["n_parts"] == 0
        assert diag["sizes"].shape == (0,)
        assert diag["total_edges"] == 0
        assert diag["mem_mb"].shape == (0,)
