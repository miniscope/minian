"""CNMF decomposition and helpers (combined module).
"""
import logging
from typing import List

import dask as da
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse
import xarray as xr




from .filters import filt_fft_vec

log = logging.getLogger(__name__)


def label_connected(adj: np.ndarray, only_connected=False) -> np.ndarray:
    """
    Label connected components given adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix. Should be 2d symmetric matrix.
    only_connected : bool, optional
        Whether to keep only the labels of connected components. If `True`, then
        all components with only one node (isolated) will have their labels set
        to -1. Otherwise all components will have unique label. By default
        `False`.

    Returns
    -------
    labels : np.ndarray
        The labels for each components. Should have length `adj.shape[0]`.
    """
    try:
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj)
        g = nx.convert_matrix.from_numpy_matrix(adj)
    except Exception:
        g = nx.convert_matrix.from_scipy_sparse_matrix(adj)
    labels = np.zeros(adj.shape[0], dtype=np.int)
    for icomp, comp in enumerate(nx.connected_components(g)):
        comp = list(comp)
        if only_connected and len(comp) == 1:
            labels[comp] = -1
        else:
            labels[comp] = icomp
    return labels

def graph_optimize_corr(
    varr: xr.DataArray,
    G: nx.Graph,
    freq: float,
    idx_dims=["height", "width"],
    chunk=600,
    step_size=50,
) -> pd.DataFrame:
    """
    Compute correlation in an optimized fashion given a computation graph.

    This function carry out out-of-core computation of large correaltion matrix.
    It takes in a computaion graph whose node represent timeseries and edges
    represent the desired pairwise correlation to be computed. The actual
    timeseries are stored in `varr` and indexed with node attributes. The
    function can carry out smoothing of timeseries before computation of
    correlation. To minimize re-computation of smoothing for each pixel, the
    graph is first partitioned using a minial-cut algorithm. Then the
    computation is performed in chunks with size `chunk`, with nodes from the
    same partition being in the same chunk as much as possible.

    Parameters
    ----------
    varr : xr.DataArray
        Input timeseries. Should have "frame" dimension in addition to those
        specified in `idx_dims`.
    G : nx.Graph
        Graph representing computation to be carried out. Should be undirected
        and un-weighted. Each node should have unique attributes with keys
        specified in `idx_dims`, which will be used to index the timeseries in
        `varr`. Each edge represent a desired correlation.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.
    idx_dims : list, optional
        The dimension used to index the timeseries in `varr`. By default
        `["height", "width"]`.
    chunk : int, optional
        Chunk size of each computation. By default `600`.
    step_size : int, optional
        Step size to iterate through all edges. If too small then the iteration
        will take a long time. If too large then the variances in the actual
        chunksize of computation will be large. By default `50`.

    Returns
    -------
    eg_df : pd.DataFrame
        Dataframe representation of edge list. Has column "source" and "target"
        representing the node index of the edge (correlation), and column "corr"
        with computed value of correlation.
    """
    # a heuristic to make number of partitions scale with nodes
    n_cuts, membership = pymetis.part_graph(
        max(int(np.ceil(G.number_of_nodes() / chunk)), 1), adjacency=adj_list(G)
    )
    nx.set_node_attributes(
        G, {k: {"part": v} for k, v in zip(sorted(G.nodes), membership)}
    )
    eg_df = nx.to_pandas_edgelist(G)
    part_map = nx.get_node_attributes(G, "part")
    eg_df["part_src"] = eg_df["source"].map(part_map)
    eg_df["part_tgt"] = eg_df["target"].map(part_map)
    eg_df["part_diff"] = (eg_df["part_src"] - eg_df["part_tgt"]).astype(bool)
    corr_ls = []
    idx_ls = []
    npxs = []
    egd_same, egd_diff = eg_df[~eg_df["part_diff"]], eg_df[eg_df["part_diff"]]
    idx_dict = {d: nx.get_node_attributes(G, d) for d in idx_dims}

    def construct_comput(edf, pxs):
        px_map = {k: v for v, k in enumerate(pxs)}
        ridx = edf["source"].map(px_map).values
        cidx = edf["target"].map(px_map).values
        idx_arr = {
            d: xr.DataArray([dd[p] for p in pxs], dims="pixels")
            for d, dd in idx_dict.items()
        }
        vsub = varr.sel(**idx_arr).data
        if len(idx_arr) > 1:  # vectorized indexing
            vsub = vsub.T
        else:
            vsub = vsub.rechunk(-1)
        with da.config.set(**{"optimization.fuse.ave-width": vsub.shape[0]}):
            return da.optimize(smooth_corr(vsub, ridx, cidx, freq=freq))[0]

    for _, eg_sub in egd_same.groupby("part_src"):
        pixels = list(set(eg_sub["source"]) | set(eg_sub["target"]))
        corr_ls.append(construct_comput(eg_sub, pixels))
        idx_ls.append(eg_sub.index)
        npxs.append(len(pixels))
    pixels = set()
    eg_ls = []
    grp = np.arange(len(egd_diff)) // step_size
    for igrp, eg_sub in egd_diff.sort_values("source").groupby(grp):
        pixels = pixels | set(eg_sub["source"]) | set(eg_sub["target"])
        eg_ls.append(eg_sub)
        if (len(pixels) > chunk - step_size / 2) or igrp == max(grp):
            pixels = list(pixels)
            edf = pd.concat(eg_ls)
            corr_ls.append(construct_comput(edf, pixels))
            idx_ls.append(edf.index)
            npxs.append(len(pixels))
            pixels = set()
            eg_ls = []
    log.info(
        "pixel recompute ratio: {}".format(sum(npxs) / G.number_of_nodes())
    )
    log.info("computing correlations")
    corr_ls = da.compute(corr_ls)[0]
    corr = pd.Series(np.concatenate(corr_ls), index=np.concatenate(idx_ls), name="corr")
    eg_df["corr"] = corr
    return eg_df

def adj_corr(
    varr: xr.DataArray, adj: np.ndarray, nod_df: pd.DataFrame, freq: float
) -> scipy.sparse.csr_matrix:
    """
    Compute correlation in an optimized fashion given an adjacency matrix and
    node attributes.

    Wraps around :func:`graph_optimize_corr` and construct computation graph
    from `adj` and `nod_df`. Also convert the result into a sparse matrix with
    same shape as `adj`.

    Parameters
    ----------
    varr : xr.DataArray
        Input time series. Should have "frame" dimension in addition to column
        names of `nod_df`.
    adj : np.ndarray
        Adjacency matrix.
    nod_df : pd.DataFrame
        Dataframe containing node attributes. Should have length `adj.shape[0]`
        and only contain columns relevant to index the time series.
    freq : float
        Cut-off frequency for the optional smoothing. If `None` then no
        smoothing will be done.

    Returns
    -------
    adj_corr : scipy.sparse.csr_matrix
        Sparse matrix of the same shape as `adj` but with values corresponding
        the computed correlation.
    """
    G = nx.Graph()
    G.add_nodes_from([(i, d) for i, d in enumerate(nod_df.to_dict("records"))])
    G.add_edges_from([(s, t) for s, t in zip(*adj.nonzero())])
    corr_df = graph_optimize_corr(varr, G, freq, idx_dims=nod_df.columns)
    return scipy.sparse.csr_matrix(
        (corr_df["corr"], (corr_df["source"], corr_df["target"])), shape=adj.shape
    )

def adj_list(G: nx.Graph) -> List[np.ndarray]:
    """
    Generate adjacency list representation from graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    adj_ls : List[np.ndarray]
        The adjacency list representation of graph.
    """
    gdict = nx.to_dict_of_dicts(G)
    return [np.array(list(gdict[k].keys())) for k in sorted(gdict.keys())]

def smooth_corr(
    X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray, freq: float
) -> np.ndarray:
    """
    Wraps around :func:`filt_fft_vec` and :func:`idx_corr` to carry out both
    smoothing and computation of partial correlation.

    Parameters
    ----------
    X : np.ndarray
        Input time series.
    ridx : np.ndarray
        Row index of the resulting correlation.
    cidx : np.ndarray
        Column index of the resulting correlation.
    freq : float
        Cut-off frequency for the smoothing.

    Returns
    -------
    corr : np.ndarray
        Resulting partial correlation.
    """
    if freq:
        X = filt_fft_vec(X, freq, "low")
    return idx_corr(X, ridx, cidx)

def idx_corr(X: np.ndarray, ridx: np.ndarray, cidx: np.ndarray) -> np.ndarray:
    """
    Compute partial pairwise correlation based on index.

    This function compute a subset of a pairwise correlation matrix. The
    correlation to be computed are specified by two vectors `ridx` and `cidx` of
    same length, representing the row and column index of the full correlation
    matrix. The function use them to index the timeseries matrix `X` and compute
    only the requested correlations. The result is returned flattened.

    Parameters
    ----------
    X : np.ndarray
        Input time series. Should have 2 dimensions, where the last dimension
        should be the time dimension.
    ridx : np.ndarray
        Row index of the correlation.
    cidx : np.ndarray
        Column index of the correlation.

    Returns
    -------
    res : np.ndarray
        Flattened resulting correlations. Has same shape as `ridx` or `cidx`.
    """
    res = np.zeros(ridx.shape[0])
    std = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
        std[i] = np.sqrt((X[i, :] ** 2).sum())
    for i, (r, c) in enumerate(zip(ridx, cidx)):
        cur_std = std[r] * std[c]
        if cur_std > 0:
            res[i] = (X[r, :] * X[c, :]).sum() / cur_std
        else:
            res[i] = 0
    return res
