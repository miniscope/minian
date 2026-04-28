"""Chunk inspection, heuristic chunk plans, optimized rechunking."""

import functools as fct
from typing import List, Optional

import dask as da
import dask.array as darr
import numpy as np
import xarray as xr

from .dask_graph import FAST_FUNCTIONS, custom_arr_optimize


def get_chk(arr: xr.DataArray) -> dict:
    """
    Get chunks of a `xr.DataArray`.

    Parameters
    ----------
    arr : xr.DataArray
        The input `xr.DataArray`

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunks.
    """
    return {d: c for d, c in zip(arr.dims, arr.chunks)}


def rechunk_like(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    """
    Rechunk the input `x` such that its chunks are compatible with `y`.

    Parameters
    ----------
    x : xr.DataArray
        The array to be rechunked.
    y : xr.DataArray
        The array where chunk information are extracted.

    Returns
    -------
    x_chk : xr.DataArray
        The rechunked `x`.
    """
    try:
        dst_chk = get_chk(y)
        comm_dim = set(x.dims).intersection(set(dst_chk.keys()))
        dst_chk = {d: max(dst_chk[d]) for d in comm_dim}
        return x.chunk(dst_chk)
    except TypeError:
        return x.compute()


def get_optimal_chk(
    arr: xr.DataArray,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype: Optional[type] = None,
) -> dict:
    """
    Compute the optimal chunk size across all dimensions of the input array.

    This function use `dask` autochunking mechanism to determine the optimal
    chunk size of an array. The difference between this and directly using
    "auto" as chunksize is that it understands which dimensions are usually
    chunked together with the help of `dim_grp`. It also support computing
    chunks for custom `dtype` and explicit requirement of chunk size.

    Parameters
    ----------
    arr : xr.DataArray
        The input array to estimate for chunk size.
    dim_grp : list, optional
        List of tuples specifying which dimensions are usually chunked together
        during computation. For each tuple in the list, it is assumed that only
        dimensions in the tuple will be chunked while all other dimensions in
        the input `arr` will not be chunked. Each dimensions in the input `arr`
        should appear once and only once across the list. By default
        `[("frame",), ("height", "width")]`.
    csize : int, optional
        The desired space each chunk should occupy, specified in MB. By default
        `256`.
    dtype : type, optional
        The datatype of `arr` during actual computation in case that will be
        different from the current `arr.dtype`. By default `None`.

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunk sizes.
    """
    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    chk_compute = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        chk = get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})
    with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = get_chunksize(arr_chk)
    chk_store = dict()
    for d in dims:
        ncomp = int(arr.sizes[d] / chk_compute[d])
        sz = np.array(factors(ncomp)) * chk_compute[d]
        chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
    return chk_compute, chk_store_da


def get_chunksize(arr: xr.DataArray) -> dict:
    """
    Get chunk size of a `xr.DataArray`.

    Parameters
    ----------
    arr : xr.DataArray
        The input `xr.DataArray`.

    Returns
    -------
    chk : dict
        Dictionary mapping dimension names to chunk sizes.
    """
    dims = arr.dims
    sz = arr.data.chunksize
    return {d: s for d, s in zip(dims, sz)}


def factors(x: int) -> List[int]:
    """
    Compute all factors of an interger.

    Parameters
    ----------
    x : int
        Input

    Returns
    -------
    factors : List[int]
        List of factors of `x`.
    """
    return [i for i in range(1, x + 1) if x % i == 0]


def optimize_chunk(arr: xr.DataArray, chk: dict) -> xr.DataArray:
    """
    Rechunk a `xr.DataArray` with constrained "rechunk-merge" tasks.

    Parameters
    ----------
    arr : xr.DataArray
        The array to be rechunked.
    chk : dict
        The desired chunk size.

    Returns
    -------
    arr_chk : xr.DataArray
        The rechunked array.
    """
    fast_funcs = FAST_FUNCTIONS + [darr.core.concatenate3]
    arr_chk = arr.chunk(chk)
    arr_opt = fct.partial(
        custom_arr_optimize,
        fast_funcs=fast_funcs,
        rewrite_dict={"rechunk-merge": "merge_restricted"},
    )
    with da.config.set(array_optimize=arr_opt):
        arr_chk.data = da.optimize(arr_chk.data)[0]
    return arr_chk


