"""CNMF decomposition and helpers (combined module)."""

import logging
import os
from typing import Optional, Tuple, Union

import cv2
import dask as da
import dask.array as darr
import numpy as np
import sparse
import xarray as xr
import zarr
from skimage import morphology as moph
from sklearn.linear_model import LassoLars

from ..utilities import (
    rechunk_like,
    save_minian,
)

log = logging.getLogger(__name__)


def update_spatial(
    Y: xr.DataArray,
    A: xr.DataArray,
    C: xr.DataArray,
    sn: xr.DataArray,
    b: xr.DataArray = None,
    f: xr.DataArray = None,
    dl_wnd=5,
    sparse_penal=0.5,
    update_background=False,
    normalize=True,
    size_thres=(9, None),
    in_memory=False,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Update spatial components given the input data and temporal dynamic for each
    cell.

    This function carries out spatial update of the CNMF algorithm. The update
    is done in parallel and independently for each pixel. To save computation
    time, we compute a subsetting matrix `sub` by dilating the initial
    spatial foorprint of each cell. The window size of the dilation is
    controled by `dl_wnd`. Then for each pixel, only cells that have a non-zero
    value in `sub` at the current pixel will be considered for update.
    Optionally, the spatial footprint of the background can be updated in the
    same fashion based on the temporal dynamic of the background. After the
    update, the spatial footprint of each cell can be optionally noramlized to
    unit sum, so that difference in fluorescent intensity will not be reflected
    in spatial footprint. A `size_thres` can be passed in to filter out cells
    whose size (number of non-zero values in spatial footprint) is outside the
    specified range. Finally, the temporal dynamic of cells `C` can either be
    load in memory before the update or lazy-loaded during the update. Note that
    if `in_memory` is `False`, then `C` must be stored under the intermediate
    folder specified as environment variable `MINIAN_INTERMEDIATE`.

    Parameters
    ----------
    Y : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    A : xr.DataArray
        Previous estimation of spatial footprints. Should have dimension
        "height", "width" and "unit_id".
    C : xr.DataArray
        Estimation of temporal component for each cell. Should have dimension
        "frame" and "unit_id".
    sn : xr.DataArray
        Estimation of noise level for each pixel. Should have dimension "height"
        and "width".
    b : xr.DataArray, optional
        Previous estimation of spatial footprint of background. Fhould have
        dimension "height" and "width".
    f : xr.DataArray, optional
        Estimation of temporal dynamic of background. Should have dimension
        "frame".
    dl_wnd : int, optional
        Window of morphological dilation in pixel when computing the subsetting
        matrix. By default `5`.
    sparse_penal : float, optional
        Global scalar controlling sparsity of the result. The higher the value,
        the sparser the spatial footprints. By default `0.5`.
    update_background : bool, optional
        Whether to update the spatial footprint of background. If `True`, then
        both `b` and `f` need to be provided. By default `False`.
    normalize : bool, optional
        Whether to normalize resulting spatial footprints of each cell to unit
        sum. By default `True`
    size_thres : tuple, optional
        The range of size in pixel allowed for the resulting spatial footprints.
        If `None`, then no filtering will be done. By default `(9, None)`.
    in_memory : bool, optional
        Whether to load `C` into memory before spatial update. By default
        `False`.

    Returns
    -------
    A_new : xr.DataArray
        New estimation of spatial footprints. Same shape as `A` except the
        "unit_id" dimension might be smaller due to filtering.
    mask : xr.DataArray
        Boolean mask of whether a cell passed size filtering. Has dimension
        "unit_id" that is same as input `A`. Useful for subsetting other
        variables based on the result of spatial update.
    b_new : xr.DataArray
        New estimation of spatial footprint of background. Only returned if
        `update_background` is `True`. Same shape as `b`.
    norm_fac : xr.DataArray
        Normalizing factor. Userful to scale temporal activity of cells. Only
        returned if `normalize` is `True`.

    Notes
    -----
    For each pixel, the solver fits non-negative temporal traces against cells
    selected by ``sub``, with sparsity controlled by noise times ``sparse_penal``
    (matching the original NNLS / Lasso form in the citation). Larger
    ``sparse_penal`` yields sparser footprints.
    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if in_memory:
        C_store = C.compute().values
    else:
        C_path = os.path.join(intpath, C.name + ".zarr", C.name)
        C_store = zarr.open_array(C_path)
    log.info("estimating penalty parameter")
    alpha = sparse_penal * sn
    alpha = rechunk_like(alpha.compute(), Y)
    log.info("computing subsetting matrix")
    selem = moph.disk(dl_wnd)
    sub = xr.apply_ufunc(
        cv2.dilate,
        A,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        kwargs=dict(kernel=selem),
        dask="parallelized",
        output_dtypes=[A.dtype],
    )
    sub = sub > 0
    sub.data = sub.data.map_blocks(sparse.COO)
    if update_background:
        assert b is not None, "`b` must be provided when updating background"
        assert f is not None, "`f` must be provided when updating background"
        b_in = rechunk_like(b > 0, Y).assign_coords(unit_id=-1).expand_dims("unit_id")
        b_in.data = b_in.data.map_blocks(sparse.COO)
        b_in = b_in.compute()
        sub = xr.concat([sub, b_in], "unit_id")
        f_in = f.compute().data
    else:
        f_in = None
    sub = rechunk_like(sub.transpose("height", "width", "unit_id").compute(), Y)
    log.info("fitting spatial matrix")
    ssub = darr.map_blocks(
        sps_any,
        sub.data,
        drop_axis=2,
        chunks=((1, 1)),
        meta=sparse.ones(1).astype(bool),
    ).compute()
    Y_trans = Y.transpose("height", "width", "frame")
    # take fast route if a lot of chunks are empty
    if ssub.sum() < 500:
        A_new = np.empty(sub.data.numblocks, dtype=object)
        for (hblk, wblk), has_unit in np.ndenumerate(ssub):
            cur_sub = sub.data.blocks[hblk, wblk, :]
            if has_unit:
                cur_blk = update_spatial_block(
                    Y_trans.data.blocks[hblk, wblk, :],
                    alpha.data.blocks[hblk, wblk],
                    cur_sub,
                    C_store=C_store,
                    f=f_in,
                )
            else:
                cur_blk = darr.array(sparse.zeros((cur_sub.shape)))
            A_new[hblk, wblk, 0] = cur_blk
        A_new = darr.block(A_new.tolist())
    else:
        A_new = update_spatial_block(
            Y_trans.data,
            alpha.data,
            sub.data,
            C_store=C_store,
            f=f_in,
        )
    with da.config.set(**{"optimization.fuse.ave-width": 6}):
        A_new = da.optimize(A_new)[0]
    A_new = xr.DataArray(
        darr.moveaxis(A_new, -1, 0).map_blocks(lambda a: a.todense(), dtype=A.dtype),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": sub.coords["unit_id"],
            "height": A.coords["height"],
            "width": A.coords["width"],
        },
    )
    A_new = save_minian(
        A_new.rename("A_new"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    add_rets = []
    if update_background:
        b_new = A_new.sel(unit_id=-1).compute()
        A_new = A_new[:-1, :, :]
        add_rets.append(b_new)
    if size_thres:
        low, high = size_thres
        A_bin = A_new > 0
        mask = np.ones(A_new.sizes["unit_id"], dtype=bool)
        if low:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) > low).compute(), mask
            )
        if high:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) < high).compute(), mask
            )
        mask = xr.DataArray(
            mask, dims=["unit_id"], coords={"unit_id": A_new.coords["unit_id"].values}
        )
    else:
        mask = (A_new.sum(["height", "width"]) > 0).compute()
    log.info(
        "{} out of {} units dropped".format(len(mask) - mask.sum().values, len(mask))
    )
    A_new = A_new.sel(unit_id=mask)
    if normalize:
        norm_fac = A_new.max(["height", "width"]).compute()
        A_new = A_new / norm_fac
        add_rets.append(norm_fac)
    return (A_new, mask, *add_rets)


def sps_any(x: sparse.COO) -> np.ndarray:
    """
    Compute `any` on a sparse array.

    Parameters
    ----------
    x : sparse.COO
        Input sparse array.

    Returns
    -------
    x_any : np.ndarray
        2d boolean numpy array.
    """
    return np.atleast_2d(x.nnz > 0)


def update_spatial_perpx(
    y: np.ndarray,
    alpha: float,
    sub: sparse.COO,
    C_store: Union[np.ndarray, zarr.core.Array],
    f: Optional[np.ndarray],
) -> sparse.COO:
    """
    Update spatial footprints across all the cells for a single pixel.

    This function use :class:`sklearn.linear_model.LassoLars` to solve the
    optimization problem. `C_store` can either be a in-memory numpy array, or a
    zarr array, in which case it will be lazy-loaded. If `f` is not `None`, then
    `sub[-1]` is expected to be the subsetting mask for background, and the last
    element of the return value will be the spatial footprint of background.

    Parameters
    ----------
    y : np.ndarray
        Input fluorescent trace for the given pixel.
    alpha : float
        Parameter of the optimization problem controlling sparsity.
    sub : sparse.COO
        Subsetting matrix.
    C_store : Union[np.ndarray, zarr.core.Array]
        Estimation of temporal dynamics of cells.
    f : np.ndarray, optional
        Temporal dynamic of background.

    Returns
    -------
    A_px : sparse.COO
        Spatial footprint values across all cells for the given pixel.

    See Also
    -------
    update_spatial : for more explanation of parameters
    """
    if f is not None:
        idx = sub[:-1].nonzero()[0]
    else:
        idx = sub.nonzero()[0]
    try:
        C = C_store.get_orthogonal_selection((idx, slice(None))).T
    except AttributeError:
        C = C_store[idx, :].T
    if (f is not None) and sub[-1]:
        C = np.concatenate([C, f.reshape((-1, 1))], axis=1)
        idx = np.concatenate([idx, np.array(len(sub) - 1).reshape(-1)])
    clf = LassoLars(alpha=alpha, positive=True)
    coef = clf.fit(C, y).coef_
    mask = coef > 0
    coef = coef[mask]
    idx = idx[mask]
    return sparse.COO(coords=idx, data=coef, shape=sub.shape)


def update_spatial_block(
    y: np.ndarray, alpha: np.ndarray, sub: sparse.COO, **kwargs
) -> sparse.COO:
    """
    Carry out spatial update for each 3d block of data.

    This function wraps around :func:`update_spatial_perpx` so that it can be
    applied to 3d blocks of data. Keyword arguments are passed to
    :func:`update_spatial_perpx`.

    Parameters
    ----------
    y : np.ndarray
        Input data, should have dimension (height, width, frame).
    alpha : np.ndarray
        Alpha parameter for the optimization problem. Should have dimension
        (height, width).
    sub : sparse.COO
        Subsetting matrix. Should have dimension (height, width, unit_id).

    Returns
    -------
    A_blk : sparse.COO
        Resulting spatial footprints. Should have dimension (height, width,
        unit_id).

    See Also
    -------
    update_spatial_perpx
    update_spatial
    """
    C_store = kwargs.get("C_store")
    f = kwargs.get("f")
    if isinstance(y, darr.Array):
        y = y.compute()
    if isinstance(alpha, darr.Array):
        alpha = alpha.compute()
    if isinstance(sub, darr.Array):
        sub = sub.compute()
    crd_ls = []
    data_ls = []
    for h, w in zip(*sub.any(axis=-1).nonzero()):
        res = update_spatial_perpx(y[h, w, :], alpha[h, w], sub[h, w, :], C_store, f)
        crd = res.coords
        crd = np.concatenate([np.full_like(crd, h), np.full_like(crd, w), crd], axis=0)
        crd_ls.append(crd)
        data_ls.append(res.data)
    if data_ls:
        return sparse.COO(
            coords=np.concatenate(crd_ls, axis=1),
            data=np.concatenate(data_ls),
            shape=sub.shape,
        )
    else:
        return sparse.zeros(sub.shape)
