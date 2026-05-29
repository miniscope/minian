import functools as fct
import os
import re
import shutil
import warnings
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, List, Optional, Union
from uuid import uuid4

import _operator
import cv2
import dask as da
import dask.array as darr
import ffmpeg
import numpy as np
import pandas as pd
import rechunker
import xarray as xr
import zarr as zr
from dask.core import flatten
from dask.delayed import optimize as default_delay_optimize
from dask.optimization import cull, fuse, inline, inline_functions
from dask.utils import ensure_dict
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import SchedulerState, cast
from natsort import natsorted
from scipy.ndimage.filters import median_filter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from tifffile import TiffFile, imread


def load_videos(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    dtype: Union[str, type] = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Load multiple videos in a folder and return a `xr.DataArray`.

    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together and return a `xr.DataArray`
    representation of the concatenated videos. The videos are sorted by
    filenames with :func:`natsort.natsorted` before concatenation. Optionally
    the data can be downsampled, and the user can pass in a custom callable to
    post-process the result.

    Parameters
    ----------
    vpath : str
        The path containing the videos to load.
    pattern : regexp, optional
        The regexp matching the filenames of the videso. By default
        `r"msCam[0-9]+\.avi$"`, which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi".
    dtype : Union[str, type], optional
        Datatype of the resulting DataArray, by default `np.float64`.
    downsample : dict, optional
        A dictionary mapping dimension names to an integer downsampling factor.
        The dimension names should be one of "height", "width" or "frame". By
        default `None`.
    downsample_strategy : str, optional
        How the downsampling should be done. Only used if `downsample` is not
        `None`. Either `"subset"` where data points are taken at an interval
        specified in `downsample`, or `"mean"` where mean will be taken over
        data within each interval. By default `"subset"`.
    post_process : Callable, optional
        An user-supplied custom function to post-process the resulting array.
        Four arguments will be passed to the function: the resulting DataArray
        `varr`, the input path `vpath`, the list of matched video filenames
        `vlist`, and the list of DataArray before concatenation `varr_list`. The
        function should output another valide DataArray. In other words, the
        function should have signature `f(varr: xr.DataArray, vpath: str, vlist:
        List[str], varr_list: List[xr.DataArray]) -> xr.DataArray`. By default
        `None`

    Returns
    -------
    varr : xr.DataArray
        The resulting array representation of the input movie. Should have
        dimensions ("frame", "height", "width").

    Raises
    ------
    FileNotFoundError
        if no files under `vpath` match the pattern `pattern`
    ValueError
        if the matched files does not have extension ".avi", ".mkv" or ".tif"
    NotImplementedError
        if `downsample_strategy` is not "subset" or "mean"
    """
    vpath = os.path.normpath(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = load_tif_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]
    return varr


def load_tif_lazy(fname: str) -> darr.array:
    """
    Lazy load a tif stack of images.

    Parameters
    ----------
    fname : str
        The filename of the tif stack to load.

    Returns
    -------
    arr : darr.array
        Resulting dask array representation of the tif stack.
    """
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_tif_perframe(fname: str, fid: int) -> np.ndarray:
    """
    Load a single image from a tif stack.

    Parameters
    ----------
    fname : str
        The filename of the tif stack.
    fid : int
        The index of the image to load.

    Returns
    -------
    arr : np.ndarray
        Array representation of the image.
    """
    return imread(fname, key=fid)


def load_avi_lazy_framewise(fname: str) -> darr.array:
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_avi_lazy(fname: str) -> darr.array:
    """
    Lazy load an avi video.

    This function construct a single delayed task for loading the video as a
    whole.

    Parameters
    ----------
    fname : str
        The filename of the video to load.

    Returns
    -------
    arr : darr.array
        The array representation of the video.
    """
    probe = ffmpeg.probe(fname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w = int(video_info["width"])
    h = int(video_info["height"])
    f = int(video_info["nb_frames"])
    return da.array.from_delayed(
        da.delayed(load_avi_ffmpeg)(fname, h, w, f), dtype=np.uint8, shape=(f, h, w)
    )


def load_avi_ffmpeg(fname: str, h: int, w: int, f: int) -> np.ndarray:
    """
    Load an avi video using `ffmpeg`.

    This function directly invoke `ffmpeg` using the `python-ffmpeg` wrapper and
    retrieve the data from buffer.

    Parameters
    ----------
    fname : str
        The filename of the video to load.
    h : int
        The height of the video.
    w : int
        The width of the video.
    f : int
        The number of frames in the video.

    Returns
    -------
    arr : np.ndarray
        The resulting array. Has shape (`f`, `h`, `w`).
    """
    out_bytes, err = (
        ffmpeg.input(fname)
        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True)
    )
    return np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)


def load_avi_perframe(fname: str, fid: int) -> np.ndarray:
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print("frame read failed for frame {}".format(fid))
        return np.zeros((h, w))


def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=False
) -> Union[dict, xr.Dataset]:
    """
    Load an existing minian dataset.

    If `dpath` is a file, then it is assumed that the full dataset is saved as a
    single file, and this function will directly call
    :func:`xarray.open_dataset` on `dpath`. Otherwise if `dpath` is a directory,
    then it is assumed that the dataset is saved as a directory of `zarr`
    arrays, as produced by :func:`save_minian`. This function will then iterate
    through all the directories under input `dpath` and load them as
    `xr.DataArray` with `zarr` backend, so it is important that the user make
    sure every directory under `dpath` can be load this way. The loaded arrays
    will be combined as either a `xr.Dataset` or a `dict`. Optionally a
    user-supplied custom function can be used to post process the resulting
    `xr.Dataset`.

    Parameters
    ----------
    dpath : str
        The path to the minian dataset that should be loaded.
    post_process : Callable, optional
        User-supplied function to post process the dataset. Only used if
        `return_dict` is `False`. Two arguments will be passed to the function:
        the resulting dataset `ds` and the data path `dpath`. In other words the
        function should have signature `f(ds: xr.Dataset, dpath: str) ->
        xr.Dataset`. By default `None`.
    return_dict : bool, optional
        Whether to combine the DataArray as dictionary, where the `.name`
        attribute will be used as key. Otherwise the DataArray will be combined
        using `xr.merge(..., compat="no_conflicts")`, which will implicitly
        align the DataArray over all dimensions, so it is important to make sure
        the coordinates are compatible and will not result in creation of large
        NaN-padded results. Only used if `dpath` is a directory, otherwise a
        `xr.Dataset` is always returned. By default `False`.

    Returns
    -------
    ds : Union[dict, xr.Dataset]
        The resulting dataset. If `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`.

    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
    """
    if isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif isdir(dpath):
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds


def open_minian_mf(
    dpath: str,
    index_dims: List[str],
    result_format="xarray",
    pattern=r"minian$",
    sub_dirs: List[str] = [],
    exclude=True,
    **kwargs,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Open multiple minian datasets across multiple directories.

    This function recursively walks through directories under `dpath` and try to
    load minian datasets from all directories matching `pattern`. It will then
    combine them based on `index_dims` into either a `xr.Dataset` object or a
    `pd.DataFrame`. Optionally a subset of paths can be specified, so that they
    can either be excluded or white-listed. Additional keyword arguments will be
    passed directly to :func:`open_minian`.

    Parameters
    ----------
    dpath : str
        The root folder containing all datasets to be loaded.
    index_dims : List[str]
        List of dimensions that can be used to index and merge multiple
        datasets. All loaded datasets should have unique coordinates in the
        listed dimensions.
    result_format : str, optional
        If `"xarray"`, the result will be merged together recursively along each
        dimensions listed in `index_dims`. Users should make sure the
        coordinates are compatible and the merging will not cause generation of
        large NaN-padded results. If `"pandas"`, then a `pd.DataFrame` is
        returned, with columns corresponding to `index_dims` uniquely identify
        each dataset, and an additional column named "minian" of object dtype
        pointing to the loaded minian dataset objects. By default `"xarray"`.
    pattern : regexp, optional
        Pattern of minian dataset directory names. By default `r"minian$"`.
    sub_dirs : List[str], optional
        A list of sub-directories under `dpath`. Useful if only a subset of
        datasets under `dpath` should be recursively loaded. By default `[]`.
    exclude : bool, optional
        Whether to exclude directories listed under `sub_dirs`. If `True`, then
        any minian datasets under those specified in `sub_dirs` will be ignored.
        If `False`, then **only** the datasets under those specified in
        `sub_dirs` will be loaded (they still have to be under `dpath` though).
        by default `True`.

    Returns
    -------
    ds : Union[xr.Dataset, pd.DataFrame]
        The resulting combined datasets. If `result_format` is `"xarray"`, then
        a `xr.Dataset` will be returned, otherwise a `pd.DataFrame` will be
        returned.

    Raises
    ------
    NotImplementedError
        if `result_format` is not "xarray" or "pandas"
    """
    minian_dict = dict()
    for nextdir, dirlist, filelist in os.walk(dpath, topdown=False):
        nextdir = os.path.abspath(nextdir)
        cur_path = Path(nextdir)
        dir_tag = bool(
            (
                (any([Path(epath) in cur_path.parents for epath in sub_dirs]))
                or nextdir in sub_dirs
            )
        )
        if exclude == dir_tag:
            continue
        flist = list(filter(lambda f: re.search(pattern, f), filelist + dirlist))
        if flist:
            print("opening dataset under {}".format(nextdir))
            if len(flist) > 1:
                warnings.warn("multiple dataset found: {}".format(flist))
            fname = flist[-1]
            print("opening {}".format(fname))
            minian = open_minian(dpath=os.path.join(nextdir, fname), **kwargs)
            key = tuple([np.array_str(minian[d].values) for d in index_dims])
            minian_dict[key] = minian
            print(["{}: {}".format(d, v) for d, v in zip(index_dims, key)])

    if result_format == "xarray":
        return xrconcat_recursive(minian_dict, index_dims)
    elif result_format == "pandas":
        minian_df = pd.Series(minian_dict).rename("minian")
        minian_df.index.set_names(index_dims, inplace=True)
        return minian_df.to_frame()
    else:
        raise NotImplementedError("format {} not understood".format(result_format))


def save_minian(
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="500MB",
) -> xr.DataArray:
    """
    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.

    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using
    :func:`rechunker.rechunk` if `chunks` are given.

    Parameters
    ----------
    var : xr.DataArray
        The array to be saved.
    dpath : str
        The path to the minian dataset directory.
    meta_dict : dict, optional
        How metadata should be retrieved from directory hierarchy. The keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory. The actual coordinate value of the dimensions will
        be the directory name of corresponding level. By default `None`.
    overwrite : bool, optional
        Whether to overwrite the result on disk. By default `False`.
    chunks : dict, optional
        A dictionary specifying the desired chunk size. The chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported. The rechunking operation will be
        carried out with on-disk algorithms using :func:`rechunker.rechunk`. By
        default `None`.
    compute : bool, optional
        Whether to compute `var` and save it immediately. By default `True`.
    mem_limit : str, optional
        The memory limit for the on-disk rechunking algorithm, passed to
        :func:`rechunker.rechunk`. Only used if `chunks` is not `None`. By
        default `"500MB"`.

    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.

    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: `{"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}`.

    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... ) # doctest: +SKIP
    """
    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
            zstore = zr.open(fp)
            rechk = rechunker.rechunk(
                zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
            )
            rechk.execute()
        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass
        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr


def xrconcat_recursive(var: Union[dict, list], dims: List[str]) -> xr.Dataset:
    """
    Recursively concatenate `xr.DataArray` over multiple dimensions.

    Parameters
    ----------
    var : Union[dict, list]
        Either a `dict` or a `list` of `xr.DataArray` to be concatenated. If a
        `dict` then keys should be `tuple`, with length same as the length of
        `dims` and values corresponding to the coordinates that uniquely
        identify each `xr.DataArray`. If a `list` then each `xr.DataArray`
        should contain valid coordinates for each dimensions specified in
        `dims`.
    dims : List[str]
        Dimensions to be concatenated over.

    Returns
    -------
    ds : xr.Dataset
        The concatenated dataset.

    Raises
    ------
    NotImplementedError
        if input `var` is neither a `dict` nor a `list`
    """
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([np.asscalar(v[d]) for d in dims]): v for v in var}
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        data = np.empty(len(var_dict), dtype=object)
        for iv, ds in enumerate(var_dict.values()):
            data[iv] = ds
        index = pd.MultiIndex.from_tuples(list(var_dict.keys()), names=dims)
        var_ps = pd.Series(data=data, index=index)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = list(var.values())
        return xr.concat(var, dim=dims[0])


def update_meta(dpath, pattern=r"^minian\.nc$", meta_dict=None, backend="netcdf"):
    for dirpath, dirnames, fnames in os.walk(dpath):
        if backend == "netcdf":
            fnames = filter(lambda fn: re.search(pattern, fn), fnames)
        elif backend == "zarr":
            fnames = filter(lambda fn: re.search(pattern, fn), dirnames)
        else:
            raise NotImplementedError("backend {} not supported".format(backend))
        for fname in fnames:
            f_path = os.path.join(dirpath, fname)
            pathlist = os.path.normpath(dirpath).split(os.sep)
            new_ds = xr.Dataset()
            old_ds = open_minian(f_path, f_path, backend)
            new_ds.attrs = deepcopy(old_ds.attrs)
            old_ds.close()
            new_ds = new_ds.assign_coords(
                **dict(
                    [(cdname, pathlist[cdval]) for cdname, cdval in meta_dict.items()]
                )
            )
            if backend == "netcdf":
                new_ds.to_netcdf(f_path, mode="a")
            elif backend == "zarr":
                new_ds.to_zarr(f_path, mode="w")
            print("updated: {}".format(f_path))


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


ANNOTATIONS = {
    "from-zarr-store": {"resources": {"MEM": 1}},
    "load_avi_ffmpeg": {"resources": {"MEM": 1}},
    "est_motion_chunk": {"resources": {"MEM": 1}},
    "transform_perframe": {"resources": {"MEM": 0.5}},
    "pnr_perseed": {"resources": {"MEM": 0.5}},
    "ks_perseed": {"resources": {"MEM": 0.5}},
    "smooth_corr": {"resources": {"MEM": 1}},
    "vectorize_noise_fft": {"resources": {"MEM": 1}},
    "vectorize_noise_welch": {"resources": {"MEM": 1}},
    "update_spatial_block": {"resources": {"MEM": 1}},
    "tensordot_restricted": {"resources": {"MEM": 1}},
    "update_temporal_block": {"resources": {"MEM": 1}},
    "merge_restricted": {"resources": {"MEM": 1}},
}
"""
Dask annotations that should be applied to each task.

This is a `dict` mapping task names (actually patterns) to a `dict` of dask
annotations that should be applied to the tasks. It is mainly used to constrain
number of tasks that can be concurrently in memory for each worker.

See Also
-------
:doc:`distributed:resources`
"""

FAST_FUNCTIONS = [
    darr.core.getter_inline,
    darr.core.getter,
    _operator.getitem,
    zr.core.Array,
    darr.chunk.astype,
    darr.core.concatenate_axes,
    darr.core._vindex_slice,
    darr.core._vindex_merge,
    darr.core._vindex_transpose,
]
"""
List of fast functions that should be inlined during optimization.

See Also
-------
:doc:`dask:optimize`
"""


class TaskAnnotation(SchedulerPlugin):
    """
    Custom `SchedulerPlugin` that implemented per-task level annotation. The
    annotations are applied according to the module constant
    :const:`ANNOTATIONS`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.annt_dict = ANNOTATIONS

    def update_graph(self, scheduler, client, tasks, **kwargs):
        parent = cast(SchedulerState, scheduler)
        for tk in tasks.keys():
            for pattern, annt in self.annt_dict.items():
                if re.search(pattern, tk):
                    ts = parent.tasks.get(tk)
                    res = annt.get("resources", None)
                    if res:
                        ts.resource_restrictions = res
                    pri = annt.get("priority", None)
                    if pri:
                        pri_org = list(ts._priority)
                        pri_org[0] = -pri
                        ts._priority = tuple(pri_org)


def custom_arr_optimize(
    dsk: dict,
    keys: list,
    fast_funcs: list = FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict: Optional[dict] = None,
    rewrite_dict: Optional[dict] = None,
    keep_patterns=[],
    flush=True,
) -> dict:
    """
    Customized implementation of array optimization function.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_funcs : list, optional
        List of fast functions to be inlined. By default :const:`FAST_FUNCTIONS`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    rename_dict : dict, optional
        Dictionary mapping old task keys to new ones. Only used during fusing of
        tasks. By default `None`.
    rewrite_dict : dict, optional
        Dictionary mapping old task key substrings to new ones. Applied at the
        end of optimization to all task keys. By default `None`.
    keep_patterns : list, optional
        List of patterns of task keys that should be preserved during
        optimization. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.

    See Also
    -------
    :doc:`dask:optimize`
    `dask.array.optimization.optimize`
    """
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk


def rewrite_key(key: Union[str, tuple], rwdict: dict) -> str:
    """
    Rewrite a task key according to `rwdict`.

    Parameters
    ----------
    key : Union[str, tuple]
        Input task key.
    rwdict : dict
        Dictionary mapping old task key substring to new ones. All keys in this
        dictionary that exists in input `key` will be substituted.

    Returns
    -------
    key : str
        The new key.

    Raises
    ------
    ValueError
        if input `key` is neither `str` or `tuple`
    """
    typ = type(key)
    if typ is tuple:
        k = key[0]
    elif typ is str:
        k = key
    else:
        raise ValueError("key must be either str or tuple: {}".format(key))
    for pat, repl in rwdict.items():
        k = re.sub(pat, repl, k)
    if typ is tuple:
        ret_key = list(key)
        ret_key[0] = k
        return tuple(ret_key)
    else:
        return k


def custom_fused_keys_renamer(
    keys: list, max_fused_key_length=120, rename_dict: Optional[dict] = None
) -> str:
    """
    Custom implmentation to create new keys for `fuse` tasks.

    Uses custom `split_key` implementation.

    Parameters
    ----------
    keys : list
        List of task keys that should be fused together.
    max_fused_key_length : int, optional
        Used to limit the maximum string length for each renamed key. If `None`,
        there is no limit. By default `120`.
    rename_dict : dict, optional
        Dictionary used to rename keys during fuse. By default `None`.

    Returns
    -------
    fused_key : str
        The fused task key.

    See Also
    -------
    split_key
    dask.optimization.fuse
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)

    if max_fused_key_length:  # Take into account size of hash suffix
        max_fused_key_length -= 5

    def _enforce_max_key_limit(key_name):
        if max_fused_key_length and len(key_name) > max_fused_key_length:
            name_hash = f"{hash(key_name):x}"[:4]
            key_name = f"{key_name[:max_fused_key_length]}-{name_hash}"
        return key_name

    if typ is str:
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key)
        concatenated_name = "-".join(names)
        return _enforce_max_key_limit(concatenated_name)
    elif typ is tuple and len(first_key) > 0 and isinstance(first_key[0], str):
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key[0])
        concatenated_name = "-".join(names)
        return (_enforce_max_key_limit(concatenated_name),) + first_key[1:]


def split_key(key: Union[tuple, str], rename_dict: Optional[dict] = None) -> str:
    """
    Split, rename and filter task keys.

    This is custom implementation that only keeps keys found in :const:`ANNOTATIONS`.

    Parameters
    ----------
    key : Union[tuple, str]
        The input task key.
    rename_dict : dict, optional
        Dictionary used to rename keys. By default `None`.

    Returns
    -------
    new_key : str
        New key.
    """
    if type(key) is tuple:
        key = key[0]
    kls = key.split("-")
    if rename_dict:
        kls = list(map(lambda k: rename_dict.get(k, k), kls))
    kls_ft = list(filter(lambda k: k in ANNOTATIONS.keys(), kls))
    if kls_ft:
        return "-".join(kls_ft)
    else:
        return kls[0]


def check_key(key: Union[str, tuple], pat: str) -> bool:
    """
    Check whether `key` contains pattern.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat : str
        Pattern to check.

    Returns
    -------
    bool
        Whether `key` contains pattern.
    """
    try:
        return bool(re.search(pat, key))
    except TypeError:
        return bool(re.search(pat, key[0]))


def check_pat(key: Union[str, tuple], pat_ls: List[str]) -> bool:
    """
    Check whether `key` contains any pattern in a list.

    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat_ls : List[str]
        List of pattern to check.

    Returns
    -------
    bool
        Whether `key` contains any pattern in the list.
    """
    for pat in pat_ls:
        if check_key(key, pat):
            return True
    return False


def inline_pattern(dsk: dict, pat_ls: List[str], inline_constants: bool) -> dict:
    """
    Inline tasks whose keys match certain patterns.

    Parameters
    ----------
    dsk : dict
        Input dask graph.
    pat_ls : List[str]
        List of patterns to check.
    inline_constants : bool
        Whether to inline constants.

    Returns
    -------
    dsk : dict
        Dask graph with keys inlined.

    See Also
    -------
    dask.optimization.inline
    """
    keys = [k for k in dsk.keys() if check_pat(k, pat_ls)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants)
        for k in keys:
            del dsk[k]
        if inline_constants:
            dsk, dep = cull(dsk, set(list(flatten(keys))))
    return dsk


def custom_delay_optimize(
    dsk: dict, keys: list, fast_functions=[], inline_patterns=[], **kwargs
) -> dict:
    """
    Custom optimization functions for delayed tasks.

    By default only fusing of tasks will be carried out.

    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_functions : list, optional
        List of fast functions to be inlined. By default `[]`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.

    Returns
    -------
    dsk : dict
        Optimized dask graph.
    """
    dsk, _ = fuse(ensure_dict(dsk), rename_keys=custom_fused_keys_renamer)
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if fast_functions:
        dsk = inline_functions(
            dsk,
            [],
            fast_functions=fast_functions,
        )
    return dsk


def unique_keys(keys: list) -> np.ndarray:
    """
    Returns only unique keys in a list of task keys.

    Dask task keys regarding arrays are usually tuples representing chunked
    operations. This function ignore different chunks and only return unique keys.

    Parameters
    ----------
    keys : list
        List of dask keys.

    Returns
    -------
    unique : np.ndarray
        Unique keys.
    """
    new_keys = []
    for k in keys:
        if isinstance(k, tuple):
            new_keys.append("chunked-" + k[0])
        elif isinstance(k, str):
            new_keys.append(k)
    return np.unique(new_keys)


def get_keys_pat(pat: str, keys: list, return_all=False) -> Union[list, str]:
    """
    Filter a list of task keys by pattern.

    Parameters
    ----------
    pat : str
        Pattern to check.
    keys : list
        List of keys to be filtered.
    return_all : bool, optional
        Whether to return all keys matching `pat`. If `False` then only the
        first match will be returned. By default `False`.

    Returns
    -------
    keys : Union[list, str]
        If `return_all` is `True` then a list of keys will be returned.
        Otherwise only one key will be returned.
    """
    keys_filt = list(filter(lambda k: check_key(k, pat), list(keys)))
    if return_all:
        return keys_filt
    else:
        return keys_filt[0]


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


def local_extreme(fm: np.ndarray, k: np.ndarray, etype="max", diff=0) -> np.ndarray:
    """
    Find local extreme of a 2d array.

    Parameters
    ----------
    fm : np.ndarray
        The input 2d array.
    k : np.ndarray
        Structuring element defining the locality of the result, passed as
        `kernel` to :func:`cv2.erode` and :func:`cv2.dilate`.
    etype : str, optional
        Type of local extreme. Either `"min"` or `"max"`. By default `"max"`.
    diff : int, optional
        Threshold of difference between local extreme and its neighbours. By
        default `0`.

    Returns
    -------
    fm_ext : np.ndarray
        The returned 2d array whose non-zero elements represent the location of
        local extremes.

    Raises
    ------
    ValueError
        if `etype` is not "min" or "max"
    """
    fm_max = cv2.dilate(fm, k)
    fm_min = cv2.erode(fm, k)
    fm_diff = ((fm_max - fm_min) > diff).astype(np.uint8)
    if etype == "max":
        fm_ext = (fm == fm_max).astype(np.uint8)
    elif etype == "min":
        fm_ext = (fm == fm_min).astype(np.uint8)
    else:
        raise ValueError("Don't understand {}".format(etype))
    return cv2.bitwise_and(fm_ext, fm_diff).astype(np.uint8)


def med_baseline(a: np.ndarray, wnd: int) -> np.ndarray:
    """
    Subtract baseline from a timeseries as estimated by median-filtering the
    timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    wnd : int
        Window size of the median filter. This parameter is passed as `size` to
        :func:`scipy.ndimage.filters.median_filter`.

    Returns
    -------
    a : np.ndarray
        Timeseries with baseline subtracted.
    """
    base = median_filter(a, size=wnd)
    a -= base
    return a.clip(0, None)


@darr.as_gufunc(signature="(m,n),(m)->(n)", output_dtypes=float)
def sps_lstsq(a: csc_matrix, b: np.ndarray, **kwargs):
    out = np.zeros((b.shape[0], a.shape[1]))
    for i in range(b.shape[0]):
        out[i, :] = lsqr(a, b[i, :].squeeze(), **kwargs)[0]
    return out
