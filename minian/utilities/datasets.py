"""Video loaders, persisted Minian datasets, concat helpers, metadata updates."""

import functools as fct
import logging
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

import cv2
import dask as da
import dask.array as darr
import ffmpeg
import numpy as np
import pandas as pd
import rechunker
import xarray as xr
import zarr as zr
from dask.delayed import optimize as default_delay_optimize
from natsort import natsorted
from tifffile import TiffFile, imread

from ..constants import MINIAN
from ..visualization._ffmpeg_constants import RawGray
from .dask_graph import custom_arr_optimize

log = logging.getLogger(__name__)

# Arrays this size or smaller are loaded into memory before zarr write when
# ``compute=True``, so workers never execute graphs that only produce a tiny
# on-disk array but still depend on large zarr-backed movies (e.g. ``motion``).
_SAVE_MATERIALIZE_NBYTES_DEFAULT = 256 * 1024 * 1024

# Minian's global ``custom_arr_optimize`` can confuse local schedulers during
# ``load()`` (``ValueError: Missing dependency`` on ``rechunk-merge`` / gufuncs).
_EAGER_LOAD_DASK = {
    "array_optimize": darr.optimization.optimize,
    "delayed_optimize": default_delay_optimize,
}


def _has_distributed_client() -> bool:
    try:
        from distributed import default_client

        default_client()
        return True
    except (ImportError, ValueError):
        return False


def _dataset_to_zarr_compute(ds: xr.Dataset, fp: str, mode: str):
    """``Dataset.to_zarr(compute=True)`` avoiding worker OOM when a Client is active.

    Uses the threaded scheduler plus default optimizers so the write does not
    fan out entirely on low-RAM workers. Falls back to the distributed scheduler
    if the graph raises ``Missing dependency`` (e.g. after ``persist()``).
    """
    if not _has_distributed_client():
        return ds.to_zarr(fp, compute=True, mode=mode)
    try:
        with da.config.set(scheduler="threads", **_EAGER_LOAD_DASK):
            return ds.to_zarr(fp, compute=True, mode=mode)
    except ValueError as err:
        if "Missing dependency" not in str(err):
            raise
        log.warning(
            "save_minian: threads to_zarr failed (%s); retrying distributed", err
        )
        return ds.to_zarr(fp, compute=True, mode=mode)


def _eager_load_for_zarr(var: xr.DataArray, nbytes: int) -> xr.DataArray:
    """
    Load ``var`` into memory so ``to_zarr`` does not ship a heavy upstream graph.

    Uses Dask's **default** array/delayed optimizers for this step only (not
    :func:`~minian.utilities.custom_arr_optimize`). With a ``Client``, first
    tries **threads** on the client (avoids heavy zarr-backed work on workers
    for small outputs); on ``Missing dependency``, retries a normal ``load()``
    on the cluster (same pattern as :func:`_dataset_to_zarr_compute`).

    Without a distributed ``Client``, uses the **synchronous** scheduler with
    the same default optimizers.
    """
    log.info(
        "save_minian: loading %r into memory before zarr write (%d bytes)",
        var.name,
        nbytes,
    )
    log.info("save_minian: computing %r before zarr write", var.name)
    if _has_distributed_client():
        try:
            with da.config.set(scheduler="threads", **_EAGER_LOAD_DASK):
                return var.load()
        except ValueError as err:
            if "Missing dependency" not in str(err):
                raise
            log.warning(
                "save_minian: threads preload failed (%s); retrying distributed",
                err,
            )
            return var.load()
    with da.config.set(scheduler="synchronous", **_EAGER_LOAD_DASK):
        return var.load()


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
        The regexp matching the filenames of the videos. By default
        `r"msCam[0-9]+\\.avi$"`, which can be interpreted as filenames starting
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
        function should output another valid DataArray. In other words, the
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
            "No data with pattern {} found in the specified folder {}".format(
                pattern, vpath
            )
        )
    log.info("loading {} videos in folder {}".format(len(vlist), vpath))

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
        .video.output(RawGray.PIPE, format=RawGray.FORMAT, pix_fmt=RawGray.PIX_FMT)
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
        log.warning("frame read failed for frame {}".format(fid))
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
        each dataset, and an additional column (name :data:`~minian.constants.MINIAN`)
        of object dtype
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
            log.info("opening dataset under {}".format(nextdir))
            if len(flist) > 1:
                warnings.warn("multiple dataset found: {}".format(flist))
            fname = flist[-1]
            log.info("opening {}".format(fname))
            minian = open_minian(dpath=os.path.join(nextdir, fname), **kwargs)
            key = tuple([np.array_str(minian[d].values) for d in index_dims])
            minian_dict[key] = minian
            log.info("%s", ["{}: {}".format(d, v) for d, v in zip(index_dims, key)])

    if result_format == "xarray":
        return xrconcat_recursive(minian_dict, index_dims)
    elif result_format == "pandas":
        minian_df = pd.Series(minian_dict).rename(MINIAN)
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
    materialize_nbytes: int = _SAVE_MATERIALIZE_NBYTES_DEFAULT,
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
    materialize_nbytes : int, optional
        If ``compute`` is ``True`` and the array's ``nbytes`` is at most this
        value, load it into memory before calling ``to_zarr``: **synchronous**
        when no ``Client`` is active; with a ``Client``, **threads** on the
        client first (default Dask optimizers for that step only), falling back
        to distributed compute if ``Missing dependency`` occurs. Set to ``0``
        to disable. By default 256 MiB.

    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.

    Examples
    -------
    The following will save the variable `var` under a subdirectory named after
    :data:`~minian.constants.MINIAN`, e.g.
    ``/spatial_memory/alpha/learning1/minian/important_array.zarr``, with the
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
    _fp_abs = os.path.abspath(fp)
    log.info(
        "save_minian: begin %r -> %s compute=%s chunks=%s",
        var.name,
        _fp_abs,
        compute,
        chunks,
    )
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    if compute and materialize_nbytes > 0:
        try:
            _nb = int(var.nbytes)
        except (TypeError, ValueError):
            _nb = None
        if _nb is not None and _nb <= materialize_nbytes:
            var = _eager_load_for_zarr(var, _nb)
            ds = var.to_dataset()
            if meta_dict is not None:
                pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
                ds = ds.assign_coords(
                    **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
                )
    if compute:
        try:
            log.info("save_minian: computing + writing zarr %s", _fp_abs)
            arr = _dataset_to_zarr_compute(ds, fp, md)
        except Exception:
            log.exception(
                "save_minian: zarr write failed; %r may be incomplete", _fp_abs
            )
            raise
        log.info("save_minian: finished %s", _fp_abs)
    else:
        arr = ds.to_zarr(fp, compute=False, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        log.info(
            "save_minian: on-disk rechunk %r chunks=%s mem_limit=%s",
            var.name,
            chunks,
            mem_limit,
        )
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
            log.info("updated: {}".format(f_path))
