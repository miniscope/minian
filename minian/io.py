"""Video and movie file I/O for MiniAn."""

import functools as fct
import os
import re
import shutil
import sys
from collections.abc import Callable
from uuid import uuid4

import cv2
import dask as da
import dask.array as darr
import ffmpeg
import numpy as np
import skvideo.io
import xarray as xr
from natsort import natsorted
from tifffile import TiffFile, imread

from .utilities import custom_arr_optimize

if sys.version_info < (3, 13):
    from typing_extensions import deprecated
else:
    from warnings import deprecated


def _ensure_ffmpeg() -> None:
    """Require ``ffmpeg`` and ``ffprobe`` on ``PATH`` before video I/O."""
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError(
                f"{name!r} not found on PATH. MiniAn needs FFmpeg for "
                "AVI/MKV ingest and MP4 export. Install FFmpeg and ensure it "
                "is on PATH (https://ffmpeg.org/download.html)."
            )


def load_videos(
    vpath: str,
    pattern: str = r"msCam[0-9]+\.avi$",
    dtype: str | type = np.float32,
    downsample: dict | None = None,
    downsample_strategy: str = "subset",
    post_process: Callable | None = None,
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
        Datatype of the resulting DataArray, by default `np.float32`. The source
        footage is 8-bit, so its values are represented exactly in `float32`
        while halving memory and on-disk size relative to `float64`.
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
        list[str], varr_list: list[xr.DataArray]) -> xr.DataArray`. By default
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
    vlist = natsorted([vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)])
    if not vlist:
        raise FileNotFoundError(
            f"No data with pattern {pattern} found in the specified folder {vpath}"
        )
    print(f"loading {len(vlist)} videos in folder {vpath}")

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = _load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = _load_tif_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(varr.shape[0]),
            "height": np.arange(varr.shape[1]),
            "width": np.arange(varr.shape[2]),
        },
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


def _load_tif_lazy(fname: str) -> darr.array:
    """Lazy load a tif stack of images."""
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(_load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape) for fm in flist]
    return da.array.stack(arr, axis=0)


def _load_tif_perframe(fname: str, fid: int) -> np.ndarray:
    """Load a single image from a tif stack."""
    return imread(fname, key=fid)


@deprecated(
    "load_avi_lazy_framewise is deprecated in v1.3.0 and will be removed in "
    "v2.0.0. Use load_videos (ffmpeg-based lazy loading) instead."
)
def load_avi_lazy_framewise(fname: str) -> darr.array:
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape) for fm in flist]
    return da.array.stack(arr, axis=0)


def _load_avi_lazy(fname: str) -> darr.array:
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
    _ensure_ffmpeg()
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


@deprecated(
    "load_avi_perframe is deprecated in v1.3.0 and will be removed in v2.0.0. "
    "Use load_videos (ffmpeg-based lazy loading) instead."
)
def load_avi_perframe(fname: str, fid: int) -> np.ndarray:
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print(f"frame read failed for frame {fid}")
        return np.zeros((h, w))


@deprecated(
    "write_vid_blk is deprecated in v1.3.0 and will be removed in v2.0.0. Use write_video instead."
)
def write_vid_blk(arr: np.ndarray | darr.Array, vpath: str, options: dict) -> str:
    _ensure_ffmpeg()
    uid = uuid4()
    vname = f"{uid}.mp4"
    fpath = os.path.join(vpath, vname)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)
    writer = skvideo.io.FFmpegWriter(fpath, outputdict={"-" + k: v for k, v in options.items()})
    for fm in arr:
        writer.writeFrame(fm)
    writer.close()
    return fpath


def write_video(
    arr: xr.DataArray,
    vname: str | None = None,
    vpath: str | None = ".",
    norm: bool = True,
    options: dict | None = None,
) -> str:
    """
    Write a video from a movie array using `python-ffmpeg`.

    Parameters
    ----------
    arr : xr.DataArray
        Input movie array. Should have dimensions: ("frame", "height", "width")
        and should only be chunked along the "frame" dimension.
    vname : str, optional
        The name of output video. If `None` then a random one will be generated
        using :func:`uuid4.uuid`. By default `None`.
    vpath : str, optional
        The path to the folder containing the video. By default `"."`.
    norm : bool, optional
        Whether to normalize the values of the input array such that they span
        the full pixel depth range (0, 255). By default `True`.
    options : dict, optional
        Optional output arguments passed to `ffmpeg`. By default `{"crf": "18",
        "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        The absolute path to the video file.

    See Also
    --------
    ffmpeg.output
    """
    if options is None:
        options = {"crf": "18", "preset": "ultrafast"}
    _ensure_ffmpeg()
    if not vname:
        vname = f"{uuid4()}.mp4"
    fname = os.path.join(vpath, vname)
    if norm:
        arr_opt = fct.partial(custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"})
        with da.config.set(array_optimize=arr_opt):
            arr = arr.astype(np.float32)
            arr_max = arr.max().compute().values
            arr_min = arr.min().compute().values
        den = arr_max - arr_min
        arr -= arr_min
        arr /= den
        arr *= 255
    arr = arr.clip(0, 255).astype(np.uint8)
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s=f"{w}x{h}")
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return fname


@deprecated("concat_video_recursive is deprecated in v1.3.0 and will be removed in v2.0.0.")
def concat_video_recursive(vlist: list[str], vname: str | None = None) -> str:
    _ensure_ffmpeg()
    if not len(vlist) > 1:
        return vlist[0]
    if len(vlist) > 256:
        vlist = np.array_split(vlist, 256)
        vlist = [concat_video_recursive(list(v)) for v in vlist]
    vpath = os.path.dirname(vlist[0])
    streams = [ffmpeg.input(p) for p in vlist]
    if vname is None:
        vname = f"{uuid4()}.mp4"
    fpath = os.path.join(vpath, vname)
    ffmpeg.concat(*streams).output(fpath).run(overwrite_output=True)
    for vp in vlist:
        os.remove(vp)
    return fpath
