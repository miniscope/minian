"""Video export helpers (ffmpeg, skvideo, OpenCV-free paths where possible)."""

import logging
import os
from typing import Optional
from uuid import uuid4

import dask
import ffmpeg
import numpy as np
import scipy.sparse as scisps
import skvideo.io
import xarray as xr

from ..cnmf import compute_AtC
from ..constants import MINIAN

log = logging.getLogger(__name__)

RGB_MAX = 255
RGB_MIN = 0


def _stats_chunked_for_reduce(arr: xr.DataArray) -> xr.DataArray:
    """
    Rechunk for global min/max so each task stays small.

    Wide spatial concat + distributed workers with low RAM can otherwise build
    huge ``rechunk-merge`` tasks during tree reductions.
    """
    if not getattr(arr.data, "chunks", None):
        return arr
    nf = max(1, int(arr.sizes["frame"]))
    frame_chunk = min(32, nf)
    return arr.chunk({"frame": frame_chunk, "height": -1, "width": -1})


def write_vid_blk(arr, vpath, options):
    uid = uuid4()
    vname = "{}.mp4".format(uid)
    fpath = os.path.join(vpath, vname)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)
    writer = skvideo.io.FFmpegWriter(
        fpath, outputdict={"-" + k: v for k, v in options.items()}
    )
    for fm in arr:
        writer.writeFrame(fm)
    writer.close()
    return fpath


def write_video(
    arr: xr.DataArray,
    vname: Optional[str] = None,
    vpath: Optional[str] = ".",
    norm=True,
    options={"crf": "18", "preset": "ultrafast"},
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
        the full pixel depth range (RGB_MIN, RGB_MAX). By default `True`.
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
    if not vname:
        vname = "{}.mp4".format(uuid4())
    fname = os.path.join(vpath, vname)
    # Thread scheduler: avoids tiny distributed workers OOMing on rechunk-merge
    # during full-array reductions and per-block reads for ffmpeg.
    with dask.config.set(scheduler="threads"):
        if norm:
            arr = arr.astype(np.float32)
            stats_arr = _stats_chunked_for_reduce(arr)
            arr_max, arr_min = dask.compute(stats_arr.max(), stats_arr.min())
            arr_max = arr_max.item()
            arr_min = arr_min.item()
            den = arr_max - arr_min
            arr -= arr_min
            arr /= den
            arr *= RGB_MAX
        arr = arr.clip(RGB_MIN, RGB_MAX).astype(np.uint8)
        w, h = arr.sizes["width"], arr.sizes["height"]
        process = (
            ffmpeg.input(
                "pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h)
            )
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


def concat_video_recursive(vlist, vname=None):
    if not len(vlist) > 1:
        return vlist[0]
    if len(vlist) > 256:
        vlist = np.array_split(vlist, 256)
        vlist = [concat_video_recursive(list(v)) for v in vlist]
    vpath = os.path.dirname(vlist[0])
    streams = [ffmpeg.input(p) for p in vlist]
    if vname is None:
        vname = "{}.mp4".format(uuid4())
    fpath = os.path.join(vpath, vname)
    ffmpeg.concat(*streams).output(fpath).run(overwrite_output=True)
    for vp in vlist:
        os.remove(vp)
    return fpath


def generate_videos(
    varr: xr.DataArray,
    Y: xr.DataArray,
    A: Optional[xr.DataArray] = None,
    C: Optional[xr.DataArray] = None,
    AC: Optional[xr.DataArray] = None,
    nfm_norm: int = None,
    gain=1.5,
    vpath=".",
    vname=f"{MINIAN}.mp4",
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Generate a video visualizing the result of the minian pipeline.

    The resulting video contains four parts: Top left is a original reference
    movie supplied as `varr`; Top right is the input to CNMF algorithm supplied
    as `Y`; Bottom right is a movie `AC` representing cellular activities as
    computed by :func:`minian.cnmf.compute_AtC`; Bottom left is a residule movie
    computed as the difference between `Y` and `AC`. Since the CNMF algorithm
    contains various arbitrary scaling process, a normalizing scalar is computed
    with least square using a subset of frames from `Y` and `AC` such that their
    numerical values matches.

    Parameters
    ----------
    varr : xr.DataArray
        Input reference movie data. Should have dimensions ("frame", "height",
        "width"), and should only be chunked along "frame" dimension.
    Y : xr.DataArray
        Movie data representing input to CNMF algorithm. Should have dimensions
        ("frame", "height", "width"), and should only be chunked along "frame"
        dimension.
    A : xr.DataArray, optional
        Spatial footprints of cells. Only used if `AC` is `None`. By default
        `None`.
    C : xr.DataArray, optional
        Temporal activities of cells. Only used if `AC` is `None`. By default
        `None`.
    AC : xr.DataArray, optional
        Spatial-temporal activities of cells. Should have dimensions ("frame",
        "height", "width"), and should only be chunked along "frame" dimension.
        If `None` then both `A` and `C` should be supplied and
        :func:`minian.cnmf.compute_AtC` will be used to compute this variable.
        By default `None`.
    nfm_norm : int, optional
        Number of frames to randomly draw from `Y` and `AC` to compute the
        normalizing factor with least square. By default `None`.
    gain : float, optional
        A gain factor multiplied to `Y`. Useful to make the results visually
        brighter. By default `1.5`.
    vpath : str, optional
        Desired folder containing the resulting video. By default `"."`.
    vname : str, optional
        Desired name of the video (default basename ``minian.mp4`` from :data:`~minian.constants.MINIAN`).
    options : dict, optional
        Output options for `ffmpeg`, passed directly to :func:`write_video`. By
        default `{"crf": "18", "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        Absolute path of the resulting video.
    """
    if AC is None:
        log.info("generating traces")
        AC = compute_AtC(A, C)
    log.info("normalizing")
    gain = RGB_MAX / Y.max().compute().values * gain
    Y = Y * gain
    if nfm_norm is not None:
        norm_idx = np.sort(
            np.random.choice(np.arange(Y.sizes["frame"]), size=nfm_norm, replace=False)
        )
        Y_sub = Y.isel(frame=norm_idx).values.reshape(-1)
        AC_sub = scisps.csc_matrix(AC.isel(frame=norm_idx).values.reshape((-1, 1)))
        lsqr = scisps.linalg.lsqr(AC_sub, Y_sub)
        norm_factor = lsqr[0].item()
        del Y_sub, AC_sub
    else:
        norm_factor = gain
    AC = AC * norm_factor
    res = Y - AC
    log.info("writing videos")
    vid = xr.concat(
        [
            xr.concat([varr, Y], "width", coords="minimal"),
            xr.concat([res, AC], "width", coords="minimal"),
        ],
        "height",
        coords="minimal",
    )
    return write_video(vid, vname, vpath, norm=False, options=options)
