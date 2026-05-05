"""HoloViews plots for pipeline exploration and diagnostics.

Plots follow **HoloViews + modern Bokeh** expectations: subplot text is driven by
:class:`holoviews:holoviews.core.dimension.Dimensioned.relabel`, not ``title=``
on :class:`~holoviews:element.Image`/:class:`~holoviews:element.RGB`/etc. Passing
``title`` inside ``hv.opts.*`` or flattened ``.opts()`` can bind a plain Python
string to ``bokeh`` ``Figure.title``; HoloViews then calls ``.update()`` as if it
were a ``Title`` model and raises ``AttributeError`` (see visualization package
docstring).

"""

import itertools as itt
from typing import Callable, List, Optional, Union

import holoviews as hv
import numpy as np
import pandas as pd
import sklearn.mixture
import xarray as xr
from bokeh.palettes import Category10_10, Viridis256
from datashader import count_cat
from holoviews import dim
from holoviews.operation.datashader import datashade, dynspread
from holoviews.util import Dynamic

from ._numeric import centroid, construct_pulse_response, normalize
from ._viz_constants import (
    Datashade,
    Gmm,
    ImagePalette,
    Motion,
    Preprocess,
    Seeds,
    Spatial,
    Temporal,
)


def datashade_ndcurve(
    ovly: hv.NdOverlay, kdim: Optional[Union[str, List[str]]] = None, spread=False
) -> hv.Overlay:
    """
    Apply datashading to an overlay of curves with legends.

    Parameters
    ----------
    ovly : hv.NdOverlay
        The input overlay of curves.
    kdim : Union[str, List[str]], optional
        Key dimensions of the overlay. If `None` then the first key dimension of
        `ovly` will be used. By default `None`.
    spread : bool, optional
        Whether to apply :func:`holoviews.operation.datashader.dynspread` to the
        result. By default `False`.

    Returns
    -------
    hvres : hv.Overlay
        Resulting overlay of datashaded curves and points (for legends).
    """
    if not kdim:
        kdim = ovly.kdims[0].name
    var = np.unique(ovly.dimension_values(kdim)).tolist()
    color_key = [(v, Category10_10[iv]) for iv, v in enumerate(var)]
    color_pts = hv.NdOverlay(
        {k: hv.Points([0, 0], label=str(k)).opts(color=v) for k, v in color_key}
    )
    ds_ovly = datashade(
        ovly,
        aggregator=count_cat(kdim),
        color_key=dict(color_key),
        min_alpha=Datashade.NDCURVE_MIN_ALPHA,
    )
    if spread:
        ds_ovly = dynspread(ds_ovly)
    return ds_ovly * color_pts


def visualize_preprocess(
    fm: xr.DataArray, fn: Optional[Callable] = None, include_org=True, **kwargs
) -> hv.HoloMap:
    """
    Generalized visualization of preprocessing functions.

    This function facilitates parameter exploration of preprocessing functions
    by plotting a single frame before and after the application of the function,
    along with a contour plot. All keyword arguments not listed below are passed
    directly to `fn`.

    Parameters
    ----------
    fm : xr.DataArray
        The input frame.
    fn : Callable, optional
        The function to apply. If `None` then the original frame are visualized
        unchanged. By default `None`.
    include_org : bool, optional
        Whether to include the original frame in the visualization. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        The resulting visualization containing images and contour plots.

    See Also
    --------
    minian.preprocessing
    """
    fh, fw = fm.sizes["height"], fm.sizes["width"]
    asp = fw / fh
    # Type-scoped opts so a Layout with Image + datashaded RGB does not merge ``cmap``
    # onto ``hv.RGB`` (RGB has no cmap; flat ``.opts(cmap=...)`` on the layout does).
    # Do not set ``title`` in ``hv.opts.*`` — Bokeh 3.x can leave ``figure.title`` as a
    # plain ``str`` and HoloViews then errors on ``title.update()``; use ``.relabel()`` for text.
    opts_im = hv.opts.Image(
        frame_width=Preprocess.FRAME_WIDTH,
        aspect=asp,
        cmap=ImagePalette.VIRIDIS_BOKEH,
    )
    opts_cnt_lines = hv.opts.Contours(
        frame_width=Preprocess.FRAME_WIDTH,
        aspect=asp,
        cmap=ImagePalette.VIRIDIS_BOKEH,
    )
    opts_cnt_rgb = hv.opts.RGB(
        frame_width=Preprocess.FRAME_WIDTH,
        aspect=asp,
    )

    def _vis(f):
        im_plot = hv.Image(f, kdims=["width", "height"]).opts(opts_im)
        cnt = (
            hv.operation.contours(im_plot)
            .opts(opts_cnt_lines)
            .relabel(Preprocess.CONTOURS_TITLE)
        )
        im = im_plot.relabel(Preprocess.IMAGE_TITLE)
        return im, cnt

    if fn is not None:
        pkey = kwargs.keys()
        pval = kwargs.values()
        im_dict = dict()
        cnt_dict = dict()
        for params in itt.product(*pval):
            fm_res = fn(fm, **dict(zip(pkey, params)))
            cur_im, cur_cnt = _vis(fm_res)
            cur_im = cur_im.relabel("After")
            cur_cnt = cur_cnt.relabel("After")
            p_str = tuple(
                [str(p) if not isinstance(p, (int, float)) else p for p in params]
            )
            im_dict[p_str] = cur_im
            cnt_dict[p_str] = cur_cnt
        hv_im = Dynamic(hv.HoloMap(im_dict, kdims=list(pkey)).opts(opts_im))
        hv_cnt = datashade(
            hv.HoloMap(cnt_dict, kdims=list(pkey)), precompute=True, cmap=Viridis256
        ).opts(opts_cnt_rgb)
        if include_org:
            im, cnt = _vis(fm)
            im = im.relabel("Before").opts(opts_im)
            cnt = (
                datashade(cnt, precompute=True, cmap=Viridis256)
                .relabel("Before")
                .opts(opts_cnt_rgb)
            )
        return (im + cnt + hv_im + hv_cnt).cols(2)
    else:
        im, cnt = _vis(fm)
        im = im.relabel("Before")
        cnt = cnt.relabel("Before")
        return im + cnt


def visualize_seeds(
    max_proj: xr.DataArray, seeds: pd.DataFrame, mask: Optional[str] = None
) -> hv.Overlay:
    """
    Visualization of seeds.

    This function plot seeds on top of a max projection. It can also visualize
    certain refining step of seeds by coloring the filtered-out seeds in red.

    Parameters
    ----------
    max_proj : xr.DataArray
        Max projection used as the background of the plot.
    seeds : pd.DataFrame
        The seed dataframe.
    mask : str, optional
        The name of the mask of seeds to visualize. If specified, then `seeds`
        must contain a boolean column with the same name. By default `None`.

    Returns
    -------
    hvres : hv.Overlay
        The resulting overlay of seeds and max projection.

    See Also
    --------
    minian.initialization
    """
    h, w = max_proj.sizes["height"], max_proj.sizes["width"]
    asp = w / h
    pt_cmap = {True: Seeds.POINTS_UNMASKED_COLOR, False: Seeds.MASK_FALSE_COLOR}
    opts_im = dict(
        frame_width=Seeds.FRAME_WIDTH,
        aspect=asp,
        cmap=ImagePalette.VIRIDIS_DISPLAY,
    )
    opts_pts = dict(
        frame_width=Seeds.FRAME_WIDTH,
        aspect=asp,
        size=dim("seeds") * 6 + 8,
        tools=["hover"],
        fill_alpha=0.8,
        line_alpha=0,
    )
    if mask:
        vdims = ["seeds", mask]
        opts_pts["color"] = dim(mask)
        opts_pts["cmap"] = pt_cmap
    else:
        vdims = ["seeds"]
        opts_pts["color"] = Seeds.POINTS_UNMASKED_COLOR
    im = hv.Image(max_proj, kdims=["width", "height"])
    pts = hv.Points(seeds, kdims=["width", "height"], vdims=vdims)
    return im.opts(**opts_im) * pts.opts(**opts_pts)


def visualize_gmm_fit(
    values: np.ndarray, gmm: sklearn.mixture.GaussianMixture, bins: int
) -> hv.Overlay:
    """
    Visualization of the Gaussian mixture model fit.

    This function visualize GMM fit by plotting the fitted gaussian curves on
    top of the histograms of values.

    Parameters
    ----------
    values : np.ndarray
        The raw values to which GMM is fitted.
    gmm : sklearn.mixture.GaussianMixture
        The fitted GMM model object.
    bins : int
        Number of bins when plotting the histogram.

    Returns
    -------
    hvres : hv.Overlay
        The resulting visualization.

    See Also
    --------
    minian.initialization.gmm_refine
    """

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    hist = np.histogram(values, bins=bins, density=True)
    gss_dict = dict()
    for igss, (mu, sig) in enumerate(zip(gmm.means_, gmm.covariances_)):
        mu_f = float(np.asarray(mu, dtype=float).squeeze())
        sig_f = float(np.sqrt(np.asarray(sig, dtype=float)).squeeze())
        gss = gaussian(hist[1], mu_f, sig_f)
        gss_dict[igss] = hv.Curve((hist[1], gss))
    return (
        hv.Histogram(((hist[0] - hist[0].min()) / np.ptp(hist[0]), hist[1])).opts(
            fill_alpha=Gmm.HIST_FILL_ALPHA, fill_color=Gmm.HIST_FILL_COLOR
        )
        * hv.NdOverlay(gss_dict)
    ).opts(height=Gmm.FIG_HEIGHT, width=Gmm.FIG_WIDTH)


def _regularize_unit_id_coord_for_image_grid(data: xr.DataArray) -> xr.DataArray:
    """If ``unit_id`` spacing is uneven, remap coords to ``0..N-1`` for HV Image grids."""
    if "unit_id" not in data.coords:
        return data
    uid = np.asarray(data["unit_id"].values)
    if uid.size > 1:
        uid_step = np.diff(uid)
        if not np.allclose(uid_step, uid_step[0], rtol=1e-3, atol=0):
            return data.assign_coords(unit_id=np.arange(uid.size))
    return data


def visualize_spatial_update(
    A_dict: dict,
    C_dict: dict,
    kdims: Optional[Union[str, List[str]]] = None,
    norm=True,
    datashading=True,
) -> hv.HoloMap:
    """
    Visualization of spatial update.

    This function facilitates parameter exploration for spatial update by
    plotting the resulting spatial footprints and binarized spatial footprints
    from different runs of spatial update for a subset of cells, along with
    their corresponding temporal activities.

    Parameters
    ----------
    A_dict : dict
        A dictionary containing resulting spatial footprints from different runs
        of spatial update. Keys should be tuple containing the values of
        parameters that uniquely identify each run. Values should be spatial
        footprints of type `xr.DataArray`.
    C_dict : dict
        A dictionary containing temporal activities of each cells in the same
        format as `A_dict`. The temporal activities of cells are not expected to
        change across different runs of spatial update, except the number of
        cells may be different due to dropping of cells in the update process.
    kdims : Union[str, List[str]], optional
        Names of key dimensions identifying the parameter space. Should have
        same length as the keys in `A_dict` and `C_dict`. If `None` then a
        dimension names "dummy" will be created and the visualization can be
        used to visualize results across cells. By default `None`.
    norm : bool, optional
        Whether to normalize the temporal activities of each cell to range (0,
        1) for visualization. By default `True`.
    datashading : bool, optional
        Whether to apply datashading to temporal activities of cells. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        Resulting visualization.

    See Also
    --------
    minian.cnmf.update_spatial
    """
    if not kdims:
        A_dict = dict(dummy=A_dict)
        C_dict = dict(dummy=C_dict)
    hv_pts_dict, hv_A_dict, hv_Ab_dict, hv_C_dict = (dict(), dict(), dict(), dict())
    for key, A in A_dict.items():
        A = A.compute()
        C = C_dict[key]
        if norm:
            C = xr.apply_ufunc(
                normalize,
                C.chunk(dict(frame=-1)),
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[C.dtype],
            )
        C = C.compute()
        C = _regularize_unit_id_coord_for_image_grid(C)
        h, w = A.sizes["height"], A.sizes["width"]
        cents_df = centroid(A)
        hv_pts_dict[key] = hv.Points(
            cents_df, kdims=["width", "height"], vdims=["unit_id"]
        ).opts(
            tools=["hover"],
            fill_alpha=Spatial.POINTS_FILL_ALPHA,
            line_alpha=0,
            size=Spatial.POINTS_SIZE,
        )
        hv_A_dict[key] = hv.Image(
            A.sum("unit_id").rename("A"), kdims=["width", "height"]
        )
        hv_Ab_dict[key] = hv.Image(
            (A > 0).sum("unit_id").rename("A_bin"), kdims=["width", "height"]
        )
        hv_C_dict[key] = hv.Dataset(C.rename("C")).to(hv.Curve, kdims="frame")
    hv_pts = Dynamic(hv.HoloMap(hv_pts_dict, kdims=kdims))
    hv_A = Dynamic(hv.HoloMap(hv_A_dict, kdims=kdims))
    hv_Ab = Dynamic(hv.HoloMap(hv_Ab_dict, kdims=kdims))
    hv_C = (
        hv.HoloMap(hv_C_dict, kdims=kdims)
        .collate()
        .grid("unit_id")
        .add_dimension("time", 0, 0)
    )
    if datashading:
        hv_C = datashade(hv_C)
    else:
        hv_C = Dynamic(hv_C)
    hv_A = hv_A.opts(
        frame_width=Spatial.IMAGE_FRAME_WIDTH,
        aspect=w / h,
        colorbar=True,
        cmap=ImagePalette.VIRIDIS_BOKEH,
    )
    hv_Ab = hv_Ab.opts(
        frame_width=Spatial.IMAGE_FRAME_WIDTH,
        aspect=w / h,
        colorbar=True,
        cmap=ImagePalette.VIRIDIS_BOKEH,
    )
    hv_C = hv_C.map(
        lambda cr: cr.opts(
            frame_width=Spatial.TEMPORAL_CURVE_FRAME_WIDTH,
            frame_height=Spatial.TEMPORAL_CURVE_FRAME_HEIGHT,
        ),
        hv.RGB if datashading else hv.Curve,
    )
    return hv.NdLayout(
        {"pseudo-color": (hv_pts * hv_A), "binary": (hv_pts * hv_Ab)},
        kdims="Spatial Matrix",
    ).cols(1) + hv_C.relabel("Temporal Components")


def visualize_temporal_update(
    YA_dict: dict,
    C_dict: dict,
    S_dict: dict,
    g_dict: dict,
    sig_dict: dict,
    A_dict: dict,
    kdims: Optional[Union[str, List[str]]] = None,
    norm=True,
    datashading=True,
) -> hv.HoloMap:
    """
    Visualization of temporal update.

    This function facilitates parameter exploration for temporal update by
    plotting various temporal traces along with a model calcium response and the
    spatial footprint for each cell across different runs of temporal update.
    Four traces are plotted: "Raw Signal" correspond to the `YrA` variable,
    "Fitted Calcium Trace" correspond to `C` after update, "Fitted Spikes"
    correspond to `S` after update, and "Fitted Signal" correspond to `C + b0 +
    c0` after update. See :func:`minian.cnmf.update_temporal` for interpretation
    of each variable.

    Parameters
    ----------
    YA_dict : dict
        A dictionary containing the `YrA` variables in the same format as
        `C_dict`. The `YrA` variable is not updated and is not expected to be
        different across different runs of temporal update.
    C_dict : dict
        A dictionary containing resulting calcium traces (`C_new`) from
        different runs of temporal update. Keys should be tuple containing the
        values of parameters that uniquely identify each run. Values should be
        temporal traces of type `xr.DataArray`.
    S_dict : dict
        A dictionary containing resulting deconvolved spike traces (`S_new`)
        from different runs of temporal update, in the same format as `C_dict`.
    g_dict : dict
        A dictionary containing resulting AR coefficients (`g`) from different
        runs of temporal update, in the same format as `C_dict`.
    sig_dict : dict
        A dictionary containing resulting fitted signals (`C_new + b0_new +
        c0_new`) from different runs of temporal update, in the same format as
        `C_dict`.
    A_dict : dict
        A dictionary containing spatial footprint of cells in the same format as
        `C_dict`. The spatial footprints of cells are note expected to change
        across different runs of temporal update, except the number of cells may
        be different due to dropping of cells in the update process.
    kdims : Union[str, List[str]], optional
        Names of key dimensions identifying the parameter space. Should have
        same length as the keys in `C_dict` etc. If `None` then a dimension
        names "dummy" will be created and the visualization can be used to
        visualize results across cells. By default `None`.
    norm : bool, optional
        Whether to normalize the temporal activities of each cell to range (0,
        1) for visualization. By default `True`.
    datashading : bool, optional
        Whether to apply datashading to temporal activities of cells. By default
        `True`.

    Returns
    -------
    hvres : hv.HoloMap
        Resulting visualization.

    See Also
    --------
    minian.cnmf.update_temporal
    """
    inputs = [YA_dict, C_dict, S_dict, sig_dict, g_dict]
    if not kdims:
        inputs = [dict(dummy=i) for i in inputs]
        A_dict = dict(dummy=A_dict)
    input_dict = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A = [dict() for _ in range(7)]
    for k, ins in input_dict.items():
        if norm:
            ins[:-1] = [
                xr.apply_ufunc(
                    normalize,
                    i.chunk(dict(frame=-1)),
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[i.dtype],
                )
                for i in ins[:-1]
            ]
        ins[:] = [i.compute() for i in ins]
        ya, c, s, sig, g = ins
        f_crd = ya.coords["frame"]
        pul_crd = f_crd.values[: Temporal.PULSE_PREVIEW_LEN]
        s_pul, c_pul = xr.apply_ufunc(
            construct_pulse_response,
            g,
            input_core_dims=[["lag"]],
            output_core_dims=[["t"], ["t"]],
            vectorize=True,
            kwargs=dict(length=len(pul_crd)),
            output_sizes=dict(t=len(pul_crd)),
        )
        s_pul, c_pul = (s_pul.assign_coords(t=pul_crd), c_pul.assign_coords(t=pul_crd))
        if norm:
            c_pul = xr.apply_ufunc(
                normalize,
                c_pul.chunk(dict(t=-1)),
                input_core_dims=[["t"]],
                output_core_dims=[["t"]],
                dask="parallelized",
                output_dtypes=[c_pul.dtype],
            ).compute()
        pul_range = (
            f_crd.min(),
            int(np.around(f_crd.min() + (f_crd.max() - f_crd.min()) / 2)),
        )
        hv_S_pul[k], hv_C_pul[k] = [
            (hv.Dataset(tr.rename("Response (A.U.)")).to(hv.Curve, kdims=["t"]))
            for tr in [s_pul, c_pul]
        ]
        hv_YA[k] = hv.Dataset(ya.rename("Intensity (A.U.)")).to(
            hv.Curve, kdims=["frame"]
        )
        if c.sizes["unit_id"] > 0:
            hv_C[k], hv_S[k], hv_sig[k] = [
                (
                    hv.Dataset(tr.rename("Intensity (A.U.)")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                for tr in [c, s, sig]
            ]
        hv_A[k] = hv.Dataset(A_dict[k].rename("A")).to(
            hv.Image, kdims=["width", "height"]
        )
        h, w = A_dict[k].sizes["height"], A_dict[k].sizes["width"]
    hvobjs = [hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A]
    hvobjs[:] = [hv.HoloMap(hvobj, kdims=kdims).collate() for hvobj in hvobjs]
    hv_unit = {
        "Raw Signal": hvobjs[0],
        "Fitted Calcium Trace": hvobjs[1],
        "Fitted Spikes": hvobjs[2],
        "Fitted Signal": hvobjs[3],
    }
    hv_pul = {"Simulated Calcium": hvobjs[4], "Simulated Spike": hvobjs[5]}
    hv_unit = hv.HoloMap(hv_unit, kdims="traces").collate().overlay("traces")
    hv_pul = hv.HoloMap(hv_pul, kdims="traces").collate().overlay("traces")
    hv_A = Dynamic(hvobjs[6])
    if datashading:
        hv_unit = datashade_ndcurve(hv_unit, "traces")
    else:
        hv_unit = Dynamic(hv_unit)
    hv_pul = Dynamic(hv_pul)
    hv_unit = hv_unit.map(
        lambda p: p.opts(
            frame_height=Temporal.UNIT_MAP_FRAME_HEIGHT,
            frame_width=Temporal.UNIT_MAP_FRAME_WIDTH,
        )
    )
    hv_pul = hv_pul.opts(
        frame_width=Temporal.SPATIAL_FOOTPRINT_FRAME_WIDTH, aspect=w / h
    ).redim(t=hv.Dimension("t", soft_range=pul_range))
    hv_A = hv_A.opts(
        frame_width=Temporal.SPATIAL_FOOTPRINT_FRAME_WIDTH,
        aspect=w / h,
        cmap=ImagePalette.VIRIDIS_DISPLAY,
    )
    return (
        hv_unit.relabel("Current Unit: Temporal Traces")
        + hv.NdLayout(
            {"Simulated Pulse Response": hv_pul, "Spatial Footprint": hv_A},
            kdims="Current Unit",
        )
    ).cols(1)


def visualize_motion(motion: xr.DataArray) -> Union[hv.Layout, hv.NdOverlay]:
    """
    Visualize result of motion estimation.

    This function plot motions across time. If the input has two dimensions,
    they are interpreted as rigid shifts along the "height" and "width"
    dimension of the movie, and plotted as curves across time. If the input has
    more than two dimensions, it is assumed that non-rigid motion estimation was
    enabled and each frame is split into several patches that will each have
    their own shifts. The separate shifts for patches within each frame are
    flattened into a column, then shifts along "height" and "width" dimensions
    are separately plotted as 2d images across time, whose columns represent
    frames and colors represent degree of shift.

    Parameters
    ----------
    motion : xr.DataArray
        Estimated motion.

    Returns
    -------
    Union[hv.Layout, hv.NdOverlay]
        If `motion` contains rigid shifts, then an overlay of two curves are
        returned. Otherwise two images representing non-rigid motions are
        returned.
    """
    if motion.ndim > 2:
        opts_im = {
            "frame_width": Motion.IMAGE_FRAME_WIDTH,
            "aspect": Motion.IMAGE_ASPECT,
            "cmap": Motion.DIVERGING_CMAP,
            "symmetric": True,
            "colorbar": True,
        }
        mheight = motion.sel(shift_dim="height").stack(grid=["grid0", "grid1"])
        mwidth = motion.sel(shift_dim="width").stack(grid=["grid0", "grid1"])
        mheight = mheight.assign_coords(grid=np.arange(mheight.sizes["grid"]))
        mwidth = mwidth.assign_coords(grid=np.arange(mwidth.sizes["grid"]))
        return (
            hv.Image(mheight.rename("height_motion"), kdims=["frame", "grid"])
            .opts(**opts_im)
            .relabel("height_motion")
            + hv.Image(mwidth.rename("width_motion"), kdims=["frame", "grid"])
            .opts(**opts_im)
            .relabel("width_motion")
        ).cols(1)
    else:
        opts_cv = {
            "frame_width": Motion.CURVE_FRAME_WIDTH,
            "tools": ["hover"],
            "aspect": Motion.CURVE_ASPECT,
        }
        return hv.NdOverlay(
            dict(
                width=hv.Curve(motion.sel(shift_dim="width")).opts(**opts_cv),
                height=hv.Curve(motion.sel(shift_dim="height")).opts(**opts_cv),
            )
        )
