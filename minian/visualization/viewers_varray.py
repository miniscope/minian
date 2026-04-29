"""Interactive viewer for movie / VArray data."""

import functools as fct
import itertools as itt
import logging
from collections import OrderedDict
from typing import List, Union

import holoviews as hv
import panel as pn
import param
import xarray as xr
from holoviews.streams import BoxEdit, RangeXY, Stream
from panel import widgets as pnwgt

from ._viz_constants import ImagePalette, PanelLayout, Player, SummaryStat, VArray
from ._viewer_helpers import (
    build_meta_select_widgets,
    wire_frame_player_index,
    wire_meta_select_widgets,
)
from .pipeline_plots import datashade_ndcurve

log = logging.getLogger(__name__)


class VArrayViewer:
    """
    Interactive visualization for movie data arrays.

    Hint
    ----
    .. figure:: img/vaviewer.png
        :width: 500px
        :align: left

    The visualization contains following panels from top to bottom:

    Play Toolbar
        A toolbar that controls playback of the video. Additionally, when the
        button "Update Mask" is clicked, the coordinates of the box drawn in
        *Current Frame* panel will be used to update the `mask` attribute of the
        `VArrayViewer` instance, which can be later used to subset the data. If
        multiple arrays are visualized and `layout` is `False`, then drop-down
        lists corresponding to each metadata dimensions will show up so the user
        can select which array to visualize.
    Current Frame
        Images of the current frame. If multiple movie array are passed in,
        multiple frames will be labeled and shown. To the side of each frame
        there is a histogram of intensity values. The "Box Select" tool can be
        used on the histogram to limit the range of intensity used for
        color-mapping. Additionally, the "Box Edit Tool" is available for use on
        the frame image, where you can hold "Shift" and draw a box, whose
        coordinates can be used to update the `mask` attribute of the
        `VarrayViewer` instance (remember to click "Update Mask" after drawing).
    Summary
        Summary statistics of each frame across time. Only shown if `summary` is
        not empty. The red vertical line indicate current frame.

    Attributes
    ----------
    mask : dict
        Instance attribute that can be retrieved and used to subset data later.
        Keys are `tuple` with values corresponding to each `meta_dims` and
        uniquely identify each input array. If `meta_dims` is empty then keys
        will be empty `tuple` as well. Values are `dict` mapping dimension names
        (of the arrays) to subsetting slices. The slices are in the plotting
        coorandinates and can be directly passed to `xr.DataArray.sel` method to
        subset data.
    """

    def __init__(
        self,
        varr: Union[xr.DataArray, List[xr.DataArray], xr.Dataset],
        framerate=30,
        summary=["mean"],
        meta_dims: List[str] = None,
        datashading=True,
        layout=False,
    ):
        """
        Parameters
        ----------
        varr : Union[xr.DataArray, List[xr.DataArray], xr.Dataset]
            Input array, list of arrays, or dataset to be visualized. Each array
            should contain dimensions "height", "width" and "frame". If a
            dataset, then the dimensions specified in `meta_dims` will be used
            as metadata dimensions that can uniquely identify each array. If a
            list, then a dimension "data_var" will be constructed and used as
            metadata dimension, and the `.name` attribute of each array will be
            used to identify each array.
        framerate : int, optional
            The framerate of playback when using the toolbar. By default `30`.
        summary : list, optional
            List of summary statistics to plot. The statistics should be one of
            `{"mean", "max", "min", "diff"}`. By default `["mean"]`.
        meta_dims : List[str], optional
            List of dimension names that can uniquely identify each input array
            in `varr`. Only used if `varr` is a `xr.Dataset`. By default `None`.
        datashading : bool, optional
            Whether to use datashading on the summary statistics. By default
            `True`.
        layout : bool, optional
            Whether to visualize all arrays together as layout. If `False` then
            only one array will be visualized and user can switch array using
            drop-down lists below the *Play Toolbar*. By default `False`.

        Raises
        ------
        NotImplementedError
            if `varr` is not a `xr.DataArray`, a `xr.Dataset` or a list of `xr.DataArray`
        """
        if isinstance(varr, list):
            for iv, v in enumerate(varr):
                varr[iv] = v.assign_coords(data_var=v.name)
            self.ds = xr.concat(varr, dim="data_var")
            meta_dims = ["data_var"]
        elif isinstance(varr, xr.DataArray):
            self.ds = varr.to_dataset()
        elif isinstance(varr, xr.Dataset):
            self.ds = varr
        else:
            raise NotImplementedError(
                "video array of type {} not supported".format(type(varr))
            )
        try:
            self.meta_dicts = OrderedDict(
                [(d, list(self.ds.coords[d].values)) for d in meta_dims]
            )
            self.cur_metas = OrderedDict(
                [(d, v[0]) for d, v in self.meta_dicts.items()]
            )
        except TypeError:
            self.meta_dicts = dict()
            self.cur_metas = dict()
        self._datashade = datashading
        self._layout = layout
        self.framerate = framerate
        self._f = self.ds.coords["frame"].values
        self._h = self.ds.sizes["height"]
        self._w = self.ds.sizes["width"]
        self.mask = dict()
        CStream = Stream.define(
            "CStream",
            f=param.Integer(
                default=int(self._f.min()), bounds=(self._f.min(), self._f.max())
            ),
        )
        self.strm_f = CStream()
        self.str_box = BoxEdit()
        self.widgets = self._widgets()
        if type(summary) is list:
            summ_all = {
                SummaryStat.MEAN: self.ds.mean(["height", "width"]),
                SummaryStat.MAX: self.ds.max(["height", "width"]),
                SummaryStat.MIN: self.ds.min(["height", "width"]),
                SummaryStat.DIFF: self.ds.diff("frame").mean(["height", "width"]),
            }
            summ = None
            try:
                summ = {k: summ_all[SummaryStat(k)] for k in summary}
            except (KeyError, ValueError):
                log.warning("{} Not understood for specifying summary".format(summary))
            if summ:
                log.info("computing summary")
                sum_list = []
                for k, v in summ.items():
                    sum_list.append(v.compute().assign_coords(sum_var=k))
                summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        if layout:
            self.ds_sub = self.ds
            self.sum_sub = self.summary
        else:
            self.ds_sub = self.ds.sel(**self.cur_metas)
            try:
                self.sum_sub = self.summary.sel(**self.cur_metas)
            except AttributeError:
                self.sum_sub = self.summary
        ims, summ_hv = self._build_movie_summary()
        self._pane_movie = pn.pane.HoloViews(ims, sizing_mode=PanelLayout.SIZING_STRETCH_WIDTH)
        if summ_hv is None:
            self.pnplot = pn.Column(self._pane_movie)
        else:
            self._pane_summ = pn.pane.HoloViews(
                summ_hv, sizing_mode=PanelLayout.SIZING_STRETCH_WIDTH
            )
            self.pnplot = pn.Column(self._pane_movie, self._pane_summ)

    def _build_movie_summary(self):
        def get_im_ovly(meta):
            def img(f, ds):
                return hv.Image(ds.sel(frame=f).compute(), kdims=["width", "height"])

            try:
                curds = self.ds_sub.sel(**meta).rename("_".join(meta.values()))
            except ValueError:
                curds = self.ds_sub
            fim = fct.partial(img, ds=curds)
            im = hv.DynamicMap(fim, streams=[self.strm_f]).opts(
                frame_width=VArray.FRAME_WIDTH,
                aspect=self._w / self._h,
                cmap=ImagePalette.VIRIDIS_DISPLAY,
            )
            self.xyrange = RangeXY(source=im).rename(x_range="w", y_range="h")
            if not self._layout:
                hv_box = hv.Polygons([]).opts(
                    fill_alpha=VArray.POLYGON_FILL_ALPHA,
                    line_color=VArray.POLYGON_LINE_COLOR,
                )
                self.str_box = BoxEdit(source=hv_box)
                im_ovly = im * hv_box
            else:
                im_ovly = im

            def hist(f, w, h, ds):
                if w and h:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    ).select(height=h, width=w)
                else:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    )
                return hv.operation.histogram(
                    cur_im, num_bins=VArray.HIST_NUM_BINS
                ).opts(xlabel=VArray.HIST_XLABEL, ylabel=VArray.HIST_YLABEL)

            fhist = fct.partial(hist, ds=curds)
            his = hv.DynamicMap(fhist, streams=[self.strm_f, self.xyrange]).opts(
                frame_height=int(VArray.FRAME_WIDTH * self._h / self._w),
                width=VArray.HIST_SIDE_WIDTH,
                cmap=ImagePalette.VIRIDIS_DISPLAY,
            )
            # Image and histogram already set cmap; do not .map cmap onto AdjointLayout/Overlay.
            im_ovly = im_ovly << his
            return im_ovly

        if self._layout and self.meta_dicts:
            im_dict = OrderedDict()
            for meta in itt.product(*list(self.meta_dicts.values())):
                mdict = {k: v for k, v in zip(list(self.meta_dicts.keys()), meta)}
                im_dict[meta] = get_im_ovly(mdict)
            ims = hv.NdLayout(im_dict, kdims=list(self.meta_dicts.keys()))
        else:
            ims = get_im_ovly(self.cur_metas)
        if self.summary is not None:
            hvsum = (
                hv.Dataset(self.sum_sub)
                .to(hv.Curve, kdims=["frame"])
                .overlay("sum_var")
            )
            if self._datashade:
                hvsum = datashade_ndcurve(hvsum, kdim="sum_var")
            try:
                hvsum = hvsum.layout(list(self.meta_dicts.keys()))
            except Exception:
                pass
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f]).opts(
                color=VArray.VLINE_COLOR
            )
            summ_plot = hvsum * vl
            if self._datashade:
                summ = summ_plot.opts(
                    hv.opts.RGB(
                        frame_width=VArray.FRAME_WIDTH,
                        frame_height=max(
                            VArray.SUMMARY_RGB_HEIGHT_FLOOR,
                            int(VArray.FRAME_WIDTH / VArray.SUMMARY_ASPECT),
                        ),
                    ),
                )
            else:
                summ = summ_plot.opts(
                    hv.opts.Curve(
                        frame_width=VArray.FRAME_WIDTH,
                        aspect=VArray.SUMMARY_ASPECT,
                    )
                )
            # Two separate Panel HoloViews panes (see __init__): do not use
            # ``(ims + summ).cols(1)`` — HoloViews merges Image cmap into the
            # summary RGB overlay and triggers cmap-on-RGB errors.
            return ims, summ
        return ims, None

    def show(self) -> pn.layout.Column:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolbars.
        """
        return pn.layout.Column(self.widgets, self.pnplot)

    def _widgets(self):
        w_play = pnwgt.Player(
            length=len(self._f),
            interval=Player.INTERVAL_MS,
            value=0,
            width=Player.WIDTH,
            height=Player.HEIGHT,
        )
        wire_frame_player_index(
            w_play, lambda i: self.strm_f.event(f=int(self._f[i]))
        )
        w_box = pnwgt.Button(
            name="Update Mask",
            button_type="primary",
            width=Player.UPDATE_MASK_BUTTON_WIDTH,
            height=Player.UPDATE_MASK_BUTTON_HEIGHT,
        )
        w_box.param.watch(self._update_box, "clicks")
        if not self._layout:
            wgt_meta = build_meta_select_widgets(self.meta_dicts)
            wire_meta_select_widgets(wgt_meta, self.cur_metas, self._update_subs)
            wgts = pn.layout.WidgetBox(w_box, w_play, *list(wgt_meta.values()))
        else:
            wgts = pn.layout.WidgetBox(w_box, w_play)
        return wgts

    def _update_subs(self):
        self.ds_sub = self.ds.sel(**self.cur_metas)
        if self.sum_sub is not None:
            self.sum_sub = self.summary.sel(**self.cur_metas)
        ims, summ_hv = self._build_movie_summary()
        self._pane_movie.object = ims
        if summ_hv is not None and getattr(self, "_pane_summ", None) is not None:
            self._pane_summ.object = summ_hv

    def _update_box(self, click):
        box = self.str_box.data
        if not isinstance(box, dict):
            log.warning("Update Mask: no box data; draw on the frame first.")
            return
        xs0, xs1, ys0, ys1 = (
            box.get("x0") or [],
            box.get("x1") or [],
            box.get("y0") or [],
            box.get("y1") or [],
        )
        if not (xs0 and xs1 and ys0 and ys1):
            log.warning(
                "Update Mask: use Box Edit on the movie frame (Shift+drag), then click again."
            )
            return
        self.mask.update(
            {
                tuple(self.cur_metas.values()): {
                    "height": slice(ys0[0], ys1[0]),
                    "width": slice(xs0[0], xs1[0]),
                }
            }
        )
