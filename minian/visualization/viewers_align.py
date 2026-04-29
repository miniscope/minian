"""Interactive cross-registration alignment viewer."""

import logging

import colorcet as cc
import cv2
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from matplotlib import cm
from matplotlib.colors import ListedColormap
from panel import widgets as pnwgt

import holoviews as hv

from ..motion_correction import apply_shifts
from ..utilities import rechunk_like

from ._numeric import norm

log = logging.getLogger(__name__)

# AlignViewer RGB channels: colorcet black → saturated hue ramps (kr, kg, kb).
_ALIGN_CHANNEL_CMAPS = {
    "r": ListedColormap(cc.kr),
    "g": ListedColormap(cc.kg),
    "b": ListedColormap(cc.kb),
}


class AlignViewer:
    """
    Interactive visualization of cross-registration resuls.

    Hint
    ----
    .. image:: img/alignviewer.png
        :width: 700px

    This class visualize the result of cross-registration by color-mapping
    spatial footprints of cells from three selected sessions as red, green and
    blue channel and show an overlay image. In addition to the overlay image,
    following tools are available:

    Channel Selector
        Contains "sessionR", "sessionG", and "sessionB" dropdowns, allowing the
        user to select which sessions are colormapped to each channel.
    Display Settings
        Contains the following tools:

        * "erode" dropdown, set window size of an optional erode operation
          applied to the spatial footprints for display to reduce overlaps.
        * "show matched" and "show unmatched" checkboxes, set whether to show
          cells that are matched or not matched across all three selected sessions.
    Metadata Selector
        If additional metadata are present, dropdowns corresponding to each
        metadata dimensions will be shown.

    """

    def __init__(
        self,
        minian_ds: xr.Dataset,
        cents: pd.DataFrame,
        mappings: pd.DataFrame,
        shiftds: xr.Dataset,
        brt_offset=0,
    ) -> None:
        """
        Parameters
        ----------
        minian_ds : xr.Dataset
            Input dataset. Should contain `minian_ds["A"]`.
        cents : pd.DataFrame
            Input centroids of cells.
        mappings : pd.DataFrame
            Input mappings of cells.
        shiftds : xr.Dataset
            Input dataset of shift results. Should contain `shiftds["shifts"]`.
        brt_offset : int, optional
            Brightness offset added on top of the color-mapped image. Useful to
            make the image visually brighter. By default `0`.
        """
        # init
        self.minian_ds = minian_ds
        self.cents = cents
        self.mappings = mappings
        self.shiftds = shiftds
        self.brt_offset = brt_offset
        A = self.minian_ds["A"]
        self.shifts = rechunk_like(self.shiftds["shifts"], A)
        self.Ash = apply_shifts(A, self.shifts, fill=0)
        # option widgets
        self.erode = 3
        wgt_er = pnwgt.Select(name="erode", options=np.arange(0, 20).tolist(), value=3)
        wgt_er.param.watch(self.cb_update_erd, "value")
        self.show_ma = True
        wgt_ma = pnwgt.Checkbox(name="show matched", value=True)
        wgt_ma.param.watch(self.cb_showma, "value")
        self.show_uma = True
        wgt_uma = pnwgt.Checkbox(name="show unmatched", value=True)
        wgt_uma.param.watch(self.cb_showuma, "value")
        self.wgt_opt = pn.layout.WidgetBox(wgt_er, wgt_ma, wgt_uma)
        self.processA()
        # handling meta
        try:
            self.meta_dict = {
                col: c.unique().tolist() for col, c in mappings["meta"].iteritems()
            }
        except KeyError:
            self.meta_dict = None
        if self.meta_dict:
            self.meta = {d: v[0] for d, v in self.meta_dict.items()}
            wgt_meta = [
                pnwgt.Select(name=dim, options=vals)
                for dim, vals in self.meta_dict.items()
            ]
            for w in wgt_meta:
                w.param.watch(lambda v, n=w.name: self.cb_update_meta(n, v), "value")
            self.wgt_meta = pn.layout.WidgetBox(*wgt_meta)
        else:
            self.wgt_meta = None
        self.update_meta()
        # sessionRGB
        sess = list(mappings["session"].columns)
        self.sess_rgb = {"r": sess[0], "g": sess[0], "b": sess[0]}
        wgt_sess = {
            c: pnwgt.Select(name="session{}".format(c.upper()), options=sess)
            for c in ["r", "g", "b"]
        }
        for wname, w in wgt_sess.items():
            w.param.watch(lambda v, n=wname: self.cb_update_rgb(n, v), "value")
        self.wgt_rgb = pn.layout.WidgetBox(*list(wgt_sess.values()))
        self.plot = self.update_plot()

    def processA(self):
        A = self.Ash
        if self.erode >= 3:
            A = xr.apply_ufunc(
                cv2.erode,
                A,
                input_core_dims=[["height", "width"]],
                output_core_dims=[["height", "width"]],
                vectorize=True,
                dask="parallelized",
                kwargs={"kernel": np.ones((self.erode, self.erode))},
                output_dtypes=[float],
            )
        self.dataA = xr.apply_ufunc(
            norm,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def update_plot(self):
        Adict = {
            c: self.curA.sel(session=self.sess_rgb[c])
            .dropna("unit_id", how="all")
            .compute()
            for c in self.sess_rgb.keys()
        }
        map_sub = self.curmap["session"][list(self.sess_rgb.values())].dropna(how="all")
        map_sub = map_sub.loc[:, ~map_sub.columns.duplicated()]
        ma_mask = map_sub.notnull().all(axis="columns")
        imdict = {
            c: np.zeros((A.sizes["height"], A.sizes["width"])) for c, A in Adict.items()
        }
        if self.show_ma:
            ma_map = map_sub.loc[ma_mask]
            for c, im in imdict.items():
                uids = ma_map[self.sess_rgb[c]].values
                imdict[c] = im + Adict[c].sel(unit_id=uids).sum("unit_id").compute()
        if self.show_uma:
            uma_map = map_sub.loc[~ma_mask]
            for c, im in imdict.items():
                uids = uma_map[self.sess_rgb[c]].dropna().values
                imdict[c] = im + Adict[c].sel(unit_id=uids).sum("unit_id").compute()
        for c, im in imdict.items():
            imdict[c] = cm.ScalarMappable(cmap=_ALIGN_CHANNEL_CMAPS[c]).to_rgba(im)
        im_ovly = xr.DataArray(
            np.clip(imdict["r"] + imdict["g"] + imdict["b"] + self.brt_offset, 0, 1),
            dims=["height", "width", "rgb"],
            coords={
                "height": self.curA.coords["height"].values,
                "width": self.curA.coords["width"].values,
            },
        )
        im_opts = {
            "frame_height": self.curA.sizes["height"],
            "frame_width": self.curA.sizes["width"],
        }
        return pn.panel(
            hv.RGB(
                (
                    im_ovly.coords["width"],
                    im_ovly.coords["height"],
                    im_ovly[:, :, 0],
                    im_ovly[:, :, 1],
                    im_ovly[:, :, 2],
                    im_ovly[:, :, 3],
                ),
                kdims=["width", "height"],
            ).opts(**im_opts)
        )

    def update_meta(self):
        if self.meta_dict:
            self.curA = self.dataA.sel(**self.meta).persist()
            self.curmap = (
                self.mappings.set_index([("meta", d) for d in self.meta.keys()])
                .loc[tuple(self.meta.values())]
                .reset_index()
            )
        else:
            self.curA = self.dataA.persist()
            self.curmap = self.mappings

    def cb_update_erd(self, val):
        self.erode = val.new
        self.processA()
        self.update_meta()
        self.plot.object = self.update_plot().object

    def cb_update_meta(self, dim, val):
        self.meta[dim] = val.new
        self.update_meta()
        self.plot.object = self.update_plot().object

    def cb_update_rgb(self, ch, ss):
        self.sess_rgb[ch] = ss.new
        self.plot.object = self.update_plot().object

    def cb_showma(self, val):
        self.show_ma = val.new
        self.plot.object = self.update_plot().object

    def cb_showuma(self, val):
        self.show_uma = val.new
        self.plot.object = self.update_plot().object

    def show(self) -> pn.layout.Row:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Row
            Resulting visualizations containing both plots and toolbars.
        """
        return pn.layout.Row(
            self.plot, pn.layout.Column(self.wgt_meta, self.wgt_rgb, self.wgt_opt)
        )
