"""Interactive CNMF result viewer."""

import functools as fct
import logging
from collections import OrderedDict
from typing import Optional

import dask.array as da
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr
from holoviews.streams import DoubleTap, Pipe, Selection1D, Stream
from panel import widgets as pnwgt

from ._numeric import NNsort, centroid, normalize_along_frame_per_unit
from ._viewer_helpers import (
    build_meta_select_widgets,
    footprint_image_opts,
    wire_frame_player_index,
    wire_meta_select_widgets,
)
from .pipeline_plots import datashade_ndcurve

log = logging.getLogger(__name__)


class CNMFViewer:
    """
    Interactive visualization for CNMF results.

    Hint
    ----
    .. figure:: img/cnmfviewer.png
        :width: 1000px

    The visualization can be divided into two parts vertically:

    Spatial
        Top part of the visualization. Shows spatial plots at a given time. From
        left to right:

        Spatial Footprints
            Shows the spatial footprints of all cells. The "Box Select" tool can
            be used in this panel to select a subset of cells to visualize for
            both the *Isolated Activities* panel and the *Temporal Activities*
            panel.
        Isolated Activities
            Shows activities of selected cells only. If the "UseAC" checkbox
            under *General Toolbox* is enabled, then the `AtC` variable computed
            with the selected cells will be visualized at the given frame (See
            :func:`minian.cnmf.compute_AtC`). Otherwise the spatial footprints
            of the cells will be plotted, which would be invariant across time.
            The "unit_id" coordinates for each cell are shown on top of each
            cell.
        Original Movie
            Shows a single frame of an arbitrary movie data supplied in `org`.

    Temporal
        Bottom part of the visualization. Shows temporal activities across time
        and various toolboxes. From left to right:

        General Toolbox
            Contains the following tools:

            * "Refresh" button, will refresh all visualization when clicked.
            * "Load Data" button, will load all data in memory for faster
              visualization, can be very memory-demanding.
            * "UseAC" checkbox, whether to plot spatial-temporal activities for
              the *Isolated Activities* panel.
            * "ShowC", "ShowS", "Normalize" checkboxes, whether to show the
              calcium traces, the spike signals, or to normalize both traces
              to unit range for each cell.
            * "Group" dropbox, "Previous Group" and "Next Group" buttons, select
              the group of cells to visualize. The grouping is controlled by
              `sortNN` parameter.
            * Playback toolbar, used to control which timepoint is visualized.
            * Additional metadata dropdown, if the input dataset contains
              additional metadata dimensions then dropdown will show up so
              user can select which dataset to visualize.
        Temporal Activities
            Shows temporal activities of selected subset of cells. The red
            vertical line indicate current frame. Additionally user can
            double-click anywhere in the plot to move current frame to that
            location.
        Manual Label
            Shows tools to carry out manual labeling of cells. User can either
            manually assign unit label using the dropdown for each cell, or
            select some cells with the checkboxes corresponding to the
            "unit_id", and then merge or discard the units using the buttons.
            The "Unit Label" dropdowns should update and refelect the merging or
            discarding actions.

    Attributes
    ----------
    unit_labels : xr.DataArray
        1d array whose values represent the result of manual refinement of
        cells. The "unit_id" coordinate of this array is identical to input
        data. The values of this array can be interpreted as new "unit_id" after
        the manual refinement, where duplicated values indicate merged cells,
        and values of -1 indicate discarded cells.
    """

    def __init__(
        self,
        minian: Optional[xr.Dataset] = None,
        A: Optional[xr.DataArray] = None,
        C: Optional[xr.DataArray] = None,
        S: Optional[xr.DataArray] = None,
        org: Optional[xr.DataArray] = None,
        sortNN=True,
    ):
        """
        Parameters
        ----------
        minian : xr.Dataset, optional
            Input minian dataset containing all necessary variables. If `None`
            then all other arguments should be supplied. By default `None`.
        A : xr.DataArray, optional
            Spatial footprints of cells. If `None` then it will be retrieved as
            `minian["A"]`. By default `None`.
        C : xr.DataArray, optional
            Calcium dynamic of cells. If `None` then it will be retrieved as
            `minian["C"]`. By default `None`.
        S : xr.DataArray, optional
            Deconvolved spikes of cells. If `None` then it will be retrieved as
            `minian["S"]`. By default `None`.
        org : xr.DataArray, optional
            Arbitrary movie data to be visualized along with results of CNMF. If
            `None` then it will be retrieved as `minian["org"]`. If this array
            contains dimensions other than "height", "width" or "frame" then
            they will be used as metadata dimensions. By default `None`.
        sortNN : bool, optional
            Whether to sort the units using :func:`NNsort` so that cells close
            together will appear in same group for visualization. If `False`
            then cells are simply grouped in 5 by ascending "unit_id". By
            default `True`.
        """
        self._init_core_arrays(minian, A, C, S, org)
        self._init_unit_labels(minian)
        self._init_trace_normalizations()
        self._init_spatial_aggregates()
        self._init_view_flags(sortNN)
        self._init_org_metadata_selectors()
        self._init_nn_sort_coords_if_enabled()
        self.update_subs()
        self._init_holostreams_pipes_and_movie()
        self._init_panel_children()

    def _init_core_arrays(
        self,
        minian: Optional[xr.Dataset],
        A: Optional[xr.DataArray],
        C: Optional[xr.DataArray],
        S: Optional[xr.DataArray],
        org: Optional[xr.DataArray],
    ) -> None:
        self._A = A if A is not None else minian["A"]
        self._C = C if C is not None else minian["C"]
        self._S = S if S is not None else minian["S"]
        self._org = org if org is not None else minian["org"]

    def _init_unit_labels(self, minian: Optional[xr.Dataset]) -> None:
        if minian is not None and "unit_labels" in minian:
            self.unit_labels = minian["unit_labels"].compute()
        else:
            self.unit_labels = xr.DataArray(
                self._A["unit_id"].values.copy(),
                dims=self._A["unit_id"].dims,
                coords=self._A["unit_id"].coords,
            ).rename("unit_labels")

    def _init_trace_normalizations(self) -> None:
        self._C_norm = normalize_along_frame_per_unit(
            self._C, output_dtype=self._C.dtype
        )
        self._S_norm = normalize_along_frame_per_unit(
            self._S, output_dtype=self._C.dtype
        )

    def _init_spatial_aggregates(self) -> None:
        self.cents = centroid(self._A, verbose=True)
        log.info("computing sum projection")
        self.Asum = self._A.sum("unit_id").compute()

    def _init_view_flags(self, sortNN: bool) -> None:
        self._NNsort = sortNN
        self._normalize = False
        self._useAC = True
        self._showC = True
        self._showS = True

    def _init_org_metadata_selectors(self) -> None:
        self._meta_dims = list(set(self._org.dims) - {"frame", "height", "width"})
        self.meta_dicts = {d: list(self._org.coords[d].values) for d in self._meta_dims}
        self.metas = {d: v[0] for d, v in self.meta_dicts.items()}

    def _init_nn_sort_coords_if_enabled(self) -> None:
        if not self._NNsort:
            return
        try:
            self.cents["NNord"] = self.cents.groupby(
                self._meta_dims, group_keys=False
            ).apply(NNsort)
        except ValueError:
            self.cents["NNord"] = NNsort(self.cents)
        nn_coords = self.cents.set_index(self._meta_dims + ["unit_id"])[
            "NNord"
        ].to_xarray()
        self._A = self._A.assign_coords(NNord=nn_coords)
        self._C = self._C.assign_coords(NNord=nn_coords)
        self._S = self._S.assign_coords(NNord=nn_coords)
        self._C_norm = self._C_norm.assign_coords(NNord=nn_coords)
        self._S_norm = self._S_norm.assign_coords(NNord=nn_coords)

    def _init_holostreams_pipes_and_movie(self) -> None:
        # stream for frame index
        self.strm_f = DoubleTap(rename=dict(x="f"))
        self.strm_f.add_subscriber(self.callback_f)
        # stream for unit index
        self.strm_uid = Selection1D()
        self.strm_uid.add_subscriber(self.callback_uid)
        # stream for subset of units
        stream_usub = Stream.define("Stream_usub", usub=param.List())
        self.strm_usub = stream_usub()
        self.strm_usub.add_subscriber(self.callback_usub)
        self.usub_sel = self.strm_usub.usub
        # _AC: spatial-temporal activities
        self._AC = self._org.sel(**self.metas)
        # _mov: original movie
        self._mov = self._org.sel(**self.metas)
        # pipAC
        self.pipAC = Pipe([])
        self.pipmov = Pipe([])
        self.pipusub = Pipe([])

    def _init_panel_children(self) -> None:
        self.wgt_meta = self._meta_wgt()
        self.wgt_spatial_all = self._spatial_all_wgt()
        self.spatial_all = self._spatial_all()
        self.temp_comp_sub = self._temp_comp_sub(self._u[:5])
        self.wgt_man = self._man_wgt()
        self.wgt_temp_comp = self._temp_comp_wgt()

    def update_subs(self) -> None:
        """Align all subset views with ``self.metas`` and refresh derived UI coordinates.

        Steps: slice full arrays → optional NN walk order → footprint/time/unit
        coordinates for widgets → centroids row-filter for overlays.
        """
        self._slice_arrays_to_current_metadata()
        self._sort_subviews_by_nn_when_enabled()
        self._derive_ui_axis_coordinates()
        self._filter_centroids_to_current_metadata()

    def _slice_arrays_to_current_metadata(self) -> None:
        m = self.metas
        self.A_sub = self._A.sel(**m)
        self.C_sub = self._C.sel(**m)
        self.S_sub = self._S.sel(**m)
        self.org_sub = self._org.sel(**m)
        self.C_norm_sub = self._C_norm.sel(**m)
        self.S_norm_sub = self._S_norm.sel(**m)

    def _sort_subviews_by_nn_when_enabled(self) -> None:
        if not self._NNsort:
            return
        self.A_sub = self.A_sub.sortby("NNord")
        self.C_sub = self.C_sub.sortby("NNord")
        self.S_sub = self.S_sub.sortby("NNord")
        self.C_norm_sub = self.C_norm_sub.sortby("NNord")
        self.S_norm_sub = self.S_norm_sub.sortby("NNord")

    def _derive_ui_axis_coordinates(self) -> None:
        """Footprint height/width grids, frame axis, and unit list (from first cell / slice)."""
        ref_unit = self.A_sub.isel(unit_id=0)
        self._h = ref_unit.dropna("height", how="all").coords["height"].values
        self._w = ref_unit.dropna("width", how="all").coords["width"].values
        self._f = self.C_sub.isel(unit_id=0).dropna("frame").coords["frame"].values
        self._u = self.C_sub.isel(frame=0).dropna("unit_id").coords["unit_id"].values

    def _filter_centroids_to_current_metadata(self) -> None:
        if not self.meta_dicts:
            self.cents_sub = self.cents
            return
        row_mask = pd.concat(
            [self.cents[d] == v for d, v in self.metas.items()],
            axis="columns",
        ).all(axis="columns")
        self.cents_sub = self.cents[row_mask]

    def compute_subs(self, clicks=None):
        self.A_sub = self.A_sub.compute()
        self.C_sub = self.C_sub.compute()
        self.S_sub = self.S_sub.compute()
        self.org_sub = self.org_sub.compute()
        self.C_norm_sub = self.C_norm_sub.compute()
        self.S_norm_sub = self.S_norm_sub.compute()

    def update_all(self, clicks=None):
        self.update_subs()
        self.strm_uid.event(index=[])
        self.strm_f.event(x=0)
        self.update_spatial_all()

    def callback_uid(self, index=None):
        self.update_temp()
        self.update_AC()
        self.update_usub_lab()

    def callback_f(self, f, y):
        if len(self._AC) > 0 and len(self._mov) > 0:
            fidx = np.abs(self._f - f).argmin()
            f = self._f[fidx]
            if self._useAC:
                AC = self._AC.sel(frame=f)
            else:
                AC = self._AC
            mov = self._mov.sel(frame=f)
            self.pipAC.send(AC)
            self.pipmov.send(mov)
            try:
                self.wgt_temp_comp[1].value = int(fidx)
            except AttributeError:
                pass
        else:
            self.pipAC.send([])
            self.pipmov.send([])

    def callback_usub(self, usub=None):
        self.update_temp_comp_sub(usub)
        self.update_AC(usub)
        self.update_usub_lab(usub)

    def _meta_wgt(self):
        wgt_meta = build_meta_select_widgets(self.meta_dicts)
        wire_meta_select_widgets(wgt_meta, self.metas, self.update_subs)
        wgt_update = pnwgt.Button(
            name="Refresh", button_type="primary", height=30, width=120
        )
        wgt_update.param.watch(self.update_all, "clicks")
        wgt_load = pnwgt.Button(
            name="Load Data", button_type="danger", height=30, width=120
        )
        wgt_load.param.watch(self.compute_subs, "clicks")
        return pn.layout.WidgetBox(
            *(list(wgt_meta.values()) + [wgt_update, wgt_load]), width=150
        )

    def show(self) -> pn.layout.Column:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolboxes.
        """
        return pn.layout.Column(
            self.spatial_all,
            pn.layout.Row(
                pn.layout.Column(
                    pn.layout.Row(self.wgt_meta, self.wgt_spatial_all),
                    self.wgt_temp_comp,
                ),
                self.temp_comp_sub,
                self.wgt_man,
            ),
        )

    def _temp_comp_sub(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if self._normalize:
            C, S = self.C_norm_sub, self.S_norm_sub
        else:
            C, S = self.C_sub, self.S_sub
        cur_temp = dict()
        if self._showC:
            cur_temp["C"] = hv.Dataset(
                C.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        if self._showS:
            cur_temp["S"] = hv.Dataset(
                S.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        cur_vl = hv.DynamicMap(
            lambda f, y: hv.VLine(f) if f else hv.VLine(0), streams=[self.strm_f]
        ).opts(color="red")
        cur_cv = hv.Curve([], kdims=["frame"], vdims=["Internsity (A.U.)"])
        self.strm_f.source = cur_cv
        h_cv = len(self._w) // 8
        w_cv = len(self._w) * 2
        temp_comp = (
            cur_cv
            * datashade_ndcurve(
                hv.HoloMap(cur_temp, "trace")
                .collate()
                .overlay("trace")
                .grid("unit_id")
                .add_dimension("time", 0, 0),
                "trace",
            )
            .opts(shared_xaxis=True)
            .opts(hv.opts.RGB(frame_height=h_cv, frame_width=w_cv))
            * cur_vl
        )
        temp_comp[temp_comp.keys()[0]] = temp_comp[temp_comp.keys()[0]].opts(
            height=h_cv + 75
        )
        return pn.panel(temp_comp)

    def update_temp_comp_sub(self, usub=None):
        self.temp_comp_sub.object = self._temp_comp_sub(usub).object
        self.wgt_man.objects = self._man_wgt().objects

    def update_norm(self, norm):
        self._normalize = norm.new
        self.update_temp_comp_sub()

    def _temp_comp_wgt(self):
        if self.strm_uid.index:
            cur_idxs = self.strm_uid.index
        else:
            cur_idxs = self._u
        ntabs = np.ceil(len(cur_idxs) / 5)
        sub_idxs = np.array_split(cur_idxs, ntabs)
        idxs_dict = OrderedDict(
            [("group{}".format(i), g.tolist()) for i, g in enumerate(sub_idxs)]
        )
        def_idxs = list(idxs_dict.values())[0]
        wgt_grp = pnwgt.Select(
            name="", options=idxs_dict, width=120, height=30, value=def_idxs
        )

        def update_usub(usub):
            self.usub_sel = []
            self.strm_usub.event(usub=usub.new)

        wgt_grp.param.watch(update_usub, "value")
        wgt_grp.value = def_idxs
        self.strm_usub.event(usub=def_idxs)
        wgt_grp_prv = pnwgt.Button(
            name="Previous Group", width=120, height=30, button_type="primary"
        )

        def prv(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                prv_val = idxs_dict[list(idxs_dict.keys())[ig - 1]]
                wgt_grp.value = prv_val
            except:
                pass

        wgt_grp_prv.param.watch(prv, "clicks")
        wgt_grp_nxt = pnwgt.Button(
            name="Next Group", width=120, height=30, button_type="primary"
        )

        def nxt(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                nxt_val = idxs_dict[list(idxs_dict.keys())[ig + 1]]
                wgt_grp.value = nxt_val
            except:
                pass

        wgt_grp_nxt.param.watch(nxt, "clicks")
        wgt_norm = pnwgt.Checkbox(
            name="Normalize", value=self._normalize, width=120, height=10
        )
        wgt_norm.param.watch(self.update_norm, "value")
        wgt_showC = pnwgt.Checkbox(
            name="ShowC", value=self._showC, width=120, height=10
        )

        def callback_showC(val):
            self._showC = val.new
            self.update_temp_comp_sub()

        wgt_showC.param.watch(callback_showC, "value")
        wgt_showS = pnwgt.Checkbox(
            name="ShowS", value=self._showS, width=120, height=10
        )

        def callback_showS(val):
            self._showS = val.new
            self.update_temp_comp_sub()

        wgt_showS.param.watch(callback_showS, "value")
        wgt_play = pnwgt.Player(length=len(self._f), interval=10, value=0, width=280)
        wire_frame_player_index(wgt_play, lambda i: self.strm_f.event(x=self._f[i]))
        wgt_groups = pn.layout.Row(
            pn.layout.WidgetBox(wgt_norm, wgt_showC, wgt_showS, wgt_grp, width=150),
            pn.layout.WidgetBox(wgt_grp_prv, wgt_grp_nxt, width=150),
        )
        return pn.layout.Column(wgt_groups, wgt_play)

    def _man_wgt(self):
        usub = self.strm_usub.usub
        usub.sort()
        usub.reverse()
        ulabs = self.unit_labels.sel(unit_id=usub).values
        wgt_sel = {
            uid: pnwgt.Select(
                name="Unit Label",
                options=usub + [-1] + ulabs.tolist(),
                value=ulb,
                height=50,
                width=80,
            )
            for uid, ulb in zip(usub, ulabs)
        }

        def callback_ulab(value, uid):
            self.unit_labels.loc[uid] = value.new

        for uid, sel in wgt_sel.items():
            cb = fct.partial(callback_ulab, uid=uid)
            sel.param.watch(cb, "value")
        wgt_check = {
            uid: pnwgt.Checkbox(
                name="Unit ID: {}".format(uid), value=False, height=50, width=100
            )
            for uid in usub
        }

        def callback_chk(val, uid):
            if not val.old == val.new:
                if val.new:
                    self.usub_sel.append(uid)
                else:
                    self.usub_sel.remove(uid)

        for uid, chk in wgt_check.items():
            cb = fct.partial(callback_chk, uid=uid)
            chk.param.watch(cb, "value")
        wgt_discard = pnwgt.Button(
            name="Discard Selected", button_type="primary", width=180
        )

        def callback_discard(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = -1

        wgt_discard.param.watch(callback_discard, "clicks")
        wgt_merge = pnwgt.Button(
            name="Merge Selected", button_type="primary", width=180
        )

        def callback_merge(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = self.usub_sel[0]

        wgt_merge.param.watch(callback_merge, "clicks")
        return pn.layout.Column(
            pn.layout.WidgetBox(wgt_discard, wgt_merge, width=200),
            pn.layout.Row(
                pn.layout.WidgetBox(*wgt_check.values(), width=100),
                pn.layout.WidgetBox(*wgt_sel.values(), width=100),
            ),
        )

    def update_temp_comp_wgt(self):
        self.wgt_temp_comp.objects = self._temp_comp_wgt().objects

    def update_temp(self):
        self.update_temp_comp_wgt()

    def update_AC(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            if self._useAC:
                umask = (self.A_sub.sel(unit_id=usub) > 0).any("unit_id")
                A_sub = self.A_sub.sel(unit_id=usub).where(umask, drop=True).fillna(0)
                C_sub = self.C_sub.sel(unit_id=usub)
                AC = xr.apply_ufunc(
                    da.dot,
                    A_sub,
                    C_sub,
                    input_core_dims=[
                        ["height", "width", "unit_id"],
                        ["unit_id", "frame"],
                    ],
                    output_core_dims=[["height", "width", "frame"]],
                    dask="allowed",
                )
                self._AC = AC.compute()
                wndh, wndw = AC.coords["height"].values, AC.coords["width"].values
                window = self.A_sub.sel(
                    height=slice(wndh.min(), wndh.max()),
                    width=slice(wndw.min(), wndw.max()),
                )
                self._AC = self._AC.reindex_like(window).fillna(0)
                self._mov = (self.org_sub.reindex_like(window)).compute()
            else:
                self._AC = self.A_sub.sel(unit_id=usub).sum("unit_id")
                self._mov = self.org_sub
            self.strm_f.event(x=0)
        else:
            self._AC = xr.DataArray([])
            self._mov = xr.DataArray([])
            self.strm_f.event(x=0)

    def update_usub_lab(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            self.pipusub.send(self.cents_sub[self.cents_sub["unit_id"].isin(usub)])
        else:
            self.pipusub.send([])

    def _spatial_all_wgt(self):
        wgt_useAC = pnwgt.Checkbox(
            name="UseAC", value=self._useAC, width=120, height=15
        )

        def callback_useAC(val):
            self._useAC = val.new
            self.update_AC()

        wgt_useAC.param.watch(callback_useAC, "value")
        return pn.layout.WidgetBox(wgt_useAC, width=150)

    def _spatial_all(self):
        metas = self.metas
        _im_opts = footprint_image_opts(self._h, self._w)
        Asum = hv.Image(self.Asum.sel(**metas), ["width", "height"]).opts(**_im_opts)
        cents = (
            hv.Dataset(
                self.cents_sub.drop(list(self.meta_dicts.keys()), axis="columns"),
                kdims=["width", "height", "unit_id"],
            )
            .to(hv.Points, ["width", "height"])
            .opts(
                alpha=0.1,
                line_alpha=0,
                size=5,
                nonselection_alpha=0.1,
                selection_alpha=0.9,
            )
            .collate()
            .overlay("unit_id")
            .opts(tools=["hover", "box_select"])
        )
        self.strm_uid.source = cents
        fim = fct.partial(hv.Image, kdims=["width", "height"])
        AC = hv.DynamicMap(fim, streams=[self.pipAC]).opts(**_im_opts)
        mov = hv.DynamicMap(fim, streams=[self.pipmov]).opts(**_im_opts)
        lab = fct.partial(hv.Labels, kdims=["width", "height"], vdims=["unit_id"])
        ulab = hv.DynamicMap(lab, streams=[self.pipusub]).opts(text_color="red")
        return pn.panel(Asum * cents + AC * ulab + mov)

    def update_spatial_all(self):
        self.spatial_all.objects = self._spatial_all().objects
