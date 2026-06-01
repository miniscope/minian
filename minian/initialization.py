import itertools as itt
import os

import cv2
import dask.array as darr
import networkx as nx
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from scipy.ndimage.measurements import label
from scipy.sparse import csc_matrix
from scipy.stats import kstest, zscore
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, radius_neighbors_graph

from .cnmf import adj_corr, filt_fft, graph_optimize_corr, label_connected
from .utilities import local_extreme, med_baseline, save_minian, sps_lstsq


def seeds_init(
    varr: xr.DataArray,
    wnd_size=500,
    method="rolling",
    stp_size=200,
    nchunk=100,
    max_wnd=10,
    diff_thres=2,
):
    """
    Generate over-complete set of seeds by finding local maxima across frames.

    This function computes the maximum intensity projection of a subset of
    frames and finds the local maxima. The subsetting use either a rolling
    window or random sampling of frames. `wnd_size` `stp_size` and `nchunk`
    controls different aspects of the subsetting. `max_wnd` and `diff_thres`
    controls how local maxima are computed. The set of all local maxima found in
    this process constitutes  an overly-complete set of seeds, representing
    putative locations of cells.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "frame", "height" and "width".
    wnd_size : int, optional
        Number of frames in each chunk, for which a max projection will be
        calculated. By default `500`.
    method : str, optional
        Either `"rolling"` or `"random"`. Controls whether to use rolling window
        or random sampling of frames to construct chunks. By default
        `"rolling"`.
    stp_size : int, optional
        Number of frames between the center of each chunk when stepping through
        the data with rolling windows. Only used if `method is "rolling"`. By
        default `200`.
    nchunk : int, optional
        Number of chunks to sample randomly. Only used if `method is "random"`.
        By default `100`.
    max_wnd : int, optional
        Radius (in pixels) of the disk window used for computing local maxima.
        Local maximas are defined as pixels with maximum intensity in such a
        window. By default `10`.
    diff_thres : int, optional
        Intensity threshold for the difference between local maxima and its
        neighbours. Any local maxima that is not birghter than its neighbor
        (defined by the same disk window) by `diff_thres` intensity values will
        be filtered out. By default `2`.

    Returns
    -------
    seeds : pd.DataFrame
        Seeds dataframe with each seed as a row. Has column "height" and "width"
        which are location of the seeds. Also has column "seeds" which is an
        integer showing how many chunks where the seed is considered a local
        maxima.
    """
    int_path = os.environ["MINIAN_INTERMEDIATE"]
    print("constructing chunks")
    idx_fm = varr.coords["frame"]
    nfm = len(idx_fm)
    if method == "rolling":
        nstp = np.ceil(nfm / stp_size) + 1
        centers = np.linspace(0, nfm - 1, int(nstp))
        hwnd = np.ceil(wnd_size / 2)
        max_idx = [slice(int(np.floor(c - hwnd).clip(0)), int(np.ceil(c + hwnd))) for c in centers]
    elif method == "random":
        max_idx = [np.random.randint(0, nfm - 1, wnd_size) for _ in range(nchunk)]
    print("computing max projections")
    res = [max_proj_frame(varr, cur_idx) for cur_idx in max_idx]
    max_res = xr.concat(res, "sample")
    max_res = save_minian(max_res.rename("max_res"), int_path, overwrite=True)
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max_roll,
        max_res,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        kwargs={"k0": 2, "k1": max_wnd, "diff": diff_thres},
    ).sum("sample")
    seeds = loc_max.where(loc_max > 0).rename("seeds").to_dataframe().dropna().reset_index()
    return seeds[["height", "width", "seeds"]]


def max_proj_frame(varr: xr.DataArray, idx: np.ndarray) -> xr.DataArray:
    """
    Compute max projection on a given subset of frames.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data containing all frames.
    idx : np.ndarray
        The subset of frames to use to compute max projection.

    Returns
    -------
    max_proj : xr.DataArray
        The max projection.
    """
    return varr.isel(frame=idx).max("frame")


def local_max_roll(fm: np.ndarray, k0: int, k1: int, diff: int | float) -> np.ndarray:
    """
    Compute local maxima of a frame with a range of kernel size.

    This function wraps around :func:`minian.utilities.local_extreme` and
    compute local maxima of the input frame with kernels of size ranging from
    `k0` to `k1`. It then takes the union of all the local maxima, and
    additionally merge all the connecting local maxima by using the middle
    pixel.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    k0 : int
        The lower bound (inclusive) of the range of kernel sizes.
    k1 : int
        The upper bound (inclusive) of the range of kernel sizes.
    diff : Union[int, float]
        Intensity threshold for the difference between local maxima and its
        neighbours, passed to :func:`minian.utilities.local_extreme`.

    Returns
    -------
    max_res : np.ndarray
        The image of local maxima. Has same shape as `fm`, and 1 at local
        maxima.
    """
    max_ls = []
    for ksize in range(k0, k1):
        selem = disk(ksize)
        fm_max = local_extreme(fm, selem, diff=diff)
        max_ls.append(fm_max)
    lmax = (np.stack(max_ls, axis=0).sum(axis=0) > 0).astype(np.uint8)
    nlab, max_lab = cv2.connectedComponents(lmax)
    max_res = np.zeros_like(lmax)
    for lb in range(1, nlab):
        area = max_lab == lb
        if np.sum(area) > 1:
            crds = tuple(int(np.median(c)) for c in np.where(area))
            max_res[crds] = 1
        else:
            max_res[np.where(area)] = 1
    return max_res


def gmm_refine(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    q=(0.1, 99.9),
    n_components=2,
    valid_components=1,
    mean_mask=True,
) -> tuple[pd.DataFrame, xr.DataArray, GaussianMixture]:
    """
    Filter seeds by fitting a GMM to peak-to-peak values.

    This function assume that the distribution of peak-to-peak values of
    fluorescence across all seeds can be model by a Gaussian Mixture Model (GMM)
    with different means. It computes peak-to-peak value for all the seeds, then
    fit a GMM with `n_components` to the distribution, and filter out the seeds
    belonging to the `n_components - valid_components` number of gaussians with
    lower means.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data. Should have dimension "spatial" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    q : tuple, optional
        Percentile to use to compute the peak-to-peak values. For a given seed
        with corresponding fluorescent fluctuation `f`, the peak-to-peak value
        for that seed is computed as `np.percentile(f, q[1]) - np.percentile(f,
        q[0])`. By default `(0.1, 99.9)`.
    n_components : int, optional
        Number of components (Gaussians) in the GMM model. By default `2`.
    valid_components : int, optional
        Number of components (Gaussians) to be considered as modeling the
        distribution of peak-to-peak values of valid seeds. Should be smaller
        than `n_components`. By default `1`.
    mean_mask : bool, optional
        Whether to apply additional criteria where a seed is valid only if its
        peak-to-peak value exceeds the mean of the lowest gaussian distribution.
        Only useful in corner cases where the distribution of the gaussian
        heavily overlap. By default `True`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_gmm",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    varr_pv : xr.DataArray
        The computed peak-to-peak values for each seeds.
    gmm : GaussianMixture
        The fitted GMM model object.

    See Also
    -------
    sklearn.mixture.GaussianMixture
    """
    print("selecting seeds")
    varr_sub = varr.sel(spatial=[tuple(hw) for hw in seeds[["height", "width"]].values])
    print("computing peak-valley values")
    varr_valley = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk({"frame": -1}),
        input_core_dims=[["frame"]],
        kwargs={"q": q[0], "axis": -1},
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
    varr_peak = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk({"frame": -1}),
        input_core_dims=[["frame"]],
        kwargs={"q": q[1], "axis": -1},
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
    varr_pv = varr_peak - varr_valley
    varr_pv = varr_pv.compute()
    print("fitting GMM models")
    dat = varr_pv.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(dat)
    idg = np.argsort(gmm.means_.reshape(-1))[-valid_components:]
    idx_valid = np.isin(gmm.predict(dat), idg)
    if mean_mask:
        idx_mean = dat > np.sort(gmm.means_)[0]
        idx_valid = np.logical_and(idx_mean.squeeze(), idx_valid)
    seeds["mask_gmm"] = idx_valid
    return seeds, varr_pv, gmm


def pnr_refine(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    noise_freq=0.25,
    thres: float | str = 1.5,
    q=(0.1, 99.9),
    med_wnd: int | None = None,
) -> tuple[pd.DataFrame, xr.DataArray, GaussianMixture | None]:
    """
    Filter seeds by thresholding peak-to-noise ratio.

    For each input seed, the noise is defined as high-pass filtered fluorescence
    trace of the seed. The peak-to-noise ratio (pnr) of that seed is then
    defined as the ratio between the peak-to-peak value of the originial
    fluorescence trace and that of the noise trace. Optionally, if abrupt
    changes in baseline fluorescence is expected, then the baseline can be
    estimated by median-filtering the fluorescence trace and subtracted from the
    original trace before computing the peak-to-noise ratio. In addition, if a
    hard threshold of pnr is not desired, then a Gaussian Mixture Model with 2
    components can be fitted to the distribution of pnr across all seeds, and
    only seeds with pnr belonging to the higher-mean Gaussian will be considered
    valide.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data, should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    noise_freq : float, optional
        Cut-off frequency for the high-pass filter used to define noise,
        specified as fraction of sampling frequency. By default `0.25`.
    thres : Union[float, str], optional
        Threshold of the peak-to-noise ratio. If `"auto"` then a :class:`GMM
        <sklearn.mixture.GaussianMixture>` will be fit to the distribution of
        pnr. By default `1.5`.
    q : tuple, optional
        Percentile to use to compute the peak-to-peak values. For a given
        fluorescence fluctuation `f`, the peak-to-peak value for that seed is
        computed as `np.percentile(f, q[1]) - np.percentile(f, q[0])`. By
        default `(0.1, 99.9)`.
    med_wnd : int, optional
        Size of the median filter window to remove baseline. If `None` then no
        filtering will be done. By default `None`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_pnr",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    pnr : xr.DataArray
        The computed peak-to-noise ratio for each seeds.
    gmm : GaussianMixture, optional
        The GMM model object fitted to the distribution of pnr. Will be `None`
        unless `thres` is `"auto"`.
    """
    print("selecting seeds")
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray())
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
    if med_wnd:
        print("removing baseline")
        varr = xr.apply_ufunc(
            med_baseline,
            varr_sub,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            kwargs={"wnd": med_wnd},
            vectorize=True,
            output_dtypes=[varr.dtype],
        )
    print("computing peak-noise ratio")
    pnr = xr.apply_ufunc(
        pnr_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        kwargs={"freq": noise_freq, "q": q},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    if thres == "auto":
        gmm = GaussianMixture(n_components=2)
        gmm.fit(np.nan_to_num(pnr.values.reshape(-1, 1)))
        idg = np.argsort(gmm.means_.reshape(-1))[-1]
        idx_valid = np.isin(gmm.predict(pnr.values.reshape(-1, 1)), idg)
        seeds["mask_pnr"] = idx_valid
    else:
        mask = pnr > thres
        mask_df = mask.to_pandas().rename("mask_pnr")
        seeds["mask_pnr"] = mask_df
        gmm = None
    return seeds, pnr, gmm


def ptp_q(a: np.ndarray, q: tuple) -> float:
    """
    Compute peak-to-peak value of input with percentile values.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    q : tuple
        tuple specifying low and high percentile values.

    Returns
    -------
    ptp : float
        The peak-to-peak value.
    """
    return np.percentile(a, q[1]) - np.percentile(a, q[0])


def pnr_perseed(a: np.ndarray, freq: float, q: tuple) -> float:
    """
    Compute peak-to-noise ratio of a given timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency of the high-pass filtering used to define noise.
    q : tuple
        Percentile used to compute peak-to-peak values.

    Returns
    -------
    pnr : float
        Peak-to-noise ratio.

    See Also
    -------
    pnr_refine : for definition of peak-to-noise ratio
    """
    ptp = ptp_q(a, q)
    a = filt_fft(a, freq, btype="high")
    ptp_noise = ptp_q(a, q)
    return ptp / ptp_noise


def intensity_refine(varr: xr.DataArray, seeds: pd.DataFrame, thres_mul=2) -> pd.DataFrame:
    """
    Filter seeds by thresholding the intensity of their corresponding pixels in
    the max projection of the movie.

    This function generate a histogram of the max projection by spliting the
    intensity into bins of roughly 10 pixels. Then the intensity threshold is
    defined as the intensity of the peak of the histogram times `thres_mul`.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    thres_mul : int, optional
        Scalar multiplied to the intensity value corresponding to the peak of
        max projection histogram. By default `2`, which can be interpreted as
        "seeds are only valid if they are more than twice as bright as the
        majority of the pixels".

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_int",
        indicating whether the seed is considered valid by this function.
    """
    try:
        fm_max = varr.max("frame")
    except ValueError:
        print("using input as max projection")
        fm_max = varr
    bins = np.around(fm_max.sizes["height"] * fm_max.sizes["width"] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    try:
        thres = edges[int(np.around(np.argmax(hist) * thres_mul))]
    except IndexError:
        print("threshold out of bound, returning input")
        return seeds
    mask = (fm_max > thres).stack(spatial=["height", "width"])
    mask_df = mask.to_pandas().rename("mask_int").reset_index()
    seeds = pd.merge(seeds, mask_df, on=["height", "width"], how="left")
    return seeds


def ks_refine(varr: xr.DataArray, seeds: pd.DataFrame, sig=0.01) -> pd.DataFrame:
    """
    Filter the seeds using Kolmogorov-Smirnov (KS) test.

    This function assume that the valid seeds’ fluorescence across frames
    notionally follows a bimodal distribution: with a large normal distribution
    representing baseline activity, and a second peak representing when the
    seed/cell is active. KS allows to discard the seeds where the
    null-hypothesis (i.e. the fluorescence intensity is simply a normal
    distribution) is rejected at `sig`.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    sig : float, optional
        The significance threshold to reject null-hypothesis. By default `0.01`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_ks",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    """
    print("selecting seeds")
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray())
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
    print("performing KS test")
    ks = xr.apply_ufunc(
        ks_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    ks = (ks < sig).to_pandas().rename("mask_ks")
    seeds["mask_ks"] = ks
    return seeds


def ks_perseed(a: np.ndarray) -> float:
    """
    Perform KS test on input and return the p-value.

    Parameters
    ----------
    a : np.ndarray
        Input data.

    Returns
    -------
    p : float
        The p-value of the KS test.

    See Also
    -------
    scipy.stats.kstest
    """
    a = zscore(a)
    return kstest(a, "norm")[1]


def seeds_merge(
    varr: xr.DataArray,
    max_proj: xr.DataArray,
    seeds: pd.DataFrame,
    thres_dist=5,
    thres_corr=0.6,
    noise_freq: float | None = None,
) -> pd.DataFrame:
    """
    Merge seeds based on spatial distance and temporal correlation of their
    activities.

    This function build an adjacency matrix by thresholding spatial distance
    between seeds and temporal correlation between activities of seeds. It then
    merge seeds using the adjacency matrix by only keeping the seed with maximum
    intensity in the max projection within each connected group of seeds. The
    merge is therefore transitive.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimension "height", "width" and "frame".
    max_proj : xr.DataArray
        Max projection of the movie data.
    seeds : pd.DataFrame
        Dataframe of seeds to be merged.
    thres_dist : int, optional
        Threshold of distance between seeds in pixel. By default `5`.
    thres_corr : float, optional
        Threshold of temporal correlation between activities of seeds. By
        default `0.6`.
    noise_freq : float, optional
        Cut-off frequency for optional smoothing of activities before computing
        the correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_mrg",
        indicating whether the seed should be kept after the merge. If the
        column already exists in input `seeds` it will be overwritten.
    """
    print("computing distance")
    nng = radius_neighbors_graph(seeds[["height", "width"]], thres_dist)
    print("computing correlations")
    adj = adj_corr(varr, nng, seeds[["height", "width"]], noise_freq)
    print("merging seeds")
    adj = adj > thres_corr
    adj = adj + adj.T
    labels = label_connected(adj, only_connected=True)
    iso = np.where(labels < 0)[0]
    seeds_final = set(iso.tolist())
    for cur_cmp in np.unique(labels):
        if cur_cmp < 0:
            continue
        cur_smp = np.where(labels == cur_cmp)[0]
        cur_max = np.array(
            [
                max_proj.sel(height=seeds.iloc[s]["height"], width=seeds.iloc[s]["width"])
                for s in cur_smp
            ]
        )
        max_seed = cur_smp[np.argmax(cur_max)]
        seeds_final.add(max_seed)
    seeds["mask_mrg"] = False
    seeds.loc[list(seeds_final), "mask_mrg"] = True
    return seeds


def initA(
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    thres_corr=0.8,
    wnd=10,
    noise_freq: float | None = None,
) -> xr.DataArray:
    """
    Initialize spatial footprints from seeds.

    For each input seed, this function compute the correlation between the
    fluorescence activity of the seed and those of its neighboring pixels up to
    `wnd` pixels. It then set all correlation below `thres_corr` to zero, and
    use the resulting correlation image as the resutling spatial footprint of
    the seed.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimension "height", "width" and "frame".
    seeds : pd.DataFrame
        Dataframe of seeds.
    thres_corr : float, optional
        Threshold of correlation, below which the values will be set to zero in
        the resulting spatial footprints. By default `0.8`.
    wnd : int, optional
        Radius (in pixels) of a disk window within which correlation will be
        computed for each seed. By default `10`.
    noise_freq : float, optional
        Cut-off frequency for optional smoothing of activities before computing
        the correlation. If `None` then no smoothing will be done. By default
        `None`.

    Returns
    -------
    A : xr.DataArray
        The initial estimation of spatial footprint for each cell. Should have
        dimensions ("unit_id", "height", "width").

    See Also
    -------
    minian.cnmf.graph_optimize_corr :
        for how the correlation are computed in an out-of-core fashion
    """
    print("optimizing computation graph")
    nod_df = pd.DataFrame(
        np.array(list(itt.product(varr.coords["height"].values, varr.coords["width"].values))),
        columns=["height", "width"],
    ).merge(seeds.reset_index(), how="outer", on=["height", "width"])
    seed_df = nod_df[nod_df["index"].notnull()]
    nn_tree = KDTree(nod_df[["height", "width"]], leaf_size=(2 * wnd) ** 2)
    nns_arr = nn_tree.query_radius(seed_df[["height", "width"]], r=wnd)
    sdg = nx.Graph()
    sdg.add_nodes_from(
        [(i, d) for i, d in enumerate(nod_df[["height", "width", "index"]].to_dict("records"))]
    )
    for isd, nns in enumerate(nns_arr):
        cur_sd = seed_df.index[isd]
        sdg.add_edges_from([(cur_sd, n) for n in nns if n != cur_sd])
    sdg.remove_nodes_from(list(nx.isolates(sdg)))
    sdg = nx.convert_node_labels_to_integers(sdg)
    corr_df = graph_optimize_corr(varr, sdg, noise_freq)
    print("building spatial matrix")
    corr_df = corr_df[corr_df["corr"] > thres_corr]
    nod_df = pd.DataFrame.from_dict(dict(sdg.nodes(data=True)), orient="index")
    seed_df = nod_df[nod_df["index"].notnull()].astype({"index": int})
    A_ls = []
    ih_dict = (
        varr.coords["height"]
        .to_series()
        .reset_index(drop=True)
        .reset_index()
        .set_index("height")["index"]
        .to_dict()
    )
    iw_dict = (
        varr.coords["width"]
        .to_series()
        .reset_index(drop=True)
        .reset_index()
        .set_index("width")["index"]
        .to_dict()
    )
    Ashape = (varr.sizes["height"], varr.sizes["width"])
    for seed_id, sd in seed_df.iterrows():
        src_corr = corr_df[corr_df["target"] == seed_id].copy()
        src_nods = nod_df.loc[src_corr["source"]]
        src_corr["height"], src_corr["width"] = (
            src_nods["height"].values,
            src_nods["width"].values,
        )
        tgt_corr = corr_df[corr_df["source"] == seed_id].copy()
        tgt_nods = nod_df.loc[tgt_corr["target"]]
        tgt_corr["height"], tgt_corr["width"] = (
            tgt_nods["height"].values,
            tgt_nods["width"].values,
        )
        cur_corr = pd.concat(
            [
                src_corr,
                tgt_corr,
                pd.DataFrame([{"corr": 1, "height": sd["height"], "width": sd["width"]}]),
            ],
            ignore_index=True,
        )
        cur_corr["iheight"] = cur_corr["height"].map(ih_dict)
        cur_corr["iwidth"] = cur_corr["width"].map(iw_dict)
        cur_A = darr.array(
            sparse.COO(cur_corr[["iheight", "iwidth"]].T, cur_corr["corr"], shape=Ashape)
        )
        A_ls.append(cur_A)
    A = xr.DataArray(
        darr.stack(A_ls).map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": seed_df["index"].values,
            "height": varr.coords["height"].values,
            "width": varr.coords["width"].values,
        },
    )
    return A


def initC(varr: xr.DataArray, A: xr.DataArray) -> xr.DataArray:
    """
    Initialize temporal component given spatial footprints.

    The temporal component is computed as the least-square solution between the
    input movie and the spatial footprints over the "height" and "width"
    dimensions.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions ("height", "width", "frame").
    A : xr.DataArray
        Spatial footprints of cells. Should have dimensions ("unit_id",
        "height", "width").

    Returns
    -------
    C : xr.DataArray
        The initial estimation of temporal components for each cell. Should have
        dimensions ("unit_id", "frame").
    """
    uids = A.coords["unit_id"]
    fms = varr.coords["frame"]
    A = (
        A.stack(spatial=["height", "width"])
        .transpose("spatial", "unit_id")
        .data.map_blocks(csc_matrix)
        .rechunk(-1)
        .persist()
    )
    varr = varr.stack(spatial=["height", "width"]).transpose("frame", "spatial").data
    C = sps_lstsq(A, varr, iter_lim=10)
    C = xr.DataArray(
        C, dims=["frame", "unit_id"], coords={"unit_id": uids, "frame": fms}
    ).transpose("unit_id", "frame")
    return C


@darr.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im: np.ndarray) -> np.ndarray:
    """
    Label connected features in a 2d array.

    Parameters
    ----------
    im : np.ndarray
        Input array.

    Returns
    -------
    label : np.ndarray
        Label array. Should have same shape as input `im`.

    See Also
    -------
    scipy.ndimage.label
    """
    return label(im)[0]
