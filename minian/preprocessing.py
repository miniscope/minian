import cv2
import numpy as np
import xarray as xr
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import uniform_filter
from skimage.morphology import disk


def remove_background(varr: xr.DataArray, method: str, wnd: int) -> xr.DataArray:
    """
    Remove background from a video.

    This function remove background frame by frame. Two methods are available
    for use: if `method == "uniform"`, then the background is estimated by
    convolving the frame with a uniform/mean kernel and then subtract it from
    the frame. If `method == "tophat"`, then a morphological tophat operation is
    applied to each frame.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and
        "frame".
    method : str
        The method used to remove the background. Should be either `"uniform"`
        or `"tophat"`.
    wnd : int
        Window size of kernels used for background removal, specified in pixels.
        If `method == "uniform"`, this will be the size of a box kernel
        convolved with each frame. If `method == "tophat"`, this will be the
        radius of a disk kernel used for morphological operations.

    Returns
    -------
    res : xr.DataArray
        The resulting movie with background removed. Same shape as input `varr`
        but will have `"_subtracted"` appended to its name.

    See Also
    --------
    `Morphology <https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html>`_ :
        for details about morphological operations
    """
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=dict(method=method, wnd=wnd, selem=selem),
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_subtracted")


def remove_background_perframe(
    fm: np.ndarray, method: str, wnd: int, selem: np.ndarray
) -> np.ndarray:
    """
    Remove background from a single frame.

    Parameters
    ----------
    fm : np.ndarray
        The input frame.
    method : str
        Method to use to remove background. Should be either `"uniform"` or
        `"tophat"`.
    wnd : int
        Size of the uniform filter. Only used if `method == "uniform"`.
    selem : np.ndarray
        Kernel used for morphological operations. Only used if `method == "tophat"`.

    Returns
    -------
    fm : np.ndarray
        The frame with background removed.

    See Also
    --------
    remove_background : for detailed explanations
    """
    if method == "uniform":
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)


def stripe_correction(varr, reduce_dim="height", on="mean"):
    if on == "mean":
        temp = varr.mean(dim="frame")
    elif on == "max":
        temp = varr.max(dim="frame")
    elif on == "perframe":
        temp = varr
    else:
        raise NotImplementedError(f"on {on} not understood")
    mean1d = temp.mean(dim=reduce_dim)
    varr_sc = varr - mean1d
    return varr_sc.rename(varr.name + "_Stripe_Corrected")


def denoise(varr: xr.DataArray, method: str, **kwargs) -> xr.DataArray:
    """
    Denoise the movie frame by frame.

    This function wraps around several image processing functions to denoise the
    data frame by frame. All additional keyword arguments will be passed
    directly to the underlying functions.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and
        "frame".
    method : str
        The method to use to denoise each frame. If `"gaussian"`, then a
        gaussian filter will be applied using :func:`cv2.GaussianBlur`. If
        `"anisotropic"`, then anisotropic filtering will be applied using
        :func:`medpy.filter.smoothing.anisotropic_diffusion`. If `"median"`,
        then a median filter will be applied using :func:`cv2.medianBlur`. If
        `"bilateral"`, then a bilateral filter will be applied using
        :func:`cv2.bilateralFilter`.

    Returns
    -------
    res : xr.DataArray
        The resulting denoised movie. Same shape as input `varr` but will have
        `"_denoised"` appended to its name.

    Raises
    ------
    NotImplementedError
        if the supplied `method` is not recognized
    """
    if method == "gaussian":
        func = cv2.GaussianBlur
    elif method == "anisotropic":
        func = anisotropic_diffusion
    elif method == "median":
        func = cv2.medianBlur
    elif method == "bilateral":
        func = cv2.bilateralFilter
    else:
        raise NotImplementedError(f"denoise method {method} not understood")
    res = xr.apply_ufunc(
        func,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=kwargs,
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_denoised")
