import pytest
import numpy as np
import holoviews as hv

from ..io import load_videos
from ..preprocessing import denoise, remove_background, stripe_correction

param_load_videos = {
    "pattern": "msCam[0-9].avi",
    "dtype": np.uint8,
    "downsample": dict(frame=2, height=1, width=1),
    "downsample_strategy": "subset",
}

param_denoise = {"method": "median", "ksize": 7}

param_background_removal = {"method": "tophat", "wnd": 15}


# Module-scoped: resolving the dataset hashes ~688 MB, so do it once and share
# the (read-only) videos across the tests below. The dataset path is plucked
# from the shared ``dataset`` fixture tree (defined in conftest) rather than
# resolved directly here.
@pytest.fixture(scope="module")
def varr(dataset):
    # Same demo recording as the pipeline notebook (the msCam .avi files);
    # download/cache it, or skip if unavailable.
    dpath = dataset("pipeline-demo")
    return load_videos(str(dpath), **param_load_videos)


def test_can_load_videos(varr):
    assert varr.shape[0] == 900  # frames
    assert varr.shape[1] == 480  # height
    assert varr.shape[2] == 752  # width


def test_remove_background(varr):
    varr_ref = denoise(varr, **param_denoise)
    varr_ref_remove = remove_background(varr_ref, **param_background_removal)
    # when both are equal the denoise didn't do anything --> fail
    assert (varr_ref != varr_ref_remove).any()


def test_denoise(varr):
    varr_ref = denoise(varr, **param_denoise)
    # when both are equal the denoise didn't do anything --> fail
    assert (varr_ref != varr).any()
