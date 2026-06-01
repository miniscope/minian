import os
import subprocess
import sys

import pytest

from ..utilities import open_minian


@pytest.mark.notebook
def test_pipeline_notebook():
    os.makedirs("artifact", exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--output",
        "artifact/pipeline.ipynb",
        "--execute",
        "pipeline.ipynb",
    ]
    subprocess.run(args, check=True)
    minian_ds = open_minian("./demo_movies/minian")
    # Input dimensions are fixed by the demo movie, so check them exactly.
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    # The detected cell count and the downstream motion / CNMF (C, S, A) sums
    # drift with numerical-library versions (numpy/scipy/scikit-image/
    # scikit-learn/cvxpy solvers), platform/BLAS, and dask worker ordering, so
    # check them with a tolerance rather than exactly. Reference values were
    # captured from a verified run on the pinned snapshot (see
    # requirements/ci-constraints.txt). The CNMF sums use wider tolerances
    # because they are the most sensitive to those factors.
    assert minian_ds.sizes["unit_id"] == pytest.approx(286, abs=10)
    motion_sum = minian_ds["motion"].sum("frame").values.astype(int)
    assert list(motion_sum) == pytest.approx([391, -252], abs=30)
    assert int(minian_ds["max_proj"].sum().compute()) == pytest.approx(1501702, rel=1e-2)
    assert int(minian_ds["C"].sum().compute()) == pytest.approx(546290, rel=5e-2)
    assert int(minian_ds["S"].sum().compute()) == pytest.approx(5065, rel=1e-1)
    assert int(minian_ds["A"].sum().compute()) == pytest.approx(71468, rel=5e-2)
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    assert os.path.exists("./demo_movies/minian.mp4")
