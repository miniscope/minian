import os
import subprocess
import sys

import pytest

from ..utilities import open_minian


@pytest.mark.flaky(reruns=3)
def test_pipeline_notebook():
    os.makedirs("artifact", exist_ok=True)
    args = [
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
    # The detected cell count and the downstream motion / CNMF (C, S, A)
    # sums drift with newer numerical libraries (numpy/scipy/scikit-learn/
    # cvxpy solvers), so check them with a tolerance rather than exactly.
    assert minian_ds.sizes["unit_id"] == pytest.approx(282, abs=10)
    motion_sum = minian_ds["motion"].sum("frame").values.astype(int)
    assert list(motion_sum) == pytest.approx([423, -239], abs=20)
    assert int(minian_ds["max_proj"].sum().compute()) == pytest.approx(1501505, rel=1e-2)
    assert int(minian_ds["C"].sum().compute()) == pytest.approx(478444, rel=2e-2)
    assert int(minian_ds["S"].sum().compute()) == pytest.approx(3943, rel=5e-2)
    assert int(minian_ds["A"].sum().compute()) == pytest.approx(41755, rel=2e-2)
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    assert os.path.exists("./demo_movies/minian.mp4")
