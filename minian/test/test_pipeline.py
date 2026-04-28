import os
import subprocess
import sys

import numpy as np
import pytest

from ..constants import MINIAN
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
    minian_ds = open_minian(os.path.join("./demo_movies", MINIAN))
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    assert minian_ds.sizes["unit_id"] == 282
    assert (
        minian_ds["motion"].sum("frame").values.astype(int) == np.array([423, -239])
    ).all()
    assert int(minian_ds["max_proj"].sum().compute()) == 1501505
    assert int(minian_ds["C"].sum().compute()) == 478444
    assert int(minian_ds["S"].sum().compute()) == 3943
    assert int(minian_ds["A"].sum().compute()) == 41755
    assert os.path.exists(os.path.join("./demo_movies", f"{MINIAN}_mc.mp4"))
    assert os.path.exists(os.path.join("./demo_movies", f"{MINIAN}.mp4"))
