"""Execute pipeline.ipynb and assert golden outputs (slow; not part of default pytest)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from minian.utilities import open_minian


def main() -> None:
    os.makedirs("artifact", exist_ok=True)
    subprocess.run(
        [
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
        ],
        check=True,
    )
    minian_ds = open_minian("./demo_movies/minian")
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    assert minian_ds.sizes["unit_id"] == pytest.approx(286, abs=10)
    motion_sum = minian_ds["motion"].sum("frame").values.astype(int)
    assert list(motion_sum) == pytest.approx([391, -252], abs=30)
    assert int(minian_ds["max_proj"].sum().compute()) == pytest.approx(1501702, rel=1e-2)
    assert int(minian_ds["C"].sum().compute()) == pytest.approx(546290, rel=5e-2)
    assert int(minian_ds["S"].sum().compute()) == pytest.approx(5065, rel=1e-1)
    assert int(minian_ds["A"].sum().compute()) == pytest.approx(71468, rel=5e-2)
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    assert os.path.exists("./demo_movies/minian.mp4")


if __name__ == "__main__":
    main()
