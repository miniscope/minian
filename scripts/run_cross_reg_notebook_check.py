"""Execute cross-registration.ipynb and assert golden outputs (slow; not default pytest)."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

DEMO_DATA_PATH = "demo_data"


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
            "artifact/cross-registration.ipynb",
            "--execute",
            "cross-registration.ipynb",
        ],
        check=True,
    )
    assert os.path.exists(f"{DEMO_DATA_PATH}/shiftds.nc")
    assert os.path.exists(f"{DEMO_DATA_PATH}/cents.pkl")
    assert os.path.exists(f"{DEMO_DATA_PATH}/mappings.pkl")
    cents = pd.read_pickle(f"{DEMO_DATA_PATH}/cents.pkl")
    mappings = pd.read_pickle(f"{DEMO_DATA_PATH}/mappings.pkl")
    assert len(cents) == 508
    assert cents["height"].sum() == pytest.approx(99091, rel=1e-3)
    assert cents["width"].sum() == pytest.approx(213627, rel=1e-3)
    assert len(mappings) == pytest.approx(431, abs=3)
    group_counts = mappings[("group", "group")].value_counts().to_dict()
    assert set(group_counts) == {
        ("session1",),
        ("session2",),
        ("session1", "session2"),
    }
    assert group_counts[("session2",)] == pytest.approx(182, abs=5)
    assert group_counts[("session1",)] == pytest.approx(172, abs=5)
    assert group_counts[("session1", "session2")] == pytest.approx(77, abs=5)


if __name__ == "__main__":
    main()
