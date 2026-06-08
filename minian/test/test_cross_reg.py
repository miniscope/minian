import pandas as pd
import pytest

from ._notebook import execute_notebook


@pytest.mark.slow
def test_cross_reg_notebook(clean_dataset_outputs):
    # The notebook fetches the two-session demo and writes its outputs back into
    # that dataset directory, so resolve it (via clean_dataset_outputs, which
    # clears stale outputs first and cleans up on teardown) and read the outputs
    # from there.
    dpath = clean_dataset_outputs("cross-reg-sessions")
    execute_notebook("cross_registration/cross-registration.ipynb", "cross-registration")

    assert (dpath / "shiftds.nc").exists()
    assert (dpath / "cents.pkl").exists()
    assert (dpath / "mappings.pkl").exists()
    # Read back the notebook's own outputs.
    cents = pd.read_pickle(dpath / "cents.pkl")
    mappings = pd.read_pickle(dpath / "mappings.pkl")
    assert len(cents) == 508
    # Use a relative tolerance: the exact centroid sums drift slightly
    # (~0.005%) across numpy/scipy/scikit-image versions.
    assert cents["height"].sum() == pytest.approx(99091, rel=1e-3)
    assert cents["width"].sum() == pytest.approx(213627, rel=1e-3)
    # The number of matched cell-pairs drifts by one or two across
    # library versions; check the structure exactly but the counts loosely.
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
