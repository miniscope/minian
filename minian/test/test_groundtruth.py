"""Dedicated execution test for the ground-truth pipeline notebook.

The notebook needs ``minisim`` (the optional ``training`` extra, not a core
dependency), so the test is gated on that import: environments without the extra
skip it rather than failing on collection. It executes the ``quick`` default end
to end; the guarantee is "it runs", not golden values - recovery scores depend on
the minisim/minian versions and are checked inside the notebook itself.
"""

import pytest

pytest.importorskip("minisim")

from ._notebook import execute_notebook


@pytest.mark.slow
def test_pipeline_groundtruth_executes():
    """The ground-truth notebook runs end to end on its default dataset."""
    execute_notebook(
        "pipeline_groundtruth/pipeline_groundtruth.ipynb", "pipeline_groundtruth"
    )
