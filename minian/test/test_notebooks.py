"""Coverage + smoke-execution for bundled notebooks.

The two heavy notebooks (pipeline, cross-registration) have dedicated
golden-value tests (``test_pipeline.py`` / ``test_cross_reg.py``) and are not
re-executed here. Any *other* bundled notebook (examples, training, ...) is
discovered automatically and smoke-executed so it cannot rot silently.
"""

import pytest

from ._notebook import discover_notebooks, execute_notebook

# Notebooks executed (with assertions) by their own dedicated tests.
GOLDEN = {
    "pipeline/pipeline.ipynb",
    "cross_registration/cross-registration.ipynb",
}

_EXTRA = [nb for nb in discover_notebooks() if nb not in GOLDEN]


def test_golden_notebooks_present():
    """The golden-tested notebooks are actually shipped where tests expect."""
    found = set(discover_notebooks())
    missing = GOLDEN - found
    assert not missing, f"expected bundled notebooks are missing: {sorted(missing)}"


@pytest.mark.skipif(not _EXTRA, reason="no non-golden notebooks to smoke-test yet")
@pytest.mark.parametrize("relpath", _EXTRA)
def test_extra_notebook_executes(relpath):
    """Every non-golden bundled notebook at least runs end to end."""
    execute_notebook(relpath, relpath.rsplit("/", 1)[-1].removesuffix(".ipynb"))
