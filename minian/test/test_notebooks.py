"""Smoke-execution for bundled notebooks without a dedicated test.

The two heavy notebooks (pipeline, cross-registration) have dedicated tests
(``test_pipeline.py`` / ``test_cross_reg.py``) that execute them AND assert
golden values, so they are not re-executed here. Any *other* shipped notebook
is discovered automatically and smoke-executed so it cannot rot silently.
"""

import pytest

from ..notebooks import notebook_files
from ._notebook import execute_notebook

# Notebooks executed (with value assertions) by their own dedicated tests; the
# regression guarantee lives there, so we just skip re-running them here.
DEDICATED_TESTS = {
    "pipeline/pipeline.ipynb",
    "cross_registration/cross-registration.ipynb",
}

_SMOKE = [nb for nb in notebook_files() if nb not in DEDICATED_TESTS]


@pytest.mark.slow
@pytest.mark.skipif(not _SMOKE, reason="every shipped notebook has a dedicated test")
@pytest.mark.parametrize("relpath", _SMOKE)
def test_notebook_executes(relpath):
    """Every shipped notebook without a dedicated test at least runs end to end."""
    execute_notebook(relpath, relpath.rsplit("/", 1)[-1].removesuffix(".ipynb"))
