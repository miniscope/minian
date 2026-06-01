import os

import psutil
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--with-notebooks",
        action="store_true",
        help="Include slow tests that execute notebooks",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--with-notebooks"):
        skip_notebooks = pytest.mark.skip(reason="need --with-notebooks option to run")
        for item in items:
            if item.get_closest_marker("notebook"):
                item.add_marker(skip_notebooks)


def pytest_sessionstart(session):
    """Set env vars for dask resource limits"""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (2**30)
    os.environ["MINIAN_NWORKERS"] = "1"
    os.environ["MINIAN_MEM_LIMIT"] = f"{total_gb * .75:.2f}GB"
    os.environ["MINIAN_INTERACTIVE"] = "False"
