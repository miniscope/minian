import os

import psutil
import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set env vars for dask resource limits"""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (2**30)
    os.environ["MINIAN_NWORKERS"] = "1"
    os.environ["MINIAN_MEM_LIMIT"] = f"{total_gb * 0.75:.2f}GB"
    os.environ["MINIAN_INTERACTIVE"] = "False"
