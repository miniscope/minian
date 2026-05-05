import dask as da
import os

# When running under IPython/Jupyter, attach a stderr StreamHandler so
# ``log.info`` from ``minian.*`` shows up without a separate setup cell.
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        from .utilities import configure_logging as _configure_logging

        _configure_logging()
except (ImportError, NameError):
    pass

from ._version import get_package_version
from .constants import MINIAN, MINIAN_CONFIG_EFFECTIVE_FILENAME, MINIAN_CONFIG_FILENAME
from .utilities import (
    configure_logging,
    custom_arr_optimize,
    custom_delay_optimize,
)

__all__ = [
    "__version__",
    "MINIAN",
    "MINIAN_CONFIG_EFFECTIVE_FILENAME",
    "MINIAN_CONFIG_FILENAME",
    "configure_logging",
    "custom_arr_optimize",
    "custom_delay_optimize",
]
__version__ = get_package_version()

da.config.set(
    array_optimize=custom_arr_optimize, delayed_optimize=custom_delay_optimize
)
# setting fuse width ref: https://github.com/dask/dask/issues/5105
da.config.set(
    {
        "distributed.worker.memory.target": 0.8,
        "distributed.worker.memory.spill": 0.85,
        "distributed.worker.memory.pause": 0.9,
        "distributed.worker.memory.terminate": 0.95,
        "distributed.admin.log-length": 100,
        # Formerly `distributed.scheduler.transition-log-length` (deprecated).
        "distributed.admin.low-level-log-length": 100,
        "optimization.fuse.ave-width": 3,
        # "optimization.fuse.subgraphs": False,
        # "distributed.scheduler.allowed-failures": 1,
        "array.slicing.split_large_chunks": False,
    }
)
# ref: https://github.com/dask/dask/issues/3530
# on linux, after conda installing jemalloc, one can use the following line to
# get around threaded scheduler memory leak issue.
# os.environ["LD_PRELOAD"] = "~/.conda/envs/minian-dev/lib/libjemalloc.so"
# alternatively one can limit the malloc pool, which is the default for minian
os.environ["MALLOC_MMAP_THRESHOLD_"] = "16384"
