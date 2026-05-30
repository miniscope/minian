import dask as da
import os
from importlib.metadata import version
from .utilities import custom_arr_optimize, custom_delay_optimize

try:
    __version__ = version("minian")
except:
    __version__ = "0.0.0"

da.config.set(
    array_optimize=custom_arr_optimize, delayed_optimize=custom_delay_optimize
)
# setting fuse width ref: https://github.com/dask/dask/issues/5105
da.config.set(
    **{
        "distributed.worker.memory.target": 0.8,
        "distributed.worker.memory.spill": 0.85,
        "distributed.worker.memory.pause": 0.9,
        "distributed.worker.memory.terminate": 0.95,
        "distributed.admin.log-length": 100,
        "distributed.scheduler.transition-log-length": 100,
        "optimization.fuse.ave-width": 3,
        # "optimization.fuse.subgraphs": False,
        # "distributed.scheduler.allowed-failures": 1,
        "array.slicing.split_large_chunks": False,
        # Force tasks-based rechunking instead of dask >=2025's default
        # P2P shuffle. P2P is more memory-efficient when the cluster has
        # spare worker capacity, but under memory pressure it surfaces as
        # `FutureCancelledError: _finalize_store-... cancelled for reason:
        # lost dependencies` (a paused/spilling worker has its in-flight
        # P2P task cancelled, which cascades to every dependent task).
        # Hit reliably in update_temporal where five concurrent
        # `var.chunk({"unit_id": 1})` rechunks race for memory. Tasks-
        # based rechunk spills more predictably and survives memory
        # pauses. Honored only on dask >=2024.x; older dask ignores it.
        "array.rechunk.method": "tasks",
    }
)
# ref: https://github.com/dask/dask/issues/3530
# on linux, after conda installing jemalloc, one can use the following line to
# get around threaded scheduler memory leak issue.
# os.environ["LD_PRELOAD"] = "~/.conda/envs/minian-dev/lib/libjemalloc.so"
# alternatively one can limit the malloc pool, which is the default for minian
os.environ["MALLOC_MMAP_THRESHOLD_"] = "16384"

