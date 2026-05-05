Understanding Performance
=========================

MiniAn heavily use the `dask <https://dask.org>`_ package to carry out all computations in parallel.
Here are some tips on working with this setup:

Dask cluster
------------

MiniAn by default would create a dask local cluster for you when you run **start cluster** in the pipeline notebook; see :ref:`tips-pipeline-start-cluster`.
The relevant line reads like the following:

.. code-block:: python

    cluster = LocalCluster(..., n_workers=4, memory_limit="2GB")

The argument ``n_workers`` controls the number of parallel processes (workers) that will be used for computation.
Almost all computations in MiniAn can benefit from parallel processing, so the more ``n_workers`` you have, the better performance in general.

However, to do its job each worker needs to reserve a certain amount of memory.
The maximum amount of memory **each worker** can have is controlled by ``memory_limit``.
Hence, the maximum amount of memory the whole cluster may consume is ``n_workers * memory_limit`` (which would be "8GB" in the example above), and you should make sure you have this amount of free RAM available.
Otherwise you might encounter ``MemoryError`` or killed kernel.

Monitoring cluster
------------------

Dask provide a dashboard for the cluster, which is very helpful for monitoring the progress of computations and the memory consumed in real time.
By default you can access it by navigating to `http://localhost:8787/status` in browser (the same way you navigate to jupyter notebook interface).
The address of the dashboard can be configured with the ``dashboard_address`` argument to :py:class:`distributed:distributed.LocalCluster`.
See :doc:`dask documentation <dask:diagnostics-distributed>` for how to interprete and configure the dashboard.

Here are some patterns you may observe from the dashboard that would be very valuable when debugging performance (usually memory) issues:

* Over the course of the pipeline the amount of cached data may increase a bit, but not indefinitely.
  If you are using the default setting, the cache should always be around *512MB* per worker when the cluster is idle (no task is running in the task stream).
  If this is not the case, you may see that some steps would result in ``KilledWorker`` exception when running continously from the beginning of the pipeline, but not when restarted from a "checkpoint" with a fresh kernel/cluster.
  In such cases it's very valuable to note down the amount of cache/available memory **before** running each step to find out which step is leaking memory.
* Sometimes when the worker is struggling with memory, it fall into an indefinite pause state instead of throwing a ``KilledWorker`` exception.
  When this happens you would see that all the cache "bars" turns orange and stays high, and at the same time no updates in the task stream panel.
  This is also likely accompanioned by messages like "memory usage is high but no data to store to disk" being constantly printed out in log/terminal.
  In such cases it makes no sense to keep waiting since it will never "finish", and it's best to just interrupt and potentially restart the kernel.
  Note that some long-running tasks (such as temporal update) may produce identical behavior.
  A good way to tell whether a task is still running or falled to the indefinite pause state is to check the CPU usage using your system monitor, which would be very low if the task is no longer running.
* By default dask would scatter the tasks evenly across all available workers.
  So once the cluster has a few dozen of tasks processed you should see them fill up the task stream panel evenly and form a "block", with number of rows equal to ``n_workers * threads_per_worker``.
  Normally this pattern should be stable during the whole pipeline.
  However, if some worker gets killed you will see the corresponding row would remain blank for the rest of the time, and at the same time some new row (corresponding to the new worker) would spawn on top of the existing "block" of streams.
  You should also be able to see message like "remove worker" / "add worker" in the log/terminal.
  When this happens, by default dask would retry the offending task a few times before actually throwing the ``KilledWorker`` exception.
  Sometimes the computations would finish successfully despite this and everything would be fine.
  However these "silently" killed workers tend to leave something in the cache, which can cause trouble for downstream tasks.
  Hence it's valuable to note down such incidents, especially when it cause issues later.

Chunked computation
-------------------

You may notice that each worker only needs a relatively small (comparing to the size of data) amount of memory to perform computations.
This is because almost all variables in MiniAn are :doc:`Dask Arrays <dask:array>`, which are composed of individual :doc:`Chunks <dask:array-chunks>`.
Each chunk has certain size and would occupy some amount of memory, which should absolutely be smaller than the ``memory_limit`` of each worker.
Computationis on each chunk oftentimes involve creating several intermediate variables that's of the same size as the input chunk.
Hence, usually the chunk size should be several folds smaller than ``memory_limit`` to be safe.
The downside of having too small chunk size is that your data would be divided into more number of chunks.
Since each chunk produce some overhead when reading/writing to disk as well as when generating computation graph, having too many chunks would hurt performance.

MiniAn try to find the best chunk size for you using :py:func:`get_optimal_chk <minian.utilities.get_optimal_chk>` during :ref:`tips-pipeline-loading-videos` in the pipeline notebook.
The default is to produce chunks that use around *256MB* of memory (controlled by ``csize`` argument), which is roughly 1/10 of the default ``memory_limit`` for each worker.
Hence, if you find your workers struggling with memory, and you don't have more physical RAM to spare to increase ``memory_limit``, you may consider decreasing the chunk size.
Conversely, if you have lots of RAM to spare and you believe you have too many chunks than necessary, you may consider increasing the chunk size (this is rarely necessary though).

Note that it's important to have consistent chunk size across different variables in a single run of the pipeline.
This is why :py:func:`get_optimal_chk <minian.utilities.get_optimal_chk>` is only executed once in the beginning of the pipeline and everything afterwards use the same ``chk`` dictionary.
If for some reason you have to restart the python kernel and the ``chk`` dictionary is lost, you can execute :py:func:`get_optimal_chk <minian.utilities.get_optimal_chk>` again to get the same chunk size.
You can also note down or save the ``chk`` dictionary for future uses.

.. _killedworker:

Dealing with ``KilledWorker``
-----------------------------

A ``KilledWorker`` exception happens when a worker is about to use memory that exceeds ``memory_limit``.
Note that this **does not** imply you are running out of RAM.
In fact the cluster is supposed to kill the workers before filling up your computer RAM if the ``memory_limit`` are set properly.
MiniAn try to minimize such incidents, and the default parameters has been tested successfully with ~60min miniscope v4 recordings on Linux.
However, unfortunately there is always inconsistencies between platforms/computers.
Also longer data may further increase the memory demands.
In any case, the first thing to do when you see a ``KilledWorker`` is to try to figure out the exact condition that this happens.
Does it always happen at a certain step?
Is it related to what steps you run and how much cache you have before the offending step?
Is it related to the size of input data?
Depending on the conditions the solution are usually one of the following:

#. Sometimes all you need is to try running the step again.
#. If there is no particular step that would result in the exception but you tend to get it once you have run several steps, then that usually indicate some build up in the cache, and you might need some :ref:`custom checkpoints <tips/variables:Custom checkpoints>`.
#. If you have free RAM to spare then increasing ``memory_limit`` would almost certainly solve the problem.
#. Otherwise you might have to limit the chunk size.

However, if you can find a **reproducible** case where the default pipeline/settings would fail for a reasonable sized data (<60min recording), please do not hesitate to `file a bug report on github <https://github.com/denisecailab/minian/issues/new/choose>`_.