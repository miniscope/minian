Command-line tools
==================

MiniAn registers several console scripts in ``pyproject.toml`` (``[project.scripts]``). From a clone, run them with ``uv run <command>``; if MiniAn is installed in an active environment or via ``uv tool install``, use the command name directly.

You can also invoke modules as ``uv run python -m minian.pipelines.cnmf_process``, ``python -m minian.pipelines.cross_reg``, or ``python -m minian.install`` (same entry points as ``minian-pipeline``, ``minian-cross-reg``, and ``minian-install``).

``minian-pipeline``
-------------------

Headless CNMF pipeline with a local Dask cluster (``minian.pipelines.cnmf_process``).

.. code-block:: text

   $ uv run minian-pipeline -h
   usage: minian-pipeline [-h] [-d DATA] [-c PATH] [--worker-cpu-ratio RATIO]

   Run minian headless pipeline with a Dask LocalCluster.

   options:
     -h, --help            show this help message and exit
     -d DATA, --data DATA  Directory containing input videos (absolutized). Default: "." (current working directory).
     -c PATH, --config PATH
                           Pipeline JSON (see PipelineConfig). Default: minian_config.json in the current working directory if present; else built-in defaults (those defaults are written to <data>/minian_config.json at run start).
     --worker-cpu-ratio RATIO
                           When pipeline JSON leaves n_workers null: fraction of (logical CPUs − reserve) used as LocalCluster n_workers. If omitted, use JSON worker_cpu_ratio or default 2/3.

``minian-cross-reg``
--------------------

Cross-session registration over session folders containing ``minian.nc`` (``minian.pipelines.cross_reg``).

.. code-block:: text

   $ uv run minian-cross-reg -h
   usage: minian-cross-reg [-h] [-d DATA] [--param-dist PIXELS]

   Cross-register Minian outputs across sessions (writes mappings.pkl, cents.pkl,
   shiftds.nc under the data directory).

   options:
     -h, --help            show this help message and exit
     -d DATA, --data DATA  Directory containing session subfolders with minian
                           results (e.g. session1/minian.nc). Default: "."
     --param-dist PIXELS   Keep only cell pairs whose centroid Euclidean distance
                           (height/width coordinates) is strictly less than this
                           value in pixels (default: 5).

``minian-pipeline-defaults``
----------------------------

Writes the default :class:`minian.config.PipelineConfig` as JSON (``minian.config:main``)—for editing, diffing, or checking in next to a dataset. Output filename is ``minian_config.json`` (see ``minian.constants.MINIAN_CONFIG_FILENAME``).

.. code-block:: text

   $ uv run minian-pipeline-defaults -h
   usage: minian-pipeline-defaults [-h] [--dest DIR] [--stdout] [--resolve-paths]
                                   [--include-resolved-workers]

   Write default :class:`PipelineConfig` as JSON (for notebooks / pipeline drivers).

   options:
     -h, --help            show this help message and exit
     --dest DIR, -d DIR    Directory where minian_config.json is written (created if needed). Default: current directory. Use --stdout instead of creating a file.
     --stdout              Print JSON to stdout instead of writing the file.
     --resolve-paths       Use absolute intpath and param_save_minian['dpath'] (if set) before export.
     --include-resolved-workers
                           Add resolved_n_workers and resolved_worker_cpu_ratio (CPU-based from JSON fields).

``minian-install``
------------------

Downloads notebooks and/or demo assets from GitHub into ``--dest`` (``minian.install``).

.. code-block:: text

   $ uv run minian-install -h
   usage: minian-install [-h] [--notebooks] [--demo] [-v V] [--dest DIR]

   options:
     -h, --help          show this help message and exit
     --notebooks         Installs the notebooks
     --demo              Installs the demo data
     -v V                Git repo branch or tag name, default 2.0.0
     --dest DIR, -d DIR  Directory to download into (default: current working directory)

.. _cli-pipeline-env:

Pipeline configuration and environment
----------------------------------------

Dask ``LocalCluster`` sizing (``n_workers``, ``worker_cpu_ratio``, ``reserve_cores_for_os``, ``dask_worker_memory``, ``dask_threads_per_worker``, ``dask_chunk_target_mb``) is read from **pipeline JSON** (:class:`~minian.config.PipelineConfig`), not from ``MINIAN_*`` environment variables. Edit ``minian_config.json`` next to your data (or pass ``-c`` / ``--config``) and call :meth:`~minian.config.PipelineConfig.apply_environment` in notebooks the same way the CLI driver does.

CLI and logging still honor:

* **MINIAN_LOG_LEVEL** — Log level for CLI entrypoints that configure logging (default ``INFO``).
* **NO_COLOR** — When non-empty, log output skips ANSI colors on terminals (`no-color.org <https://no-color.org/>`_).

The driver registers the configured intermediate directory (under ``--data`` by default) via :meth:`minian.config.PipelineConfig.apply_environment` together with the fields above.

See also :doc:`install`, :doc:`pipeline_config_json`, :doc:`../pipeline/index`, and :doc:`../cross_reg/index`.
