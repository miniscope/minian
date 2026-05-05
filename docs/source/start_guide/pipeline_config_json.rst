.. _pipeline-config-json:

Pipeline JSON (``minian_config.json``)
======================================

The headless driver and :func:`minian.config.load_pipeline_config` read a single JSON file—by default :data:`minian.constants.MINIAN_CONFIG_FILENAME` next to your data or current working directory. The on-disk shape matches :class:`minian.config.PipelineConfig`: top-level keys are dataclass fields; nested ``param_*`` objects are passed as ``**kwargs`` into the functions listed below (see each function’s API page for every allowed key and default).

Export built-in defaults with ``uv run minian-pipeline-defaults`` (see :doc:`cli`). For CLI flags and logging env vars, see :ref:`pipeline configuration and environment <cli-pipeline-env>` in :doc:`cli`.

Effective run record (``minian_config_effective.json``)
-------------------------------------------------------

This is **not** a second config file you maintain by hand. On each **successful** headless pipeline finish, the driver writes :data:`minian.constants.MINIAN_CONFIG_EFFECTIVE_FILENAME` next to ``--data`` (same directory convention as the main JSON). The payload is built by :func:`minian.config.build_pipeline_effective_record` and is a small **audit snapshot**: minian version, a digest of built-in defaults, the sparse diff between the **resolved** effective config and those defaults (e.g. absolutized ``intpath``), and how the Dask ``LocalCluster`` was actually sized (``n_workers``, memory limit, threads, chunk budget, resolved CPU ratio).

``timestamp`` is new every run. Other fields stay the same when you rerun with the same install, machine, ``minian_config.json``, and data path; they change when defaults in code, hardware-derived worker counts, or your JSON differ.

Paths and run layout
--------------------

``intpath`` (string, required in practice)
   Absolute or relative path to the **Zarr scratch tree** (intermediate arrays). The driver overwrites this with a path under ``--data``; :meth:`minian.config.PipelineConfig.__post_init__` always absolutizes it. Downstream code reads it via :func:`minian.config.get_active_pipeline_config` after :meth:`minian.config.PipelineConfig.apply_environment`.

``param_save_minian`` (object)
   Keyword arguments for :func:`minian.utilities.save_minian` when writing merged results (not intermediate-only writes). Typical keys:

   - ``meta_dict`` — dimension metadata for saved datasets (e.g. session / animal indices from folder depth).
   - ``overwrite`` — whether to replace existing Zarr stores.
   - ``dpath`` — merged Minian output root; the driver sets this under ``--data`` at run time, so JSON may omit it or use a placeholder.

Subset and UI-ish fields
------------------------

``subset`` (object)
   Per-dimension slices applied to the raw movie after load (e.g. ``frame``). In JSON, slices use the ``{"__slice__": [start, stop, step]}`` encoding produced by :func:`minian.config.pipeline_config_to_jsonable`.

``subset_mc`` (object or null)
   Subset passed to motion estimation (:func:`minian.motion_correction.estimate_motion`). ``null`` means use the full ``varr_ref`` selection.

``interactive`` (bool)
   Reserved for notebook-style tooling; the headless pipeline does not branch on it.

``output_size`` (int)
   Used by visualization helpers (e.g. export / viewers), not by the core CNMF driver loop.

Dask cluster and CPUs
---------------------

``n_workers`` (int or null)
   If set, ``LocalCluster(n_workers=…)`` uses exactly this count (at least 1). If ``null``, worker count is derived from CPUs using ``reserve_cores_for_os`` and ``worker_cpu_ratio`` (see :meth:`minian.config.PipelineConfig.resolved_n_workers`).

``reserve_cores_for_os`` (int)
   Logical CPUs reserved for the OS when auto-picking ``n_workers``.

``worker_cpu_ratio`` (float or null)
   Fraction in ``(0, 1]`` for CPU-based worker count when ``n_workers`` is ``null``. ``null`` uses the built-in default ratio (``2/3``).

``dask_worker_memory`` (string)
   ``memory_limit`` string for each Dask worker (e.g. ``"2GB"``).

``dask_threads_per_worker`` (int)
   ``threads_per_worker`` for ``LocalCluster`` (minimum 1 after load).

``dask_chunk_target_mb`` (int)
   Target chunk size in megabytes for :func:`minian.utilities.get_optimal_chk` when choosing ``chk`` for the pipeline (minimum 1 after load).

BLAS / OpenMP
-------------

``thread_env`` (object, string keys and string values)
   Applied to ``os.environ`` by :meth:`minian.config.PipelineConfig.apply_environment` (unless ``blas_threads=`` is passed there). Typically ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``.

Algorithm parameters (``param_*``)
----------------------------------

Each ``param_*`` field is a JSON object. The CNMF driver unpacks them with :meth:`minian.config.PipelineConfig.algorithm_param_dicts` and passes them to exactly one call site in :mod:`minian.pipelines.cnmf_process`. For **full** keyword lists and semantics, open the linked function.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - JSON field
     - Passed to
   * - ``param_load_videos``
     - :func:`minian.utilities.load_videos`
   * - ``param_denoise``
     - :func:`minian.preprocessing.denoise`
   * - ``param_background_removal``
     - :func:`minian.preprocessing.remove_background`
   * - ``param_estimate_motion``
     - :func:`minian.motion_correction.estimate_motion`
   * - ``param_seeds_init``
     - :func:`minian.initialization.seeds_init`
   * - ``param_pnr_refine``
     - :func:`minian.initialization.pnr_refine`
   * - ``param_ks_refine``
     - :func:`minian.initialization.ks_refine`
   * - ``param_seeds_merge``
     - :func:`minian.initialization.seeds_merge`
   * - ``param_initialize``
     - :func:`minian.initialization.initA` (``initC`` uses fixed behavior in the driver)
   * - ``param_init_merge``
     - :func:`minian.cnmf.unit_merge` after initialization
   * - ``param_get_noise``
     - :func:`minian.cnmf.get_noise_fft`
   * - ``param_first_spatial``
     - :func:`minian.cnmf.update_spatial` (first spatial update in the driver)
   * - ``param_first_temporal``
     - :func:`minian.cnmf.update_temporal` (first full temporal pass)
   * - ``param_first_merge``
     - :func:`minian.cnmf.unit_merge` (merge after first temporal)
   * - ``param_second_spatial``
     - :func:`minian.cnmf.update_spatial` (second spatial pass)
   * - ``param_second_temporal``
     - :func:`minian.cnmf.update_temporal` (second full temporal pass)

**Note:** Spatial kwargs are forwarded into a small wrapper that persists ``C`` / ``C_chk`` under ``intpath``; behavior matches the public ``update_spatial`` options (e.g. ``dl_wnd``, ``sparse_penal``, ``size_thres``, ``in_memory``).

See also
--------

- :class:`minian.config.PipelineConfig` — field defaults in code.
- :func:`minian.config.load_pipeline_config` / :func:`minian.config.pipeline_config_to_jsonable` — merge and export rules.
- Demo file ``demo_movies/minian_config.json`` in the repository for a full example tree.
