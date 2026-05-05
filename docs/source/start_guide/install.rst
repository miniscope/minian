Installation
============

MiniAn targets **Python 3.12** (see the repository ``pyproject.toml``). You need **FFmpeg** on ``PATH`` for video I/O. Install it with your OS package manager or follow the `FFmpeg download page <https://ffmpeg.org/download.html>`_.

Install from source with uv
-----------------------------

This is the usual path for **development** and for running the latest code from Git.

`uv <https://docs.astral.sh/uv/>`_ reads ``pyproject.toml`` and manages a project virtual environment (``.venv``). From a clone of the repository:

.. code-block:: console

    git clone https://github.com/DeniseCaiLab/minian.git
    cd minian
    uv sync

That installs the runtime dependencies needed to import MiniAn and run pipelines.

Optional dependency groups:

.. code-block:: console

    uv sync --group dev

adds developer tooling used in CI (pytest, pre-commit for Black on ``minian/``, maturin, mypy, etc.). To include **Sphinx** and documentation extras as well:

.. code-block:: console

    uv sync --group dev --extra docs

**Rust extension** (``minian.minian_rs``, optional FFT acceleration): install a `Rust toolchain <https://rustup.rs/>`_, then from the repository root:

.. code-block:: console

    uv run maturin develop --manifest-path src-rust/Cargo.toml

If you use `mise <https://mise.jdx.dev/>`_ with the repo ``.mise.toml``, the same step is ``mise run rs-dev``. If the extension is not built, MiniAn falls back to the legacy Python/PyFFTW path automatically.

Run entry points through ``uv run`` so they use the project environment, for example ``uv run minian-pipeline --help``, ``uv run minian-cross-reg --help``, or ``uv run jupyter notebook``.

Install from PyPI
-----------------

When releases are published to PyPI, you can install into any Python 3.12 environment:

.. code-block:: console

    pip install minian

or:

.. code-block:: console

    uv pip install minian

You still need FFmpeg (and any system libraries your platform requires) separately.

**pipx-style with uv:** ``uv tool install`` keeps MiniAn in its own environment and puts the package’s console scripts (``minian-pipeline``, ``minian-cross-reg``, ``minian-install``, etc.) on your ``PATH``, similar to `pipx <https://pipx.pypa.io/>`_:

.. code-block:: console

    uv tool install minian

Later you can run ``uv tool upgrade minian`` or ``uv tool uninstall minian``. See the uv `tools guide <https://docs.astral.sh/uv/guides/tools/>`_ for details.

Getting notebooks and demo data
---------------------------------

The main walkthroughs live under ``notebooks/`` in the repository (``pipeline.ipynb``, ``cross-registration.ipynb``), together with figures under ``img/``.

If you **cloned** the repo, those paths are already on disk and you can skip downloading.

Otherwise, after installing MiniAn, fetch notebooks and/or demo movies into the current directory. From a clone, use:

.. code-block:: console

    uv run minian-install --notebooks
    uv run minian-install --demo

If you installed MiniAn with ``pip`` / ``uv pip`` into another environment, activate it and run ``minian-install --notebooks`` and ``minian-install --demo``. If you used ``uv tool install minian``, those commands are already on ``PATH``—run them without ``uv run``.

Use ``--dest DIR`` (short ``-d``) to choose the download directory, and ``-v BRANCH_OR_TAG`` to pull files from another Git ref (default is the installed package version).

You can also download files directly from GitHub, for example from ``main``:

* `notebooks/pipeline.ipynb <https://github.com/DeniseCaiLab/minian/raw/main/notebooks/pipeline.ipynb>`_
* `notebooks/cross-registration.ipynb <https://github.com/DeniseCaiLab/minian/raw/main/notebooks/cross-registration.ipynb>`_

For a specific release, use the `GitHub releases page <https://github.com/DeniseCaiLab/minian/releases>`_.

Start the pipeline
------------------

**Jupyter:** from a clone use ``uv run jupyter notebook``, then open ``notebooks/pipeline.ipynb`` (or the path where you ran ``minian-install``). With ``pip`` / ``uv pip``, activate that environment and run ``jupyter notebook``. A bare ``uv tool install minian`` may not include the Jupyter UI—either stick with a clone + ``uv sync`` for notebooks or add Jupyter alongside MiniAn, e.g. ``uv tool install minian --with jupyter``.

**Headless:** from a clone use ``uv run minian-pipeline`` and ``uv run minian-cross-reg``. With ``uv tool install``, call ``minian-pipeline`` and ``minian-cross-reg`` directly. ``-d`` / ``--data`` defaults to the current directory—point it at ``./demo_movies`` or ``./demo_data`` in a checkout to try the demos. Full ``--help`` text for all commands, plus pipeline-related environment variables, is on :doc:`cli`.

See :doc:`../pipeline/index` and :doc:`../cross_reg/index` for expected behavior when using the hosted demo data.
