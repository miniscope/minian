Installation
============

You need **FFmpeg** on ``PATH`` for video I/O (``ffmpeg`` and ``ffprobe``). Install
it with your OS package manager or follow the `FFmpeg download page
<https://ffmpeg.org/download.html>`_.

Install with pip
----------------

MiniAn is available on `PyPI <https://pypi.org/project/minian/>`_:

.. code-block:: console

    # regular installation (most platform compatible)
    python -m pip install minian
    # installation with optimized numba routines
    python -m pip install 'minian[numba]'

We recommend installing into a fresh virtual environment. A pip install does not
include FFmpeg, so make sure it is on your ``PATH`` (see above).

.. note::

   **Windows:** if the install fails while unpacking deeply nested files (such as
   JupyterLab widget extensions), enable long path support and reinstall. Windows'
   default 260-character path limit (``MAX_PATH``) truncates these paths. Set the
   registry value
   ``HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`` to ``1``
   (requires admin), or run as administrator in PowerShell::

       Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1

Install with conda
------------------

MiniAn is also on `conda-forge`. Installing this way pulls in FFmpeg
automatically, so it is convenient if you do not already have it:

.. code-block:: console

    conda create -y -n minian
    conda activate minian
    conda install -y -c conda-forge minian

You can use `mamba <https://mamba.readthedocs.io/en/latest/>`_ as a faster
drop-in replacement for ``conda`` when solving dependencies:

.. code-block:: console

    mamba install -y -c conda-forge minian

Install from source
-------------------

Install from the GitHub repo to track the latest development or to contribute to
MiniAn:

.. code-block:: console

    git clone https://github.com/miniscope/minian.git
    cd minian/
    python -m pip install -e .

This gives you an editable copy of MiniAn, so any change you make to the source
is reflected in your pipeline. Maintainers can use ``pdm install`` to set up the
locked development environment.

Getting notebooks and demos
---------------------------

The main features of MiniAn are exposed through the `pipeline.ipynb` and `cross-registration.ipynb` `notebooks <https://jupyter.org/>`_, backed by demo datasets hosted on Zenodo.
Both are managed through the single ``minian`` command line tool, which has two command groups: ``minian notebooks`` for the bundled notebooks and ``minian data`` for the demo datasets.
See :doc:`cli` for the full command reference.

Notebooks
~~~~~~~~~

The notebooks ship inside the installed package. Copy one (notebook plus its figures) into your current folder with:

.. code-block:: console

    minian notebooks list              # show the available notebooks
    minian notebooks copy pipeline     # -> ./minian-notebooks/pipeline/

Use ``-o/--output DIR`` to copy somewhere other than ``./minian-notebooks/``, or ``--all`` to copy every notebook.

Data
~~~~

Each notebook fetches its demo recording automatically on first run (downloaded once, then cached and checksum-verified).
You can also manage the demo datasets directly:

.. code-block:: console

    minian data list                      # show datasets + sizes
    minian data download pipeline-demo    # download and cache the dataset
    minian data path pipeline-demo        # print the local cached path

The demo datasets are hosted on Zenodo, each with its own citable DOI; see :mod:`minian.data` for details.

If you choose to `Install from source`_ you already have a local copy of everything and can check out different versions using `git`.

Start the pipeline
------------------

And that's it!
Once you have installed MiniAn and copied out a notebook (see above), start the jupyter interface on it with:

.. code-block:: console

    jupyter notebook minian-notebooks/pipeline/pipeline.ipynb

(Remember to activate the environment if your computer complains about command not found.)

You can then either run the notebook, or refer to :doc:`../pipeline/index` and :doc:`../cross_reg/index` for some ideas about expected outcomes when running with demo data.