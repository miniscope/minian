Pipeline
========

``pipeline.ipynb`` is the main MiniAn analysis pipeline: load videos, pre-processing, motion correction, seed initialization, then CNMF spatial and temporal updates, and visualization.

Running this notebook
---------------------

Copy the notebook out of the installed package and open it with Jupyter:

.. code-block:: console

    minian notebooks copy pipeline
    jupyter notebook minian-notebooks/pipeline/pipeline.ipynb

See :doc:`../start_guide/install` for installation and :doc:`../start_guide/cli` for the full command reference.

The notebook fetches its demo movie automatically via :func:`minian.data.fetch` (cached and checksum-verified, no manual download step). The demo is a mouse CA1 recording acquired on a Miniscope V3, 5x temporally downsampled (10 ``msCam`` ``.avi`` files, 2000 frames at 480x752).

Using your own data
--------------------

To run the pipeline on your own recording, edit the ``dpath`` cell near the top of the notebook to point at your data directory.

.. toctree::
   :numbered:
   :maxdepth: 1
   :glob:

   notebook_*
