Cross-registration
==================

``cross-registration.ipynb`` aligns and matches cells across multiple recording sessions of the same animal, producing a cell-to-cell mapping table.

Running this notebook
---------------------

Copy the notebook out of the installed package and open it with Jupyter:

.. code-block:: console

    minian notebooks copy cross_registration
    jupyter notebook minian-notebooks/cross_registration/cross-registration.ipynb

See :doc:`../start_guide/install` for installation and :doc:`../start_guide/cli` for the full command reference.

The notebook fetches its two-session demo dataset automatically via :func:`minian.data.fetch` (cached and checksum-verified).

Using your own data
--------------------

To run the cross-registration on your own data, edit the ``dpath`` and ``f_pattern`` cell near the top of the notebook: point ``dpath`` at a directory of saved minian datasets and set ``f_pattern`` to match them (for example ``r"minian$"`` for the default ``zarr`` output format).

.. toctree::
   :numbered:
   :maxdepth: 1
   :glob:

   notebook_*
