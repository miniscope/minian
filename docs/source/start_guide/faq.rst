Frequently Asked Questions
==========================

I don't know Python, can I still use the pipeline?
--------------------------------------------------

Although we like Python and we recommend gaining some understanding of the foundations, we do think it's possible to use pipeline as-is with minimal knowledge of python.
For a start, you can just change ``dpath`` as described in :ref:`tips-pipeline-set-path`.
Then you should be able to run all the cells in the pipeline without modifying anything.

However, chances are you will like to tweak some parameters.
In order to do this all you need to know is how Python dictionary works.
`Here <https://www.tutorialspoint.com/python/python_dictionary.htm>`_ is a quick tutorial.
The general syntax is:

.. code-block:: python

    param_stepX["parameter_name"] = value

where ``value`` maybe string, numbers, lists or in some cases another dictionary by itself, depending on the parameter you are changing.
Once you run this line the ``param_stepX`` dictionary will be updated and you can proceed to the step that use this dictionary.
The pipeline also has a quick real-world example in :ref:`tips-pipeline-loading-videos`.

Can I get a table of content for the pipeline?
----------------------------------------------

Yes! Jupyter has a `toc2 <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html>`_ extension that should work well with the notebook.
You can install it with:

.. code-block:: console

    uv pip install jupyter-contrib-nbextensions

Then enable the contrib extensions per the `jupyter_contrib_nbextensions docs <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html>`_ (for example ``jupyter contrib nbextension install --user``).

Can I have a version of pipeline without all the notes?
-------------------------------------------------------

Yes! all the tutorial cells in the pipeline are tagged with "tutorial".
We can use ``nbconvert`` to obtain a copy of the notebook with `all those cells removed <https://nbconvert.readthedocs.io/en/latest/removing_cells.html#removing-pieces-of-cells-using-cell-tags>`_.
For example, the following command will give you a clean pipeline:

.. code-block:: console

    jupyter nbconvert pipeline.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="tutorial" \
    --to notebook \
    --output pipeline_clean.ipynb

Just note that since this is a different copy of the pipeline, there might be extra work if you want to merge you edits on this clean copy with updates in the default pipeline notebook.

How can I monitor the progress of some time consuming steps?
------------------------------------------------------------

MiniAn uses Dask for parallel work; the cluster dashboard shows live tasks, memory, and cache.
See :ref:`tips/performance:Monitoring cluster` for the dashboard URL (port ``8787`` by default), how to read it, and related troubleshooting.

``KilledWorker``, what do I do?
--------------------------------------

Short answer: sometimes all you need is to try again.
If that doesn't work or if that's getting annoying, then increasing ``memory_limit`` of the ``cluster`` would most likely fix it.
See :ref:`killedworker` for more detail.

What do I do with the output of MiniAn?
---------------------------------------

Hopefully the tutorials in the pipeline would give you an idea of how to interprete each variable.
Then if you're happy to work with `xarray`, then you will feel right at home once you do :py:func:`open_minian <minian.utilities.open_minian>`.
See :ref:`tips/variables:Saving to other formats` for details on converting the output to other formats.
