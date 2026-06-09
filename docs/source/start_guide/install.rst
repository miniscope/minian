Installation
============

You need **FFmpeg** on ``PATH`` for video I/O (``ffmpeg`` and ``ffprobe``). Install
it with your OS package manager or follow the `FFmpeg download page
<https://ffmpeg.org/download.html>`_.

Install with pip
----------------

MiniAn is available on `PyPI <https://pypi.org/project/minian/>`_:

.. code-block:: console

    python -m pip install minian

We recommend installing into a fresh virtual environment. A pip install does not
include FFmpeg, so make sure it is on your ``PATH`` (see above).

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

The main features of Minian are exposed through `pipeline.ipynb` and `cross-registration.ipynb` `notebooks <https://jupyter.org/>`_.
You can use the following links to get the latest version of the two notebooks:

* Download `pipeline.ipynb <https://github.com/miniscope/minian/raw/master/pipeline.ipynb>`_
* Download `cross-registration.ipynb <https://github.com/miniscope/minian/raw/master/cross-registration.ipynb>`_

If you'd prefer specific version of them, head to `github release page <https://github.com/miniscope/minian/releases>`_ to see all the released versions.

Alternatively, MiniAn also come with convenient scripts to help you download notebooks and demos into your current folder.
Run the following (in your activated environment if any) to get the notebooks:

.. code-block:: console
    
    minian-install --notebooks

Additionally, we also hosted some small demo data that works with the notebooks.
Once you obtained these data, you should be able to run the two notebooks locally without modifying anything.
Run the following script to get demo data:

.. code-block:: console

    minian-install --demo

The script can also help you get files from different branchs.
See ``minian-install --help`` for more detail.

Note that if you choose to `Install from source`_ you would already have a local copy of everything and you can also checkou different version of them using `git`.
You can skip this step altogether.

Start the pipeline
------------------

And that's it!
Once you have installed MiniAn and obtained a copy of notebooks through any methods above, you can then start the jupyter notebook interface with:

.. code-block:: console

    jupyter notebook

(Remeber to activate the environment if your computer complain about command not found)

You can then either run the notebook, or refer to :doc:`../pipeline/index` and :doc:`../cross_reg/index` for some ideas about expected outcomes when running with demo data.