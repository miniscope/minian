Installation
============

You need **FFmpeg** on ``PATH`` for video I/O (``ffmpeg`` and ``ffprobe``). Install
it with your OS package manager or follow the `FFmpeg download page
<https://ffmpeg.org/download.html>`_.

Install using conda
-------------------

MiniAn is available on `conda-forge` and this is the recommended way to get MiniAn.
Before you start though, we highly recommend creating an empty environment for MiniAn:

.. code-block:: console

    conda create -y -n minian
    conda activate minian

See `conda start guide <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ for more detail.

After you have created and activated an environment, you can install MiniAn with:

.. code-block:: console

    conda install -y -c conda-forge minian

and Done!

Alternatively, you can use `mamba <https://mamba.readthedocs.io/en/latest/>`_ to install minian, which usually provides faster speed when solving the dependencies.
To do so, you first need to install `mamba`, either in minian environment or in your base environment.

.. code-block:: console

    conda install -y -c conda-forge mamba

After this, you can use `mamba` as a drop-in replacement command for `conda` to install minian:

.. code-block:: console

    mamba install -y -c conda-forge minian

Install from source
-------------------

You can install MiniAn directly using github repo.
This is helpful if you want to checkout latest development, or to contribute to MiniAn.
Run the following to obtain a full copy of MiniAn repo and setup necessary dependencies.

.. code-block:: console

    git clone https://github.com/DeniseCaiLab/minian.git
    cd minian/
    conda env create -n minian -f environment.yml

You can then activate the environment and start running the notebooks.
Note that if you install in this way you will have a local copy of MiniAn scripts, and any modification made to those scripts will be reflect in your pipeline.

Getting notebooks and demos
---------------------------

The main features of MiniAn are exposed through the `pipeline.ipynb` and `cross-registration.ipynb` `notebooks <https://jupyter.org/>`_.
These notebooks ship **inside** the installed package, so once MiniAn is installed you already have them; no separate download is needed.

Copy a notebook bundle (notebook plus its figures and README) into your current folder with the ``minian-notebooks`` command:

.. code-block:: console

    minian-notebooks list              # show the available bundles
    minian-notebooks copy pipeline     # -> ./minian-notebooks/pipeline/

Each notebook fetches its demo recording automatically on first run (downloaded once, then cached and verified against a checksum), so there is no separate demo-install step.
You can also fetch demo datasets yourself with the ``minian-data`` command:

.. code-block:: console

    minian-data list                          # show datasets + sizes
    minian-data download pipeline-demo --to .  # download into the current folder

The demo datasets are hosted on Zenodo (each with its own citable DOI); see :mod:`minian.data` for details.

.. note::
    The older ``minian-install --notebooks`` / ``minian-install --demo`` commands still work as thin aliases for ``minian-notebooks`` and ``minian-data``.

Note that if you choose to `Install from source`_ you already have a local copy of everything and can check out different versions using `git`.

Start the pipeline
------------------

And that's it!
Once you have installed MiniAn and copied out a notebook (see above), start the jupyter interface on it with:

.. code-block:: console

    jupyter notebook minian-notebooks/pipeline/pipeline.ipynb

(Remember to activate the environment if your computer complains about command not found.)

You can then either run the notebook, or refer to :doc:`../pipeline/index` and :doc:`../cross_reg/index` for some ideas about expected outcomes when running with demo data.