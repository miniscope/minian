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

The main features of Minian are exposed through `pipeline.ipynb` and `cross-registration.ipynb` `notebooks <https://jupyter.org/>`_.
You can use the following links to get the latest version of the two notebooks:

* Download `pipeline.ipynb <https://github.com/DeniseCaiLab/minian/raw/master/pipeline.ipynb>`_
* Download `cross-registration.ipynb <https://github.com/DeniseCaiLab/minian/raw/master/cross-registration.ipynb>`_

If you'd prefer specific version of them, head to `github release page <https://github.com/denisecailab/minian/releases>`_ to see all the released versions.

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