Contributing to MiniAn
======================

We'd love feedback and contribution from the community!
:ref:`Fork and clone MiniAn from source <start_guide/install:Install from source>`, make you changes and submit a PR!
Below are some book-keeping notes.

Commit Messages
---------------

MiniAn is adopting `conventional commit <https://www.conventionalcommits.org>`_.
You can use `commitizen <https://commitizen-tools.github.io/commitizen/>`_ to check for the style or setup pre-commit hooks.
We also use `commitizen <https://commitizen-tools.github.io/commitizen/>`_ to automate the releasing process.
All development should be done on separate branches and squash-merge to `master`.

Code Style
----------

MiniAn use the `black coding style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_.
We also use a github action to enforce the style.
So watch out for automatic commits and avoid headache in confilicting history.

Creating release
----------------

#. ``pip install commitizen``
#. ``cz bump --dry-run`` and make note of new release tag
#. ``cz changelog --unreleased-version <TAG>`` with the tag noted in last step
#. edit `CHANGELOG.md` as desired
#. ``cz bump``

Packaging for PyPi
------------------

#. ``pip install --upgrade build``
#. ``python3 -m build``
#. ``pip install --upgrade twine``
#. ``python3 -m twine upload dist/*``

.. seealso:: `packaging <https://packaging.python.org/tutorials/packaging-projects/>`_

Packaging for conda-forge
-------------------------

#. fork and clone `https://github.com/conda-forge/minian-feedstock`
#. ``conda config --add channels conda-forge``
#. ``conda install conda-build``
#. ``conda install conda-smithy``
#. ``conda-build recipes/minian``
#. create a PR to upstream

Build documentation
-------------------

MiniAn use `numpy style docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
It also heavily rely on auto-generated notebooks and use a custom github-action-rtd workflow.

To build documentation locally run the following commands:

.. code-block:: console

    pip install -e .[doc]
    cd docs
    make html

This however does not include the auto-generated pages for `pipeline.ipynb` and `cross-registration.ipynb`.
To include those, create a folder `docs/source/artifact`.
Then copy the notebooks (preferably with output) and the `img` folder under there.