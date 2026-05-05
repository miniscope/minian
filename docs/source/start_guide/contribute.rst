Contributing to MiniAn
======================

We'd love feedback and contribution from the community!
:ref:`Fork and clone MiniAn from source <start_guide/install:Install from source>`, make your changes and submit a PR!
Below are some book-keeping notes.

Commit Messages
---------------

MiniAn is adopting `conventional commit <https://www.conventionalcommits.org>`_.
You can use `commitizen <https://commitizen-tools.github.io/commitizen/>`_ to check for the style or setup pre-commit hooks.
We also use `commitizen <https://commitizen-tools.github.io/commitizen/>`_ to automate the releasing process.
All development should be done on separate branches and squash-merge to `master`.

Code Style
----------

MiniAn follows the `Black code style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_.
Formatting is enforced with **`pre-commit`**, which runs Black on ``minian/`` (see ``.pre-commit-config.yaml`` at the repo root). CI runs the same hooks, so pull requests fail if anything under ``minian/`` is not Black-compliant.

**One-time setup** (from a clone, after installing dev dependencies):

.. code-block:: console

    uv sync --group dev
    uv run pre-commit install

That registers Git hooks so Black runs on staged files when you commit. To format or re-check the whole tree without committing:

.. code-block:: console

    uv run pre-commit run --all-files

If you use `mise <https://mise.jdx.dev/>`_ with this repository, ``mise run format`` runs the same ``pre-commit`` invocation across the repo.

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

    uv sync --group dev --extra docs
    uv run sphinx-build -M html docs/source docs/build

Alternatively, ``cd docs && make html`` after the same ``uv sync`` (it invokes Sphinx with this repo's ``Makefile``).

Read the Docs installs ``requirements/requirements-base.txt`` then ``requirements/requirements-doc.txt`` (see ``.readthedocs.yaml``). Those files are **generated** from ``pyproject.toml`` / ``uv.lock``; after changing dependencies, regenerate them from the repository root:

.. code-block:: console

    uv export --format requirements.txt --no-dev --no-annotate --no-header -o requirements/requirements-base.txt
    uv export --format requirements.txt --no-dev --extra docs --no-emit-project --no-annotate --no-header -o requirements/requirements-doc.txt

Each command refreshes ``uv.lock`` unless you pass ``--frozen``. The same two lines are wrapped as ``mise run export-reqs-base`` and ``mise run export-reqs-doc`` in ``.mise.toml``. Commit the updated ``requirements/*.txt`` (and ``uv.lock`` if it changed).

This however does not include the auto-generated pages for `pipeline.ipynb` and `cross-registration.ipynb`.
To include those, create a folder `docs/source/artifact`.
Then copy the notebooks (preferably with output) and the `img` folder under there.
