"""Jupyter notebooks shipped with minian.

The notebooks live inside the installed package so that ``pip install minian``
gives you everything needed to read and edit them offline. Use the
``minian-notebooks`` CLI to copy a bundle into your working directory::

    minian-notebooks list
    minian-notebooks copy pipeline

Each bundle is a self-contained folder (notebook + ``assets/`` + ``README.md``)
so that copying it preserves every relative reference. Large demo datasets are
*not* bundled here; they are fetched on demand via :mod:`minian.data`.
"""
