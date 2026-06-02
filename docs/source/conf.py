# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../../"))
sys.path.append(os.path.abspath("../ext"))

# -- Custom scripts ----------------------------------------------------------
import dask.array


def custom_as_gufunc(signature=None, **kwargs):
    def _as_gufunc(pyfunc):
        return pyfunc

    return _as_gufunc


dask.array.as_gufunc = custom_as_gufunc

# -- Project information -----------------------------------------------------

project = "MiniAn"
copyright = f"2018-{datetime.datetime.now().year}, MiniAn Developers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "m2r2",
    "toctree",
    "ref_alias",
    "normalize_html_id",
    "nbsplit",
]

napoleon_use_rtype = False
napoleon_use_ivar = True
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autoclass_content = "both"
autosectionlabel_prefix_document = True

doctest_global_setup = """
import numpy as np
import pandas as pd
from minian.cross_registration import *
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
"""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "sparse": ("https://sparse.pydata.org/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "ffmpeg": ("https://kkroening.github.io/ffmpeg-python/", None),
    "skimage": ("https://scikit-image.org/docs/0.18.x/", None),
    "simpleitk": ("https://simpleitk.readthedocs.io/en/v2.0.0/", None),
    "medpy": ("https://loli.github.io/medpy/", None),
    "natsort": ("https://natsort.readthedocs.io/en/master/", None),
    "rechunker": ("https://rechunker.readthedocs.io/en/latest/", None),
}

ref_aliases = {
    "xr.DataArray": ("xarray.DataArray", "xr.DataArray"),
    "xr.Dataset": ("xarray.Dataset", "xr.Dataset"),
    "np.ndarray": ("numpy.ndarray", "np.ndarray"),
    "pd.DataFrame": ("pandas.DataFrame", "pd.DataFrame"),
    "pd.Series": ("pandas.Series", "pd.Series"),
    "nx.Graph": ("networkx.Graph", "nx.Graph"),
    "darr.Array": ("dask.array.Array", "darr.Array"),
}

nbsplit_dict = {
    os.path.abspath("artifact/pipeline.ipynb"): "pipeline",
    os.path.abspath("artifact/cross-registration.ipynb"): "cross_reg",
}
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
.. note::
    This page was generated from a Jupyter notebook.
    Not all interactive visualization will work on this web page.
    Consider :ref:`downloading the notebooks <start_guide/install:Getting notebooks and demos>` for full Python-backed interactivity.
"""

rtds_action_github_token = os.getenv("GITHUB_TOKEN")
if rtds_action_github_token:
    extensions.append("rtds_action")
    rtds_action_github_repo = "denisecailab/minian"
    rtds_action_path = "artifact"
    rtds_action_artifact_prefix = "notebooks-"
    rtds_action_error_if_missing = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["artifact"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = "pydata_sphinx_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]
