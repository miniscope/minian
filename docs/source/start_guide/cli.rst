Command line interface
======================

MiniAn installs a single ``minian`` command with two command groups:

- ``minian notebooks`` lists and copies the notebooks bundled with the package.
- ``minian data`` lists, downloads, and locates the demo datasets (hosted on Zenodo, fetched on demand, cached, and checksum-verified).

The full set of commands and options is below.

.. argparse::
   :module: minian.cli
   :func: build_parser
   :prog: minian
