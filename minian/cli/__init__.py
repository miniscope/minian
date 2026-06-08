"""The unified ``minian`` command line interface.

One entrypoint with subcommand groups::

    minian data list
    minian data download pipeline-demo
    minian notebooks list
    minian notebooks copy pipeline

The historical ``minian-data`` / ``minian-notebooks`` / ``minian-install``
commands are kept as thin deprecated aliases (see :func:`data_main`,
:func:`notebooks_main`, and :mod:`minian.install`).
"""

import argparse
import sys
import warnings

from . import data, notebooks

__all__ = ["build_parser", "main", "data_main", "notebooks_main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``minian`` parser (also used to render CLI docs)."""
    parser = argparse.ArgumentParser(
        prog="minian",
        description="MiniAn command line tools: fetch demo data and copy out notebooks.",
    )
    sub = parser.add_subparsers(dest="group", required=True)
    data.add_subparser(sub)
    notebooks.add_subparser(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


def _warn_alias(old: str, new: str) -> None:
    """Emit a real DeprecationWarning (for test suites / ``-W``) plus a note."""
    msg = f"`{old}` is deprecated and will be removed in a future release; use `{new}`."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    print(f"Note: {msg}", file=sys.stderr)


def data_main(argv=None):
    """Deprecated alias for ``minian data`` (the ``minian-data`` entrypoint)."""
    _warn_alias("minian-data", "minian data")
    main(["data", *(sys.argv[1:] if argv is None else argv)])


def notebooks_main(argv=None):
    """Deprecated alias for ``minian notebooks`` (the ``minian-notebooks`` entrypoint)."""
    _warn_alias("minian-notebooks", "minian notebooks")
    main(["notebooks", *(sys.argv[1:] if argv is None else argv)])
