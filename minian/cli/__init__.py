"""The unified ``minian`` command line interface.

One entrypoint with subcommand groups::

    minian data list
    minian data download pipeline-demo
    minian notebooks list
    minian notebooks copy pipeline

The historical ``minian-install`` command (which used to fetch notebooks and
demo data over HTTP at install time) is kept as a thin deprecated alias; see
:mod:`minian.install`.
"""

import argparse

from . import data, notebooks

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``minian`` parser (also used to render CLI docs)."""
    parser = argparse.ArgumentParser(
        prog="minian",
        description="MiniAn command line tools: fetch demo data and copy out notebooks.",
    )
    sub = parser.add_subparsers(title="subcommands", dest="group", required=True)
    data.add_subparser(sub)
    notebooks.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
