"""``minian notebooks`` subcommands: copy bundled notebooks out of the package."""

from pathlib import Path

from ..notebooks import copy, notebooks
from ._common import add_select_args, print_table, selected

DEFAULT_DEST = "minian-notebooks"


def _cmd_list(args):
    print_table(list(notebooks().items()))


def _cmd_copy(args):
    dest = Path(args.output) if args.output else Path(DEFAULT_DEST)
    for name in selected(args, notebooks(), "notebook"):
        for path in copy(name, dest):
            print(f"copied {name} -> {path}")


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "notebooks", help="copy bundled notebooks out of the package"
    )
    sub = parser.add_subparsers(title="subcommands", dest="notebooks_command", required=True)

    sub.add_parser("list", help="list available notebooks").set_defaults(func=_cmd_list)

    cp = sub.add_parser("copy", help="copy a notebook into a directory")
    add_select_args(cp, "notebook")
    cp.add_argument(
        "-o", "--output", help=f"destination directory (default: ./{DEFAULT_DEST})"
    )
    cp.set_defaults(func=_cmd_copy)
