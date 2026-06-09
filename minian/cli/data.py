"""``minian data`` subcommands: fetch demo datasets from Zenodo."""

from ..data import dataset_path, datasets, fetch, fetch_all
from ..data._registry import DATASETS
from ._common import human_size, print_table


def _dataset_size(name: str) -> int:
    return sum(info["size"] for info in DATASETS[name]["files"].values())


def _cmd_list(args):
    print_table(
        [(name, human_size(_dataset_size(name)), desc)
         for name, desc in datasets().items()]
    )


def _cmd_download(args):
    if args.all:
        for path in fetch_all():
            print(f"ready: {path}")
        return
    if not args.name:
        raise SystemExit("Give a dataset name or --all (see `minian data list`).")
    print(f"{args.name} ready at {fetch(args.name)}")


def _cmd_path(args):
    print(dataset_path(args.name))


def add_subparser(subparsers):
    parser = subparsers.add_parser("data", help="fetch demo datasets")
    sub = parser.add_subparsers(title="subcommands", dest="data_command", required=True)

    sub.add_parser("list", help="list datasets and sizes").set_defaults(func=_cmd_list)

    dl = sub.add_parser("download", help="download and cache a dataset")
    dl.add_argument("name", nargs="?", help="dataset name (see `list`)")
    dl.add_argument("--all", action="store_true", help="download every dataset")
    dl.set_defaults(func=_cmd_download)

    pa = sub.add_parser("path", help="print a dataset's local path (fetching if needed)")
    pa.add_argument("name", help="dataset name")
    pa.set_defaults(func=_cmd_path)
