"""``minian-data`` command line interface.

    minian-data list                     # available datasets + sizes
    minian-data download pipeline-demo    # -> cache, returns nothing
    minian-data download pipeline-demo --to ./demo_movies/
    minian-data path pipeline-demo        # print local path (fetching if needed)
"""

import argparse
import shutil
from pathlib import Path

from . import datasets, dataset_path, fetch
from ._registry import DATASETS


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def _dataset_size(name: str) -> int:
    return sum(info["size"] for info in DATASETS[name]["files"].values())


def _cmd_list(args):
    width = max(len(n) for n in DATASETS)
    for name, desc in datasets().items():
        size = _human_size(_dataset_size(name))
        print(f"{name:<{width}}  {size:>8}  {desc}")


def _cmd_download(args):
    src = fetch(args.name)
    if args.to:
        dest = Path(args.to)
        dest.mkdir(parents=True, exist_ok=True)
        for relpath in DATASETS[args.name]["files"]:
            target = dest / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / relpath, target)
        print(f"Copied {args.name} to {dest}")
    else:
        print(f"{args.name} ready at {src}")


def _cmd_path(args):
    print(dataset_path(args.name))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="minian-data", description="Fetch MiniAn demo datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available datasets").set_defaults(func=_cmd_list)

    p_dl = sub.add_parser("download", help="download (and cache) a dataset")
    p_dl.add_argument("name", help="dataset name")
    p_dl.add_argument("--to", help="also copy the files into this directory")
    p_dl.set_defaults(func=_cmd_download)

    p_path = sub.add_parser("path", help="print the local path of a dataset")
    p_path.add_argument("name", help="dataset name")
    p_path.set_defaults(func=_cmd_path)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
