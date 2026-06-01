"""Back-compat shim for the historical ``minian-install`` command.

``minian-install`` used to download notebooks and demo data over HTTP from
GitHub at install time. That is no longer how MiniAn ships:

* Notebooks live **inside** the installed package; copy them out with the
  ``minian-notebooks`` command.
* Demo data is fetched on demand, cached, and checksum-verified by the
  ``minian-data`` command.

This command is kept as a thin, dependency-free alias for one or two releases
so existing instructions keep working; please switch to the new CLIs.
"""

import argparse
from importlib.metadata import version

try:
    VERSION = version("minian")
except Exception:
    VERSION = "0.0.0"

# Datasets pulled by the legacy ``--demo`` flag (formerly demo_movies/ + demo_data/).
_DEMO_DATASETS = ["pipeline-demo", "cross-reg-sessions"]


def notebook():
    from minian.notebooks.cli import main as nb_main

    print(
        "Note: `minian-install --notebooks` is deprecated. "
        "Use `minian-notebooks copy --all` (or `minian-notebooks list`)."
    )
    nb_main(["copy", "--all"])


def demo():
    from minian.data.cli import main as data_main

    print(
        "Note: `minian-install --demo` is deprecated. "
        "Use `minian-data download <name>` (see `minian-data list`)."
    )
    for name in _DEMO_DATASETS:
        data_main(["download", name])


def main():
    parser = argparse.ArgumentParser(
        description="Deprecated alias for `minian-notebooks` and `minian-data`."
    )
    parser.add_argument(
        "--notebooks", action="store_true", help="copy bundled notebooks (all bundles)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="download the demo datasets"
    )
    parser.add_argument(
        "-v",
        action="store",
        default=VERSION,
        help="ignored; kept for back-compat with the old URL-fetch behavior",
    )
    args = parser.parse_args()

    if not (args.notebooks or args.demo):
        parser.print_help()
        return
    if args.notebooks:
        notebook()
    if args.demo:
        demo()
