"""Back-compat shim for the historical ``minian-install`` command.

``minian-install`` used to download notebooks and demo data over HTTP from
GitHub at install time. That is no longer how MiniAn ships:

* Notebooks live **inside** the installed package; copy them out with
  ``minian notebooks copy``.
* Demo data is fetched on demand, cached, and checksum-verified by
  ``minian data download``.

This command is kept as a thin, deprecated alias for one or two releases so
existing instructions keep working; please switch to the ``minian`` CLI.
"""

import argparse
import sys
import warnings

from .cli import main as minian_main


def _deprecation(message: str) -> None:
    """Real DeprecationWarning (visible under ``-W`` / in test suites) + a note."""
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(f"Note: {message}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated alias for the `minian` CLI (`minian notebooks` / `minian data`)."
    )
    parser.add_argument(
        "--notebooks", action="store_true", help="copy bundled notebooks (all of them)"
    )
    parser.add_argument("--demo", action="store_true", help="download the demo datasets")
    parser.add_argument(
        "-v",
        action="store",
        default=None,
        help="ignored; kept for back-compat with the old URL-fetch behavior",
    )
    args = parser.parse_args()

    if not (args.notebooks or args.demo):
        parser.print_help()
        return
    if args.notebooks:
        _deprecation(
            "`minian-install --notebooks` is deprecated; use `minian notebooks copy --all`."
        )
        minian_main(["notebooks", "copy", "--all"])
    if args.demo:
        _deprecation("`minian-install --demo` is deprecated; use `minian data download --all`.")
        minian_main(["data", "download", "--all"])
