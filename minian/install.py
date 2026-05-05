import argparse
import logging
import os

import requests

from ._version import get_package_version
from .utilities import configure_logging

log = logging.getLogger(__name__)


NOTEBOOK_FILES = [
    "notebooks/pipeline.ipynb",
    "notebooks/cross-registration.ipynb",
    "img/workflow.png",
    "img/param_pnr.png",
    "img/param_spatial_update.png",
    "img/param_temporal_update.png",
]
DEMO_FILES = [f"demo_movies/msCam{i}.avi" for i in range(1, 11)] + [
    f"demo_data/session{i}/minian.nc" for i in range(1, 3)
]
VERSION = get_package_version()


def _get_file(filename: str, version: str, dest: str):
    local_path = os.path.join(dest, filename)
    if os.path.isfile(local_path):
        log.info("File %s already exists, skipping install of this file.", local_path)
        return
    for vv in [version, "v" + version]:
        r = requests.get(f"https://raw.github.com/DeniseCaiLab/minian/{vv}/{filename}")
        if r.status_code == 200:
            parent_dir = os.path.dirname(local_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(2048):
                    f.write(chunk)
            log.info("File %s installed.", local_path)
            break
    else:
        log.warning("File %s not found with version %s, skipping.", filename, version)


def demo(version: str, dest: str):
    log.info("Installing demo data")
    for file in DEMO_FILES:
        _get_file(file, version, dest)


def notebook(version: str, dest: str):
    log.info("Installing notebooks")
    for file in NOTEBOOK_FILES:
        _get_file(file, version, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebooks", action="store_true", help="Installs the notebooks"
    )
    parser.add_argument("--demo", action="store_true", help="Installs the demo data")
    parser.add_argument(
        "-v",
        action="store",
        default=VERSION,
        help="Git repo branch or tag name, default {}".format(VERSION),
    )
    parser.add_argument(
        "--dest",
        "-d",
        default=None,
        metavar="DIR",
        help="Directory to download into (default: current working directory)",
    )
    args = parser.parse_args()
    configure_logging()

    version = args.v
    dest = os.path.abspath(args.dest) if args.dest else os.getcwd()
    os.makedirs(dest, exist_ok=True)
    log.info("Using version: %s", version)
    log.info("Download directory: %s", dest)

    if args.notebooks:
        notebook(version, dest)

    if args.demo:
        demo(version, dest)
