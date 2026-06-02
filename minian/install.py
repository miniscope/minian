import argparse
import os
import requests
from importlib.metadata import version

NOTEBOOK_FILES = [
    "pipeline.ipynb",
    "cross-registration.ipynb",
    "img/workflow.png",
    "img/param_pnr.png",
    "img/param_spatial_update.png",
    "img/param_temporal_update.png",
]
DEMO_FILES = [f"demo_movies/msCam{i}.avi" for i in range(1, 11)] + [
    f"demo_data/session{i}/minian.nc" for i in range(1, 3)
]
try:
    VERSION = version("minian")
except:
    VERSION = "0.0.0"


def _get_file(filename: str, version: str):
    if os.path.isfile(f"{filename}"):
        print(f"File {filename} already exists, skipping install of this file.")
        return
    for vv in [version, "v" + version]:
        r = requests.get(f"https://raw.github.com/DeniseCaiLab/minian/{vv}/{filename}")
        if r.status_code == 200:
            parent_dir = os.path.dirname(filename)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(f"{filename}", "wb") as f:
                for chunk in r.iter_content(2048):
                    f.write(chunk)
            print(f"File {filename} installed.")
            break
    else:
        print(f"File {filename} not found with version {version}, skipping.")


def demo(version: str):
    print("Installing demo data")
    for file in DEMO_FILES:
        _get_file(file, version)


def notebook(version: str):
    print("Installing notebooks")
    for file in NOTEBOOK_FILES:
        _get_file(file, version)


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
        help=f"Git repo branch or tag name, default {VERSION}",
    )
    args = parser.parse_args()

    version = args.v
    print(f"Using version: {version}")

    if args.notebooks:
        notebook(version)

    if args.demo:
        demo(version)
