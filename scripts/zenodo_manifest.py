"""Stage MiniAn demo data for upload to Zenodo and cross-check the registry.

This is a maintainer tool, run once before publishing a Zenodo deposit (and
again whenever the demo data changes). It:

1. Locates each demo file on disk (from the in-repo ``demo_movies/`` /
   ``demo_data/`` layout, before those binaries are removed from the repo).
2. Computes its SHA256 + size and compares against
   ``minian.data._registry.DATASETS`` (fails loudly on any mismatch, so the
   committed checksums can never silently drift from the bytes you upload).
3. Copies each file into a per-dataset staging directory under the flat,
   unique filename it must have on Zenodo (``session1/minian.nc`` ->
   ``session1_minian.nc``).
4. Writes a per-dataset ``MANIFEST.txt`` and prints the suggested deposit
   metadata (one Zenodo deposit, hence one DOI, per dataset).

Usage::

    python scripts/zenodo_manifest.py            # stage into ./zenodo_upload
    python scripts/zenodo_manifest.py --out DIR  # stage elsewhere

Then, for each dataset, create a new deposit at https://zenodo.org/uploads/new,
upload every file from that dataset's staging folder, publish, and set the
dataset's ``zenodo_record`` in ``minian/data/_registry.py`` to the published
record id.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pooch

from minian.data._registry import DATASETS

# Source bytes are read from a directory laid out by dataset name, the same
# layout the pooch cache uses: ``<source>/<dataset>/<in-dataset relpath>``.
REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = str(REPO / ".minian_data")

COMMON_METADATA = """\
  Authors:  MiniAn Developers
  License:  Creative Commons Attribution 4.0 International (CC-BY-4.0)
  Keywords: miniscope, calcium imaging, CA1, hippocampus, MiniAn, demo data
  Version:  v1
  Related:  https://github.com/denisecailab/minian"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="zenodo_upload", help="staging directory")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="dir laid out as <source>/<dataset>/<relpath> (default: ./.minian_data)",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    ok = True

    for dataset, meta in DATASETS.items():
        root = source / dataset
        ddir = out / dataset
        ddir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for relpath, info in meta["files"].items():
            src = root / relpath
            if not src.is_file():
                print(f"MISSING  {dataset}: {src}")
                ok = False
                continue
            actual_sha = pooch.file_hash(str(src), alg="sha256")
            actual_size = src.stat().st_size
            tag = "ok"
            if actual_sha != info["sha256"]:
                tag, ok = "SHA MISMATCH", False
            elif actual_size != info["size"]:
                tag, ok = "SIZE MISMATCH", False
            dest = ddir / info["zenodo"]
            shutil.copy2(src, dest)
            manifest.append((info["zenodo"], actual_size, actual_sha))
            print(f"{tag:>13}  {dataset}/{relpath}  ->  {dataset}/{dest.name}")

        with open(ddir / "MANIFEST.txt", "w", encoding="utf-8") as f:
            f.write(f"# Zenodo deposit: {meta['title']}\n")
            f.write("# filename\tsize_bytes\tsha256\n")
            for name, size, sha in manifest:
                f.write(f"{name}\t{size}\t{sha}\n")

        print(f"  -> staged {len(manifest)} files into {ddir}/")
        print("  Suggested deposit metadata:")
        print(f"    Title:    {meta['title']}")
        print(COMMON_METADATA)
        print(f"    Description: {meta['description']}\n")

    if not ok:
        print("ERROR: registry checksums/sizes do not match on-disk files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
