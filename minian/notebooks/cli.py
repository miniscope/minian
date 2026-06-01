"""``minian-notebooks`` command line interface.

Notebooks ship inside the installed package; this copies a bundle (a
self-contained folder of notebook + assets + README) out into a working
directory so you can run and edit it::

    minian-notebooks list                       # bundles + descriptions
    minian-notebooks copy pipeline               # -> ./minian-notebooks/pipeline/
    minian-notebooks copy cross_registration --to ~/work
    minian-notebooks copy --all
    minian-notebooks copy training/02_cnmf       # a single notebook within a bundle
"""

import argparse
import importlib.resources as ir
from pathlib import Path

PACKAGE = "minian.notebooks"
DEFAULT_DEST = "minian-notebooks"
_SKIP = {"__pycache__", ".ipynb_checkpoints"}


def _root():
    return ir.files(PACKAGE)


def _iter_notebooks(node, prefix=""):
    """Yield notebook paths (relative to the notebooks package root)."""
    for entry in node.iterdir():
        if entry.name in _SKIP:
            continue
        rel = f"{prefix}{entry.name}"
        if entry.is_dir():
            yield from _iter_notebooks(entry, prefix=rel + "/")
        elif entry.name.endswith(".ipynb"):
            yield rel


def _bundles():
    """Map bundle name (notebook-containing dir) -> sorted notebook relpaths."""
    bundles: dict[str, list[str]] = {}
    for rel in _iter_notebooks(_root()):
        bundle = rel.rsplit("/", 1)[0] if "/" in rel else ""
        bundles.setdefault(bundle, []).append(rel)
    return {k: sorted(v) for k, v in sorted(bundles.items())}


def _traverse(relpath):
    node = _root()
    for part in relpath.split("/"):
        node = node.joinpath(part)
    return node


def _readme_summary(bundle):
    try:
        text = _traverse(f"{bundle}/README.md").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return ""


def _copy_tree(src, dest: Path):
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            if child.name in _SKIP:
                continue
            _copy_tree(child, dest / child.name)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())


def _cmd_list(args):
    bundles = _bundles()
    if not bundles:
        print("No notebook bundles found.")
        return
    width = max(len(b) for b in bundles)
    for bundle, notebooks in bundles.items():
        summary = _readme_summary(bundle)
        print(f"{bundle:<{width}}  {summary}")
        if len(notebooks) > 1:
            for nb in notebooks:
                print(f"{'':<{width}}    - {nb}")


def _resolve(spec):
    """Resolve a copy spec to (source_node, destination_relpath)."""
    bundles = _bundles()
    if spec in bundles:  # whole bundle
        return _traverse(spec), Path(spec).name
    # single notebook within a bundle
    candidates = [spec, spec + ".ipynb"] if not spec.endswith(".ipynb") else [spec]
    for rel in candidates:
        node = _traverse(rel)
        if node.is_file():
            return node, Path(rel).name
    raise SystemExit(
        f"Unknown bundle or notebook: {spec!r}. Run `minian-notebooks list`."
    )


def _cmd_copy(args):
    dest_root = Path(args.to) if args.to else Path(DEFAULT_DEST)
    if args.all:
        specs = list(_bundles())
    elif args.target:
        specs = [args.target]
    else:
        raise SystemExit("Specify a bundle/notebook to copy, or --all.")
    for spec in specs:
        src, dest_name = _resolve(spec)
        dest = dest_root / dest_name
        _copy_tree(src, dest)
        print(f"Copied {spec} -> {dest}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="minian-notebooks",
        description="Copy MiniAn notebooks out of the installed package.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available notebook bundles").set_defaults(
        func=_cmd_list
    )

    p_copy = sub.add_parser("copy", help="copy a bundle or notebook to a directory")
    p_copy.add_argument("target", nargs="?", help="bundle name or bundle/notebook")
    p_copy.add_argument("--all", action="store_true", help="copy every bundle")
    p_copy.add_argument(
        "--to", help=f"destination directory (default: ./{DEFAULT_DEST})"
    )
    p_copy.set_defaults(func=_cmd_copy)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
