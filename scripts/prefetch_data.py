"""Download all published demo datasets into the pooch cache.

CI prefetch helper: a single job runs this once per OS so the test matrix can
restore the datasets from cache instead of each job downloading them from
Zenodo in parallel.

It is deliberately standalone (needs only ``pooch``, not the full minian stack)
so the prefetch job stays cheap: it loads the dataset registry directly and
mirrors :func:`minian.data.fetch`'s download into the same cache location
(``pooch.os_cache("minian")``).
"""

import importlib.util
from pathlib import Path

import pooch

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "minian" / "data" / "_registry.py"


def _load_registry():
    spec = importlib.util.spec_from_file_location("_minian_registry", _REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    reg = _load_registry()
    registry, urls = {}, {}
    for name, meta in reg.DATASETS.items():
        record = meta.get("zenodo_record")
        if record is None:
            print(f"skip {name}: no zenodo_record yet")
            continue
        for relpath, info in meta["files"].items():
            key = f"{name}/{relpath}"
            registry[key] = f"sha256:{info['sha256']}"
            urls[key] = reg.zenodo_url(record, info["zenodo"])

    if not registry:
        print("nothing to prefetch (no published datasets)")
        return

    pup = pooch.create(
        path=pooch.os_cache("minian"), base_url="", registry=registry, urls=urls
    )
    for key in registry:
        pup.fetch(key, progressbar=False)
        print(f"fetched {key}")


if __name__ == "__main__":
    main()
