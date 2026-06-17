"""Registry of externally-hosted MiniAn demo datasets.

Large binary demo assets are **not** stored in the git repository (they used to
bloat every clone by ~700 MB). They are hosted on Zenodo, which gives them a
citable DOI, and fetched on demand by :mod:`minian.data`: downloaded once,
cached to the OS cache dir, and verified against the SHA256 checksums recorded
here on every access.

Each dataset is its own Zenodo deposit (its own DOI), so a paper can cite
exactly the data it used. To publish or update a dataset:

1. Upload its files to a Zenodo deposit, titled ``MiniAn demo data: <dataset>``,
   and publish it.
2. Set that dataset's ``zenodo_record`` below to the published record id (the
   number in the record URL, e.g. ``https://zenodo.org/records/1234567`` ->
   ``"1234567"``).
3. Make sure each file's ``zenodo`` name below matches the filename as it
   appears in the record, and that ``sha256``/``size`` match the uploaded
   bytes. ``scripts/zenodo_manifest.py`` stages files and cross-checks these.

Every dataset must carry a published ``zenodo_record``; a missing one is a bug
here, not a runtime state, and :func:`minian.data.fetch` surfaces it as a
``KeyError`` rather than silently skipping the dataset.

The in-dataset path (the dict key under ``files``) is where the file lands
relative to the dataset directory returned by :func:`minian.data.fetch`; it may
contain subdirectories (e.g. ``session1/minian.nc``). The ``zenodo`` value is
the (necessarily flat, unique-per-deposit) filename on Zenodo.
"""

from typing import TypedDict


class ZenodoFile(TypedDict):
    zenodo: str
    """filename"""
    size: int
    sha256: str
    """hex-digested sha256 hash"""


class ZenodoDataset(TypedDict):
    title: str
    description: str
    zenodo_record: int
    """The integer identifier like /records/{record}/... in the URL"""
    files: dict[str, ZenodoFile]


def zenodo_url(record: int, filename: str) -> str:
    """Direct-download URL for *filename* within a published Zenodo *record*."""
    return f"https://zenodo.org/records/{record}/files/{filename}?download=1"


DATASETS: dict[str, ZenodoDataset] = {
    "pipeline-demo": {
        "title": "Minian demo data: Miniscope V3 Hippocampal CA1",
        "description": (
            "Mouse CA1 recording on a Miniscope V3, 5x temporally downsampled. "
            "10x msCam .avi, 2000 frames @ 480x752 px (~688 MB). Drives "
            "pipeline.ipynb and the pipeline test."
        ),
        "zenodo_record": 20484805,
        "files": {
            "msCam1.avi": {
                "zenodo": "msCam1.avi",
                "size": 72203510,
                "sha256": "01cd69e2ab815ae52a10fbe561e2e5e3ee8376d549ee960f9317c60a4d2ff8d6",
            },
            "msCam2.avi": {
                "zenodo": "msCam2.avi",
                "size": 72203510,
                "sha256": "d14a272a65fd9d6962583f2f82daa2544ec4d767f1212eb0172e12d590f273e0",
            },
            "msCam3.avi": {
                "zenodo": "msCam3.avi",
                "size": 72203510,
                "sha256": "3be0c0f1c82f58b9bc3a74042e32b5fc7450ad95b3d4b82f2f99384fa8787a2c",
            },
            "msCam4.avi": {
                "zenodo": "msCam4.avi",
                "size": 72203510,
                "sha256": "ef47ade62c56533af8d3dbee031a0a69cc09d163672520eac304600bfc8b50be",
            },
            "msCam5.avi": {
                "zenodo": "msCam5.avi",
                "size": 72203510,
                "sha256": "f2408fe116be5e96883d4dc23e30606f402387c0a61dcfd20f4726edb159bbc6",
            },
            "msCam6.avi": {
                "zenodo": "msCam6.avi",
                "size": 72203510,
                "sha256": "29061c11b9f411a85f822f5341094d67d68938d7bdad063443f653578e2aad62",
            },
            "msCam7.avi": {
                "zenodo": "msCam7.avi",
                "size": 72203510,
                "sha256": "2d55efe15e755bdae6c1b18d6021498f85dfee5c839f519715191a64c676f1da",
            },
            "msCam8.avi": {
                "zenodo": "msCam8.avi",
                "size": 72203510,
                "sha256": "08b639392a14fa7d59642674097a2824481e8c25ebb8677a515aa2647ce933ee",
            },
            "msCam9.avi": {
                "zenodo": "msCam9.avi",
                "size": 72203510,
                "sha256": "f4b5c7e109dc21fbc05beec49e2263452ef9538a751570eb2f5c51cd77e6e341",
            },
            "msCam10.avi": {
                "zenodo": "msCam10.avi",
                "size": 72203510,
                "sha256": "b1f66de12d8726ab7bcf30a78c3718fe3ab5d2179dce8bcfc2c3c19dabbeab39",
            },
        },
    },
    "cross-reg-sessions": {
        "title": "Minian demo data: cross-registration sessions",
        "description": (
            "Two saved single-session minian datasets (NetCDF) for "
            "cross-registration.ipynb and the cross-reg test (~10 MB)."
        ),
        "zenodo_record": 20497476,
        "files": {
            "session1/minian.nc": {
                "zenodo": "session1_minian.nc",
                "size": 5096610,
                "sha256": "32ae54b20a4a5f6b855d182d1c26eb438f2919af76fa8d9ee2831d7815ef3746",
            },
            "session2/minian.nc": {
                "zenodo": "session2_minian.nc",
                "size": 5470609,
                "sha256": "b7dab60bb01e579481e6f5ede618b914c55e1edb36f009971895444ac78c1fd3",
            },
        },
    },
}
