# demo_movies/ has moved

The demo miniscope movies are no longer stored in this repository (they added
~700 MB to every clone). They are now hosted on Zenodo and fetched on demand,
cached, and checksum-verified.

Get them with the CLI:

```bash
minian-data list                              # see datasets + sizes
minian-data download pipeline-demo            # -> OS cache
minian-data download pipeline-demo --to .     # -> ./pipeline-demo/ here
```

Or from Python / a notebook:

```python
from minian.data import fetch
dpath = fetch("pipeline-demo")   # pathlib.Path to a folder of msCam*.avi
```

The `pipeline-demo` dataset is a mouse CA1 recording acquired on a Miniscope V3,
5x temporally downsampled (10 `msCam` `.avi` files, 2000 frames at 480x752).

See `minian/data/_registry.py` for checksums and provenance.
