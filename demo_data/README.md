# demo_data/ has moved

The demo cross-registration datasets are no longer stored in this repository.
They are now hosted on Zenodo and fetched on demand, cached, and
checksum-verified.

Get them with the CLI:

```bash
minian-data download cross-reg-sessions            # -> OS cache
minian-data download cross-reg-sessions --to .     # -> ./cross-reg-sessions/ here
```

Or from Python / a notebook:

```python
from minian.data import fetch
dpath = fetch("cross-reg-sessions")   # session1/minian.nc, session2/minian.nc
```

The `cross-reg-sessions` dataset is two saved single-session MiniAn datasets
(NetCDF) used by `cross-registration.ipynb` to demonstrate matching cells
across sessions.

See `minian/data/_registry.py` for checksums and provenance.
