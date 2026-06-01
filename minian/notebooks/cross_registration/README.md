# Cross-registration notebook

`cross-registration.ipynb` aligns and matches cells across multiple recording
sessions of the same animal, producing a cell-to-cell mapping table.

## Running it

```bash
pip install minian
minian-notebooks copy cross_registration
cd minian-notebooks/cross_registration
jupyter notebook cross-registration.ipynb
```

The notebook fetches its two-session demo dataset automatically via
`minian.data.fetch` (cached + checksum-verified). To run it on your own data,
edit the `dpath` / `f_pattern` cell near the top: point `dpath` at a directory
of saved minian datasets and set `f_pattern` to match them (e.g. `r"minian$"`
for the default `zarr` output format).
