## v2.0.0 (2026-06-16)

### Highlight

- ship notebooks inside the package and fetch demo data on demand from Zenodo
- unify the command line tools into a single `minian` CLI with `notebooks` and `data` subcommands
- ~12x faster motion estimation via dask task de-duplication and a cheaper registration/warp kernel
- consolidate video I/O into `minian.io`
- rebuild the documentation toolchain (modern Sphinx, MyST, pydata-sphinx-theme) with ReadTheDocs previews

### Feat

- bundle the pipeline and cross-registration notebooks in the package, copied out with `minian notebooks copy`
- fetch demo datasets from Zenodo on first run, cached and checksum-verified, managed with `minian data`
- add a single `minian` CLI replacing `minian-data` and `minian-notebooks`

### Perf

- de-duplicate dask tasks and use a cheaper registration/warp kernel in motion estimation (~12x faster)

### Refactor

- move video I/O to `minian.io`; importing those names from `minian.utilities`/`minian.visualization` now emits a deprecation warning

### Breaking changes

- require Python >=3.10
- install from PyPI or conda-forge instead of cloning the repo; notebooks and demo data are no longer in the repository tree
- replace the `minian-data`/`minian-notebooks` commands with the single `minian` command
- remove `requirements/*.txt`; dependencies and extras live in `pyproject.toml`

### Docs

- modern Sphinx/MyST toolchain installed via a `doc` extra; ReadTheDocs PR previews and versioned builds

## v1.3.0 (2026-06-16)

### Highlight

- modernize the dependency stack and support Python >=3.10
- publish to PyPI via Trusted Publishing and register the `minian` project
- remove the `pymetis` dependency in favor of a k-d tree spatial partitioner

### Feat

- require FFmpeg/ffprobe before video I/O, with a clear error if missing

### Fix

- update `update_meta()`

### Build

- real linting and formatting with ruff
- dependency cooldowns and an automated weekly lockfile refresh
- present pip + conda-forge install and drop the stale `environment.yml`

## v1.2.1 (2022-02-10)

### Fix

- avoid syntax error in `update_spatial` returns

## v1.2.0 (2022-02-09)

### Feat

- use least square to produce proper scaling in temporal components and background terms

### Fix

- rescale with normalizing factor when using `normalize` parameter in spatial and temporal update
- fix unit id mismatch in spatial parameter exploration

## v1.1.0 (2021-09-10)

### Fix

- pin jinja2 version to avoid doc build fail
- use fft filter for peak-to-noise ratio computation
- avoid conversion in `xrconcat_recursive`

### Feat

- baseline fluorescence correction in temporal update with median filter

## v1.0.1 (2021-05-05)

### Fix

- fix various typo and improve instructions in notebook

## v1.0.0 (2021-05-03)

### Highlight

- use dask localcluster and throttling for all computations to reduce memory demands
- add dedicated documentation site
- add testing and continuous integration
- release on conda-forge

## v1.0.0rc1 (2021-04-30)

### Feat

- graph based resolving of mappings

### Fix

- fix pipeline when `subset` is used

## v1.0.0rc0 (2021-04-11)

Candidate for first public release.
