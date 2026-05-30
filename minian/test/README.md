# Tests

MiniAn uses [pytest](https://docs.pytest.org/).

## Prerequisites

- **Python 3.12** (see `requires-python` in the repo root `pyproject.toml`).
- Install the dev environment from the **repository root**:

  ```bash
  uv sync --group dev
  ```

  With [PDM](https://pdm-project.org/) instead: `pdm install --group dev` (or `pdm install --with test`), then `pdm run rs-dev` and `pdm run test`.

- **FFmpeg**
- **`minian.minian_rs`**: some tests call the Rust extension. After `uv sync`, build it once from the repo root, e.g. `uv run maturin develop --manifest-path src-rust/Cargo.toml`, or `mise run rs-dev` if you use the repo `.mise.toml`.
- **`demo_movies/`** at the repo root (optional): a few pipeline tests skip if this folder or `msCam*.avi` clips are missing.

## Running tests

From the **repository root** (so imports and paths match CI):

```bash
uv run pytest -v --color=yes --pyargs minian
```

With coverage (same idea as GitHub Actions):

```bash
uv run pytest -v --color=yes --cov=minian --cov-report=term --pyargs minian
```

Dask worker count for the pipeline comes from **`minian_config.json`** (`n_workers` or CPU-derived defaults via `worker_cpu_ratio` / `reserve_cores_for_os`), not from shell environment variables.

## Layout

| Module                   | Focus                                                        |
| ------------------------ | ------------------------------------------------------------ |
| `test_config.py`         | Configuration / CLI defaults                                 |
| `test_cross_reg.py`      | Cross-registration pipeline pieces                           |
| `test_minian_rs.py`      | Rust extension vs legacy references                          |
| `test_pipeline.py`       | CNMF pipeline helpers, Dask smoke, optional demo video loads |
| `test_pre_processing.py` | Preprocessing utilities                                      |
| `toy_data.py`            | Shared small fixtures / helpers                              |

## Refreshing fixtures / heavy outputs

For pipeline-focused tests, re-run the relevant notebooks or pipelines on demo data, then adjust expectations or artifacts under `minian/test/` as needed. Keep large binaries out of git unless the project already tracks them (e.g. under `demo_movies/`).
