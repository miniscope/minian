<p align="center">
  <img src="https://github.com/denisecailab/minian/assets/844306/358d9a24-bc0d-40d0-8747-4c43693ce270" alt="image">
</p>


[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/DeniseCaiLab/minian/blob/main/pyproject.toml)
[![uv](https://img.shields.io/badge/uv-astral-purple.svg)](https://docs.astral.sh/uv/)

[![Build](https://github.com/DeniseCaiLab/minian/actions/workflows/build.yml/badge.svg)](https://github.com/DeniseCaiLab/minian/actions/workflows/build.yml)
[![Tests](https://github.com/DeniseCaiLab/minian/actions/workflows/testandcov.yml/badge.svg)](https://github.com/DeniseCaiLab/minian/actions/workflows/testandcov.yml)
[![Codecov](https://codecov.io/gh/DeniseCaiLab/minian/graph/badge.svg)](https://codecov.io/gh/DeniseCaiLab/minian)
[![Documentation](https://readthedocs.org/projects/minian/badge/?version=latest)](https://minian.readthedocs.io/en/latest/)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/DeniseCaiLab/minian)](https://www.gnu.org/licenses/gpl-3.0)

# MiniAn

MiniAn is an analysis pipeline and visualization tool inspired by both [CaImAn](https://github.com/flatironinstitute/CaImAn) and [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/download.html).

# Quick Start Guide

1. Clone **this fork** and `cd` into it:

   ```bash
   git clone https://github.com/thicclatka/minian.git
   cd minian
   ```

1. Create/sync environment: `uv sync`
1. Install pipeline notebooks (optional): `uv run minian-install --notebooks`
1. Install demo movies (optional): `uv run minian-install --demo`
1. You can set download location with `--dest`, for example:
   - `uv run minian-install --notebooks --dest /path/to/folder`
   - `uv run minian-install --demo --dest /path/to/folder`
1. **Headless pipelines** (after `uv sync`; `-d` / `--data` defaults to the current directory for both CLIs—use e.g. `-d ./demo_movies` for CNMF on the demo AVIs or `-d ./demo_data` for cross-reg on the demo sessions):
   - CNMF pipeline: `uv run minian-pipeline --help` · `uv run minian-pipeline` or `uv run python -m minian.pipelines.cnmf_process`
   - Cross-registration: `uv run minian-cross-reg --help` · `uv run minian-cross-reg` or `uv run python -m minian.pipelines.cross_reg`
1. **Notebook flow**: `uv run jupyter notebook` then open `pipeline.ipynb` (or the path where you installed notebooks with `--dest`).

# Rust extension (`minian.minian_rs`)

The package optionally includes a **`maturin` + PyO3** native module built from `src-rust/` (crate `src-rust`, import name **`minian.minian_rs`**). It accelerates FFT-based filters (`filt_fft`, `filt_fft_vec`); if the extension is missing, **`minian.cnmf.filters`** uses the legacy PyFFTW/Python path automatically.

**Developers editing Rust:** sync deps then install the extension into your env:

```bash
uv sync
uv run maturin develop --manifest-path src-rust/Cargo.toml
```

If you use [mise](https://mise.jdx.dev/), the repo `.mise.toml` defines **`mise run rs-dev`** for the same step.

Release wheels are built via **`uv build`** (PEP 517 **`maturin`** backend); CI runs that on Ubuntu, macOS, and Windows with Rust **1.95.0** (see repo `rust-toolchain.toml`). Parity checks live in **`minian/test/test_minian_rs.py`**.

# Current Code Flow

MiniAn currently follows this high-level flow:

1. Data I/O + utilities from the `minian.utilities` package (e.g. `load_videos`, `save_minian`).
1. Preprocessing from `minian/preprocessing.py`.
1. Motion correction from `minian/motion_correction.py`.
1. Seed/initial component setup from `minian/initialization.py`.
1. CNMF iterations and component updates in `minian/cnmf.py`.
1. Cross-session registration in `minian/cross_registration.py` (optional stage); runnable drivers live under **`minian/pipelines/`** (`cnmf_process.py`, `cross_reg.py`).
1. Visualization/UI in the `minian/visualization/` package (HoloViews, Panel, Datashader).

Notebook/asset bootstrap is handled by `minian/install.py` (`minian-install` CLI).

# Documentation

MiniAn documentation is hosted on ReadtheDocs at:

https://minian.readthedocs.io/

# Contributing to MiniAn

We would love feedback and contribution from the community!
See [the contribution page](https://minian.readthedocs.io/en/latest/start_guide/contribute.html) for more detail!

# License

This project is licensed under GNU GPLv3.
