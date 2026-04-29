[![Python version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/DeniseCaiLab/minian/blob/main/pyproject.toml)
[![uv](https://img.shields.io/badge/uv-astral-purple.svg)](https://docs.astral.sh/uv/)

[![Build](https://github.com/DeniseCaiLab/minian/actions/workflows/build.yml/badge.svg)](https://github.com/DeniseCaiLab/minian/actions/workflows/build.yml)
[![Tests](https://github.com/DeniseCaiLab/minian/actions/workflows/testandcov.yml/badge.svg)](https://github.com/DeniseCaiLab/minian/actions/workflows/testandcov.yml)
[![Codecov](https://codecov.io/gh/DeniseCaiLab/minian/graph/badge.svg)](https://codecov.io/gh/DeniseCaiLab/minian)
[![Documentation](https://readthedocs.org/projects/minian/badge/?version=latest)](https://minian.readthedocs.io/en/latest/)

[![License](https://img.shields.io/github/license/DeniseCaiLab/minian)](https://www.gnu.org/licenses/gpl-3.0)

# MiniAn

MiniAn is an analysis pipeline and visualization tool inspired by both [CaImAn](https://github.com/flatironinstitute/CaImAn) and [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Prerequisites

- [uv](https://docs.astral.sh/uv/)

# Quick Start Guide

1. Create/sync environment: `uv sync`
1. Install pipeline notebooks (optional): `uv run minian-install --notebooks`
1. Install demo movies (optional): `uv run minian-install --demo`
1. You can set download location with `--dest`, for example:
   - `uv run minian-install --notebooks --dest ./artifacts`
   - `uv run minian-install --demo --dest ./artifacts`
1. Run notebook flow (current default): `uv run jupyter notebook` then open `pipeline.ipynb`

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

1. Data I/O + utilities from `minian/utilities.py`.
1. Preprocessing from `minian/preprocessing.py`.
1. Motion correction from `minian/motion_correction.py`.
1. Seed/initial component setup from `minian/initialization.py`.
1. CNMF iterations and component updates in `minian/cnmf.py`.
1. Cross-session registration in `minian/cross_registration.py` (optional stage).
1. Visualization/UI in the `minian/visualization/` package (HoloViews, Panel, Datashader).

Notebook/asset bootstrap is handled by `minian/install.py` (`minian-install` CLI).

# Cleanup Order (Recommended)

For the Python 3.12+ modernization, work in this order:

1. Stabilize packaging/runtime baseline (`pyproject.toml`, `uv.lock`, CI build).
1. Split and harden dependencies (core vs `viz` vs `docs`; keep optional features optional).
1. Keep the HoloViews/Bokeh/Panel stack aligned on supported versions (declared in `pyproject.toml`).
1. Make notebook workflow optional and promote CLI-first pipeline execution.
1. Refactor module internals (`preprocessing` -> `motion_correction` -> `initialization` -> `cnmf`) with tests after each step.
1. Apply targeted performance work (Rust candidates only after profiling confirms bottlenecks).
1. Final cleanup pass: dead code removal, docs refresh, migration notes.

# Documentation

MiniAn documentation is hosted on ReadtheDocs at:

https://minian.readthedocs.io/

# Contributing to MiniAn

We would love feedback and contribution from the community!
See [the contribution page](https://minian.readthedocs.io/en/latest/start_guide/contribute.html) for more detail!

# License

This project is licensed under GNU GPLv3.
