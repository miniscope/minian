# Rust engine integration (incremental, no full rewrite)

Goal: add Rust where it clearly pays off (obvious leaf kernels: FFT, tight numerical loops), behind clear Python boundaries (`pyo3` / `maturin`), with a **Python fallback** until parity is validated.

## Principles

1. **Leaf kernels first** — prioritize obvious hot leaves (FFT, tight numerical loops); Dask scheduling, I/O, and memory often dominate either way.
2. **One kernel at a time** — replace the smallest hot leaf, keep `xarray`/Dask orchestration in Python.
3. **Stable ABI** — Rust functions take `numpy` buffers (or simple structs) and return `numpy`/scalars; avoid passing Python objects into Rust.
4. **Import-based dispatch** — try `import minian.minian_rs`; on `ImportError`, use the existing pure-Python path (no env var required for today’s FFT hooks).
5. **Fallback** — ship working pure-Python path for every release until Rust path is proven.

---

## Priority order (do in this sequence)

### Phase 0 — Baseline and tooling (prerequisite)

Proceed with **`maturin` + `pyo3`** and kernels that are obviously leaf-heavy (FFT, tight loops).

1. Decide build story: **`maturin` + `pyo3`**, wheels built in CI for **linux / macos / windows** (aligned with existing `build.yml` matrix).
2. **Local rebuild (editable install):** from repo root, `uv run maturin develop --manifest-path src-rust/Cargo.toml` (or `mise run rs-dev` if you use the repo `.mise.toml`).

**Layout today:** a **single** Rust package at `src-rust/` (crate `src-rust`, library name `minian_rs`, Py module `minian.minian_rs`). No separate “core” crate unless `cargo test` / linking needs drive a split later.

**Validation:** parity is checked in **Python** — see `minian/test/test_minian_rs.py` (Rust vs `minian.cnmf.legacy` PyFFTW reference). Numeric FFT paths must match **NumPy / PyFFTW amplitude** (e.g. `realfft` inverse scaling vs `numpy_fft.irfft` is handled in Rust — see `src-rust/src/filter.rs`).

**Exit criteria:** Rust extension builds everywhere we ship wheels; developers can rebuild into the `uv` env with the command above. Toolchain is pinned by **`rust-toolchain.toml`** (CI and local mise should match). User-facing developer notes are in **`README.md`** (“Rust extension”).

**Phase 0 checklist**

- [ ] `uv build` succeeds locally with Rust stable matching `rust-toolchain.toml`.
- [ ] CI **build** workflow passes on **ubuntu / macos / windows** and uploads **`dist/`** artifacts.
- [ ] `uv run pytest minian/test/test_minian_rs.py` passes after **`maturin develop`** (or **`mise run rs-dev`**).
- [ ] Fallback: run tests without rebuilding Rust — anything that skips or uses legacy should still behave (Rust tests skip if `minian_rs` absent).

---

### Phase 1 — Highest-value target: `minian/cnmf/` inner compute

**Why first:** CNMF is the algorithmic core; it already mixes Numba, sparse linear algebra, FFT, and delayed blocks—where micro-optimizations compound.

1. Pick the next **leaf** to port (work through these in rough order of impact):
   - `update_temporal` / `update_temporal_block` and helpers
   - `update_spatial` / `update_spatial_perpx` / `update_spatial_block` and helpers
   - FFT noise path: `get_noise_fft` / `noise_fft` (including aggregation around the FFT if that code is still pure Python and hot)
2. Extract that leaf into a **single function** with numpy in/out.
3. Implement Rust equivalent; prove equivalence with **pytest** against the legacy / NumPy reference (tolerance where float noise is expected), not only `cargo test` (PyO3 `extension-module` makes pure-Rust `cargo test` awkward without extra crate split).
4. Wire via Python wrapper in `minian/cnmf/`; keep **legacy** path as fallback.

**Exit criteria:** clear speedup on representative runs without changing API semantics.

---

### Phase 2 — Motion and registration (conditional)

**Only if** Phase 1 is no longer worth chasing.

1. `minian/motion_correction.py`: much work may already sit in OpenCV (C++)—Rust only makes sense for **custom Python loops** you actually want to offload.
2. `minian/cross_registration.py`: same rule—Rust for **custom loops** worth porting.

**Exit criteria:** the Rust path replaces real Python/numpy bottleneck code and simplifies or speeds the hot path.

---

### Phase 3 — Graph / partitioning / sparse plumbing (last resort)

Touches `networkx`, `pymetis`, `scipy.sparse`, etc.

- Pursue Rust **only after** higher-value kernels are done.
- Often better to optimize **data structures and batching** in Python before a Rust rewrite here.

---

### Phase 4 — Housekeeping modules (`minian.utilities`, I/O)

**Lowest priority** unless you hit an obvious tight loop.

- `minian.utilities` is largely orchestration (Dask, zarr, chunking)—Rust rarely wins first here.

---

## Rollout checklist (each kernel)

- [ ] Rust implementation; **pytest** against reference (`minian/test/` …)
- [ ] Integration / round-trip smoke from Python (`numpy`/`xarray`) where applicable
- [ ] CI builds wheel artifacts on all OS targets
- [ ] Document import / fallback behavior (`minian_rs` vs legacy)
- [ ] Roll forward only after parity checks pass

---

## Anti-patterns (avoid)

- Rewriting entire `minian/cnmf/` package or the Dask graph in Rust in one step
- Replacing OpenCV-heavy paths with Rust when OpenCV already does the work
- Blocking development on Rust for features that depend on unfinished Python modernization (3.12, deps, CLI)
