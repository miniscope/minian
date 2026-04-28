# Minian Modernization TODO

Goal: make Minian reliably work on Python 3.12+ and move the project to a modern packaging/runtime stack (`uv`, PyPI, optional Homebrew entrypoint), while reducing dependency risk and improving CLI-first usability.

## Phase 0 - Baseline and guardrails (do first)

- [ ] Define support target: Python `>=3.12` (and decide upper bound policy, e.g. `<3.14` until verified).
- [ ] Add CI matrix for Linux/macOS/Windows on 3.12+ for build + smoke test.
- [ ] Lock initial toolchain with `uv` (`uv.lock`) and document local dev bootstrap.
- [ ] Capture current known-good behavior with a short validation checklist (load data, run core pipeline, export outputs).

## Phase 1 - Packaging and distribution migration (PyPI + uv first)

- [ ] Finalize packaging metadata in `pyproject.toml` (name, classifiers, urls, license, optional extras).
- [ ] Move all install/build/test/docs commands to `uv` workflows (`uv sync`, `uv run`, `uv build`, `uv publish`).
- [ ] Remove conda-first paths from docs/CI and replace with `uv` install instructions.
- [ ] Define release pipeline for PyPI (tag -> build -> publish -> verify install).
- [ ] Add simple install verification in CI: `pip install minian` (from built wheel/sdist artifact).
- [ ] Evaluate Homebrew strategy:
  - [ ] Decide formula source (`homebrew-core` vs tap).
  - [ ] Package CLI as primary `brew` entrypoint.

## Phase 2 - Dependency rationalization (remove risky/problematic deps)

- [ ] Inventory all runtime and optional dependencies (direct + critical transitive).
- [ ] Identify blockers on Python 3.12+ and cross-platform compatibility.
- [ ] Replace/remove `holoviews` (or isolate behind optional extra if removal is staged).
- [ ] Split dependency groups into clear extras (`cli`, `viz`, `dev`, `docs`, etc.).
- [ ] Add import-time guards/friendly errors for optional features.
- [ ] Ensure minimal install path is lightweight and reliable.

## Phase 3 - Rust adoption (targeted, measurable, safe)

- [ ] Select 1-2 hotspot modules for Rust rewrite based on profiling (not guesswork).
- [ ] Define Python <-> Rust interface (`pyo3`/`maturin` expected path).
- [ ] Add optional Rust extension build path in CI and packaging.
- [ ] Preserve pure-Python fallback where practical during transition.
- [ ] Benchmark before/after and gate merge on measurable wins.

## Phase 4 - API and call-site adjustments

- [ ] Update internal call sites affected by dependency/API changes.
- [ ] Add compatibility shims where needed to keep user-facing behavior stable.
- [ ] Remove deprecated pathways after one migration window.
- [ ] Expand regression tests around changed code paths.

## Phase 5 - CLI-first pipeline (reduce notebook dependency)

- [ ] Design top-level CLI UX (subcommands, config file support, defaults).
- [ ] Add command to fetch notebook/examples:
  - [ ] `--dest <path>` support for explicit download location.
  - [ ] Interactive/default download behavior when `--dest` omitted.
- [ ] Implement end-to-end pipeline command(s) runnable from CLI without Jupyter.
- [ ] Support config-driven pipeline execution (`yaml`/`toml`) for reproducibility.
- [ ] Keep notebook as optional reference/demo rather than required runtime path.

## Phase 6 - Codebase cleanup and hardening

- [ ] Normalize formatting/linting/type checks under `uv` tooling.
- [ ] Remove dead code and stale compatibility branches.
- [ ] Improve logging and error messages for common failures.
- [ ] Add contributor docs for dev workflow, release flow, and architecture notes.

## Phase 7 - Release execution and follow-through

- [ ] Cut prerelease (`x.y.zb1`) for early adopter validation.
- [ ] Gather migration feedback from real users.
- [ ] Fix high-priority regressions and finalize stable release.
- [ ] Announce migration guide: conda -> `uv`/PyPI, notebook -> CLI workflow.

---

## Priority order (single-threaded execution)

1. Phase 0 (guardrails)
2. Phase 1 (packaging/distribution)
3. Phase 2 (dependency cleanup, especially `holoviews`)
4. Phase 5 (CLI-first pipeline and notebook downloader UX)
5. Phase 4 (call-site/API stabilization)
6. Phase 3 (Rust acceleration passes)
7. Phase 6 + 7 (cleanup, release hardening, rollout)

## Immediate next actions (this week)

- [ ] Confirm exact Python version policy (`3.12` only vs `3.12-3.13`).
- [ ] Finish minimal CI build/test baseline on all OSes.
- [ ] Draft `uv`-first install section for `README.md`.
- [ ] Open tracking issues for each phase with owners and acceptance criteria.
