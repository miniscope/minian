# 01 — Anatomy of a 1-photon miniscope recording

**Build a recording forward from its physics — the inverse of the minian analysis pipeline.**

This is the first of two training notebooks. It **constructs** a synthetic
miniscope recording one physically-meaningful stage at a time, using
[`minian.simulation`](../../../simulation). Each stage follows the same rhythm —
*understand* the physics, *explore* it with sliders, then *commit* the values you
want and move on — so the recording grows in front of you. Because every stage is
built by hand, the *exact* ground truth is known at each step, which is precisely
what the analysis pipeline has to recover. Notebook 2 (`02_pipeline_vs_truth`,
planned) then runs real minian on a recording like this and scores it against that
truth.

> **Work in progress.** Stages built so far: **the scope** (optics + sensor),
> **placing neurons**, and **calcium activity**. The remaining stages — optics
> degradation, render (the first movie), background fields, motion, and the sensor
> — are being added next.

## Run it

No data download is needed — the recording is generated on the fly.

```bash
pip install minian            # plus: ipywidgets, matplotlib, mediapy
jupyter notebook 01_anatomy.ipynb
```

Run top to bottom; each stage uses the values committed above it. The **explore**
cells are interactive and need a live kernel (e.g. the Stage-1 scope sliders for
NA / magnification / pixel pitch); in a statically-rendered copy they show their
default state — run the notebook locally to drag them.

## Dependencies beyond core minian

`ipywidgets` (sliders), `matplotlib` (figures), and `mediapy` (inline movie
playback, used by the movie stages; it relies on `ffmpeg`). All are lightweight
and need no GPU.

## What you'll learn

- A recording is a physical chain, and the `Spec` (`acquisition` + an ordered
  list of `steps`) *is* that chain written down.
- How numerical aperture (light collection ∝ NA²), magnification, pixel pitch,
  depth, and tissue scattering shape what the sensor actually sees — and set an
  **irreducible limit** on what any analysis can recover.
- What each minian stage is the inverse of: motion correction ↔ `shifts`,
  background/glow removal ↔ neuropil/leakage, denoising ↔ the sensor model.
