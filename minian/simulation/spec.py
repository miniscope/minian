"""Typed, serializable specification for the ``minian.simulation`` pipeline.

This module defines the *contract* the simulator consumes: a tree of pydantic
v2 models describing the acquisition (a real, physical interface), an ordered
list of pipeline steps, and output formatting. It is the inverse of the minian
analysis pipeline expressed as data — the same ``Spec`` object a training
notebook walks through, a test parametrizes over, and a cache keys on.

Two layers, by design (see ``proposals/simulation-spec.md`` §2):

* **Layer 1 — what you read off a datasheet.** ``Optics.na``,
  ``Optics.magnification``, ``Tissue.scatter_mfp_um`` and friends. Every knob a
  user touches here is a real, measurable property of a real scope or sample.
* **Layer 2 — what a step consumes.** Pixel size, PSF sigma, attenuation, noise
  variance — *derived* from Layer 1 by small, documented, individually-testable
  helpers.

This file defines the full Layer-1 surface, the unit conversions, the static
``AnyStep`` union (migration Step 2), and the Layer-2 physics helpers —
``Optics.diffraction_sigma_um``/``defocus_sigma_um``, ``Tissue.attenuation``/
``scatter_sigma_um``, the combined ``Acquisition.cell_optics``, and the
``ImageSensor.photons_to_counts`` sensor model (Step 3). The executable steps
that ``build()`` returns land in Step 5; until then ``StepSpec.build()`` is
intentionally unimplemented — these classes are the schema plus its physics,
not the execution engine.

Units convention: **everything physical is in seconds and µm/mm — never frames
or pixels.** ``Acquisition`` owns every conversion to pixels/frames.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from minian.simulation.steps.base import Step


class SpecWarning(UserWarning):
    """Advisory warning for unusual-but-legal spec configurations.

    The simulator distinguishes *invalid* configs (which raise) from *unusual*
    ones (which warn but still run) — e.g. a focal plane outside the cell depth
    range, or steps listed out of the natural physical order.
    """


class _Base(BaseModel):
    """Common config for every spec model.

    ``extra="forbid"`` turns a mistyped field name into a construction-time
    error instead of a silently-ignored value. ``frozen=True`` makes specs
    immutable, which is what lets a recording be identified by its spec: the
    cache keys off the canonical JSON form (see ``Spec.cache_key``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Physical interface — Acquisition / Optics / Tissue (Layer 1 + unit conversions)
# ---------------------------------------------------------------------------


class Optics(_Base):
    """Objective optics — the measurable lens properties of a 1-photon scope.

    Layer-2 phenomenological quantities (diffraction sigma, defocus blur) are
    *derived* from these fields; their math arrives in migration Step 3. Pixel
    size is a joint optics×sensor quantity (sensor pitch / magnification) and so
    lives on ``Acquisition``, not here.

    Typical 1-photon miniscope ranges: NA 0.3–0.6, magnification ~5–10×, GCaMP
    emission ~510–540 nm.
    """

    na: float = Field(gt=0, default=0.45, description="Numerical aperture of the GRIN objective.")
    magnification: float = Field(gt=0, default=8.0, description="Optical magnification (sensor side / object side).")
    emission_nm: float = Field(gt=0, default=525.0, description="Fluorophore emission wavelength, nm (GCaMP ≈ 525).")
    focal_plane_um: float | Literal["auto"] = Field(
        default="auto",
        description="Depth of the in-focus plane, µm into tissue; 'auto' resolves to the "
        "median realized cell depth when the optics step runs.",
    )
    depth_of_field_um: float = Field(
        gt=0,
        default=15.0,
        description="±range around the focal plane treated as 'in focus' for detectability.",
    )

    # ---- Layer-2 helpers: small, documented, individually-testable approximations ----

    @property
    def diffraction_sigma_um(self) -> float:
        """Diffraction-limited PSF width (Gaussian σ), µm.

        A Gaussian stand-in for the Airy disk: the diffraction FWHM is
        ``≈ 0.51·λ/NA`` and ``σ = FWHM / 2.355 ≈ 0.21·λ/NA``. Ignores
        aberrations and the finite Airy tails — adequate for showing how NA and
        emission wavelength set the resolution floor. Smaller NA ⇒ larger σ
        (blurrier). At NA 0.45, λ ≈ 525 nm this is σ ≈ 0.24 µm.
        """
        return 0.21 * (self.emission_nm / 1000.0) / self.na

    def defocus_sigma_um(self, z_um: float, focal_um: float) -> float:
        """Out-of-focus blur (Gaussian σ), µm, for a cell at depth ``z_um``.

        Geometric defocus broadens linearly with the distance from the focal
        plane: ``σ ≈ NA·|z − z_focal|``. Symmetric about the focal plane (zero
        at ``z == focal_um``) and larger for higher NA (shallower depth of
        field). INTENSITY-CONSERVING: defocus spreads light without losing it,
        so the matching peak drop is applied in
        :meth:`Acquisition.cell_optics`; that is what separates defocus
        (spreads) from scatter (attenuates). ``focal_um`` is passed explicitly
        because :attr:`focal_plane_um` may be ``"auto"``, resolved to a concrete
        depth by the optics step before this is called.
        """
        return self.na * abs(z_um - focal_um)


class ImageSensor(_Base):
    """Physical and noise properties of the bare image sensor (the detector).

    Named *image sensor*, not *camera*, on purpose: a camera bundles optics on
    top of a sensor, whereas this is only the photosensitive array and its
    readout chain. The optics live separately on ``Optics``. Together with the
    exposure scale on the ``sensor`` step, the fields here fully specify the
    photons→counts conversion.

    Typical CMOS miniscope sensors: ~2–6 µm pixel pitch, QE 0.6–0.9, read noise
    1–5 e⁻ RMS, 8–12-bit ADC.
    """

    n_px_height: int = Field(gt=0, default=256, description="Sensor height, pixels.")
    n_px_width: int = Field(gt=0, default=256, description="Sensor width, pixels.")
    pixel_pitch_um: float = Field(gt=0, default=3.0, description="Physical sensor pixel pitch, µm.")
    quantum_efficiency: float = Field(gt=0, le=1, default=0.7, description="Photon → electron conversion efficiency.")
    read_noise_e: float = Field(ge=0, default=2.0, description="Read noise, electrons RMS.")
    gain_adu_per_e: float = Field(gt=0, default=1.0, description="Camera gain, ADU per electron.")
    bit_depth: int = Field(gt=0, default=8, description="ADC bit depth; counts clipped to [0, 2^bit_depth − 1].")

    def photons_to_counts(self, photons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Forward sensor model: incident photons → digitized ADC counts.

        The only place fluorescence becomes integer counts (spec §6)::

            e⁻     = Poisson(photons · quantum_efficiency)   # shot noise
            e⁻    += Normal(0, read_noise_e)                 # read noise, electrons
            adu    = e⁻ · gain_adu_per_e
            counts = clip(floor(adu), 0, 2**bit_depth − 1)

        Shot noise is Poisson on the *detected* electrons: photon arrival is
        Poisson and detection thins it by QE, which stays Poisson. Read noise is
        additive Gaussian in electrons. Quantization is ``floor`` (an ADC
        truncates), and counts are clipped to the converter's representable
        range. ``photons`` is the per-pixel expected photon count — the
        ``Sensor`` step produces it from scene intensity × its
        ``photons_per_unit`` exposure scale. Returns a float array holding
        integer-valued counts (the float container is set by
        ``Output.store_dtype``).
        """
        photons = np.asarray(photons, dtype=float)
        electrons = rng.poisson(photons * self.quantum_efficiency).astype(float)
        electrons += rng.normal(0.0, self.read_noise_e, size=electrons.shape)
        counts = np.floor(electrons * self.gain_adu_per_e)
        return np.clip(counts, 0.0, 2**self.bit_depth - 1)


class Tissue(_Base):
    """Light-scattering properties of the imaged tissue, as a function of depth.

    Both fields parametrize Layer-2 helpers (``attenuation``, ``scatter_sigma``)
    whose closed forms arrive in migration Step 3. Typical mouse cortex emission
    mean-free-path is ~50–200 µm.
    """

    scatter_mfp_um: float = Field(
        gt=0,
        default=100.0,
        description="Mean free path for emission photons (Beer–Lambert length scale), µm.",
    )
    scatter_blur_per_um: float = Field(
        ge=0,
        default=0.02,
        description="Linear broadening of the footprint per µm of depth (µm sigma per µm depth).",
    )

    # ---- Layer-2 helpers: scattering as a function of absolute depth ----

    def attenuation(self, z_um: float) -> float:
        """Fraction of emission light surviving scatter from depth ``z_um`` — in (0, 1].

        Beer–Lambert exponential decay ``exp(−z / mfp)`` over the mean free path
        ``scatter_mfp_um``. Monotonically decreasing in depth and equal to 1 at
        the surface (``z = 0``). Genuinely *removes* light — unlike defocus — so
        a deep cell is irreversibly dimmer, which is exactly the irreducible
        limit the module teaches. Typical mouse-cortex emission MFP ≈ 50–200 µm.
        """
        return math.exp(-z_um / self.scatter_mfp_um)

    def scatter_sigma_um(self, z_um: float) -> float:
        """Scatter-induced footprint broadening (Gaussian σ), µm, at depth ``z_um``.

        Linear phenomenological model ``σ = scatter_blur_per_um · z``: deeper
        cells scatter more on the way out and so appear both larger and dimmer
        (see :meth:`attenuation`). Monotonically increasing in depth and zero at
        the surface. Unlike defocus this is not intensity-conserving — it
        co-occurs with attenuation.
        """
        return self.scatter_blur_per_um * z_um


class Acquisition(_Base):
    """The physical acquisition: optics, image sensor, tissue, and sampling.

    Owns *all* unit conversions between the physical world (µm, seconds) and the
    sampled world (pixels, frames). Pixel size is the joint optics×sensor
    quantity ``image_sensor.pixel_pitch_um / optics.magnification``; FOV is then
    derived from the sensor's pixel count — any two of {FOV, pixel size, pixel
    count} fix the third.
    """

    optics: Optics = Field(default_factory=Optics)
    image_sensor: ImageSensor = Field(default_factory=ImageSensor)
    tissue: Tissue = Field(default_factory=Tissue)
    fps: float = Field(gt=0, default=20.0, description="Frame rate, frames per second.")
    duration_s: float = Field(gt=0, default=150.0, description="Recording duration, seconds.")

    @property
    def pixel_size_um(self) -> float:
        """Object-space size of one pixel, µm (sensor pitch / magnification)."""
        return self.image_sensor.pixel_pitch_um / self.optics.magnification

    @property
    def n_frames(self) -> int:
        """Number of frames in the recording (duration × fps, rounded)."""
        return round(self.duration_s * self.fps)

    @property
    def fov_um(self) -> tuple[float, float]:
        """Field of view (height, width) in µm — derived from pixels × pixel size."""
        return (
            self.image_sensor.n_px_height * self.pixel_size_um,
            self.image_sensor.n_px_width * self.pixel_size_um,
        )

    def um_to_px(self, um: float) -> float:
        """Convert a physical distance (µm) to pixels."""
        return um / self.pixel_size_um

    def s_to_frame(self, s: float) -> float:
        """Convert a duration (seconds) to a (fractional) frame count."""
        return s * self.fps

    def cell_optics(self, z_um: float, focal_um: float) -> tuple[float, float]:
        """Combined per-cell optical degradation: ``(sigma_px, brightness)``.

        Folds the three Layer-2 effects into the two numbers the optics step
        applies to a footprint — a blur width (pixels) and a brightness scale.
        Blurs add in quadrature; brightness factors multiply (spec §2)::

            σ_0   = hypot(diffraction_sigma_um, scatter_sigma_um(z))   # all but defocus
            σ_tot = hypot(σ_0, defocus_sigma_um(z, focal))
            brightness = (σ_0² / σ_tot²) · attenuation(z)
            sigma_px   = σ_tot / pixel_size_um

        The ``σ_0²/σ_tot²`` factor is the peak drop that makes defocus
        intensity-conserving (a 2-D Gaussian's peak × area is constant), so only
        ``attenuation(z)`` actually removes light. Consequently
        ``sigma_px² · brightness`` is independent of the focal plane — the
        invariant the conservation test asserts. ``focal_um`` is the resolved
        (numeric) focal depth; ``diffraction_sigma_um > 0`` always, so ``σ_tot``
        is never zero.

        Two distinct quantities come out, consumed separately by the optics step
        (5b): ``sigma_px`` is the PSF width the footprint is *convolved* with,
        and the footprint's brightness then scales by ``attenuation(z)`` **alone**
        — the convolution itself produces the defocus peak drop, so applying
        ``brightness`` to the footprint too would double-count it. The returned
        ``brightness`` is instead the *point-source peak* scalar: how far a
        cell's peak signal sits above the noise, i.e. the per-cell effective
        brightness used (with the illumination field and sensor floor) to decide
        detectability at ``finalize()``.
        """
        sigma_0 = math.hypot(self.optics.diffraction_sigma_um, self.tissue.scatter_sigma_um(z_um))
        sigma_total = math.hypot(sigma_0, self.optics.defocus_sigma_um(z_um, focal_um))
        brightness = (sigma_0**2 / sigma_total**2) * self.tissue.attenuation(z_um)
        sigma_px = sigma_total / self.pixel_size_um
        return sigma_px, brightness


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class Output(_Base):
    """Final-array formatting — formatting only, never rescaling (honest radiometry)."""

    save_intermediates: bool = Field(
        default=False,
        description="Retain a snapshot after every step (test oracle + teaching visuals). "
        "Default False to keep memory flat for the programmatic and sweep paths; the "
        "two notebook presets opt in explicitly. When False, only `observed` + "
        "`ground_truth` are kept and stage() raises.",
    )
    store_dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="Float container for the integer-valued sensor counts.",
    )


# ---------------------------------------------------------------------------
# Step framework — StepSpec base + registry + the static AnyStep union
# ---------------------------------------------------------------------------


# Natural physical order of the pipeline. A step's domain is a class-level
# attribute (not a serialized field); the Spec validator warns if the list
# departs from this order.
_DOMAIN_RANK: dict[str, int] = {"cell": 0, "tissue": 1, "motion": 2, "sensor": 3}


class StepSpec(_Base):
    """Base class for a single pipeline step's configuration.

    A concrete step spec carries its physical parameters and a literal ``kind``
    discriminator, and declares its ``domain`` (a class attribute used for
    ordering checks). ``build()`` turns the spec into the executable step that
    mutates a ``Scene``; those bodies arrive in migration Step 5.
    """

    domain: ClassVar[Literal["cell", "tissue", "motion", "sensor"]]
    kind: str

    def build(self, acq: Acquisition, rng) -> Step:
        """Return the executable step (a callable that mutates a Scene).

        Unimplemented until migration Step 5 — at this stage these classes are
        the typed schema/contract, not the execution engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.build() is implemented in migration Step 5; "
            "Step 2 defines the spec surface only."
        )


# ---------------------------------------------------------------------------
# Step catalog (cell → tissue → motion → sensor). Fields define the v1 surface;
# `build()` bodies and the no-op placeholder steps (vasculature, fancier motion)
# arrive in Step 5.
# ---------------------------------------------------------------------------


class SNRDistribution(_Base):
    """Per-cell signal-to-noise sampling distribution."""

    distribution: Literal["uniform", "lognormal"] = "lognormal"
    low: float = Field(gt=0, default=1.5, description="Lower SNR bound / lognormal low anchor.")
    high: float = Field(gt=0, default=8.0, description="Upper SNR bound / lognormal high anchor.")

    @model_validator(mode="after")
    def _check_order(self) -> SNRDistribution:
        if self.high <= self.low:
            raise ValueError(f"SNR high ({self.high}) must exceed low ({self.low}).")
        return self


class PlaceSomata(StepSpec):
    """Place generic neuron somata (+ optional neurite stubs) in a 3-D µm volume.

    'Place' is the verb — this *positions cell bodies in space*; it is unrelated
    to hippocampal *place cells*. v1 models one generic excitable soma type (an
    irregular blob + optional proximal stubs): there is no cell-type distinction
    and no spatial/behavioral tuning. Footprints are 2-D masks carrying a scalar
    depth ``z``; out-of-focus somata that become background emerge for free
    downstream from ``z`` + ``optics``.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["place_somata"] = "place_somata"
    density_per_mm2: float = Field(gt=0, default=150.0, description="Cell areal density; count derived from FOV.")
    soma_radius_um: float = Field(gt=0, default=7.0, description="Soma radius, µm (typical cortical neuron ≈ 5–10).")
    irregularity: float = Field(
        ge=0,
        le=1,
        default=0.3,
        description="0 = smooth disk; higher = lumpier soma (low-pass-noise threshold).",
    )
    n_neurite_stubs: int = Field(
        ge=0,
        default=0,
        description="Short proximal dendrite lobes (offset, dimmer). Not implemented "
        "in v1 (soma body only) — must be 0 until the feature lands.",
    )
    depth_range_um: tuple[float, float] = Field(
        default=(0.0, 200.0), description="(min, max) depth into tissue, µm."
    )
    min_distance_um: float = Field(
        ge=0, default=0.0, description="3-D center-to-center minimum (Poisson-disk if > 0)."
    )
    snr: SNRDistribution = Field(default_factory=SNRDistribution)

    @field_validator("depth_range_um")
    @classmethod
    def _check_depth_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo < 0:
            raise ValueError(f"depth_range_um min ({lo}) must be ≥ 0.")
        if hi < lo:
            raise ValueError(f"depth_range_um max ({hi}) must be ≥ min ({lo}).")
        return v

    @field_validator("n_neurite_stubs")
    @classmethod
    def _stubs_unimplemented(cls, v: int) -> int:
        # Fail fast at construction rather than deferring a NotImplementedError to
        # the step body deep inside a run; the soma body is all v1 models.
        if v > 0:
            raise ValueError(
                f"n_neurite_stubs={v} is not implemented in v1 (soma body only); leave at 0."
            )
        return v

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.cell import PlaceSomataStep

        return PlaceSomataStep(self, acq, rng)


class CellActivity(StepSpec):
    """Calcium activity: 2-state Markov gate → Poisson spikes → double-exp kernel.

    Modeled on CaLab so synthetic traces are deconvolvable by it; we reimplement
    only the forward kernel ``k(t) = exp(-t/τ_d) − exp(-t/τ_r)``. Indicator
    saturation and per-cell τ jitter are deferred to v1.1.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["cell_activity"] = "cell_activity"
    p_quiescent_to_active: float = Field(gt=0, default=0.1, description="Per-frame quiescent→active transition prob.")
    p_active_to_quiescent: float = Field(gt=0, default=0.5, description="Per-frame active→quiescent transition prob.")
    active_rate_hz: float = Field(gt=0, default=3.0, description="Poisson firing rate while active, Hz.")
    quiescent_rate_hz: float = Field(ge=0, default=0.05, description="Poisson firing rate while quiescent, Hz.")
    tau_rise_s: float = Field(gt=0, default=0.05, description="Calcium rise time constant, s.")
    tau_decay_s: float = Field(gt=0, default=0.5, description="Calcium decay time constant, s.")
    spike_amp_cv: float = Field(ge=0, default=0.3, description="Lognormal per-spike amplitude variability.")
    f0: float = Field(ge=0, default=1.0, description="Baseline fluorescence.")
    trace_noise: float = Field(
        ge=0,
        default=0.0,
        description="Optional additive per-cell trace noise (independent of sensor noise).",
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.cell import CellActivityStep

        return CellActivityStep(self, acq, rng)


class CellOptics(StepSpec):
    """Per-cell diffraction + defocus(|z − z_f|) + scatter(z) blur & attenuation.

    No tunable fields: blur and attenuation are fully determined by each cell's
    ``z`` plus the physical ``Optics``/``Tissue`` constants on ``Acquisition``.
    Writes the observed (degraded) footprint alongside the planted (sharp) one,
    sets the geometric ``in_focus`` flag, and stores the per-cell
    ``optical_brightness`` peak scalar. ``detectable`` is *not* set here — it is
    a whole-pipeline flag (optics × illumination vs the sensor noise floor)
    assembled in ``finalize()`` (Step 6).
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["optics"] = "optics"

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.cell import CellOpticsStep

        return CellOpticsStep(self, acq, rng)


class Render(StepSpec):
    """Composite ``Σ_i degraded_footprint_i × trace_i`` into the movie.

    The built step's snapshot name is ``"cells_only"``. The planted (sharp)
    ``A``/``C`` remain the ideal CNMF target in ground truth.
    """

    domain: ClassVar[str] = "tissue"
    kind: Literal["render"] = "render"

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.tissue import RenderStep

        return RenderStep(self, acq, rng)


class Neuropil(StepSpec):
    """Additive diffuse background: smooth spatial field × slow OU temporal."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["neuropil"] = "neuropil"
    spatial_sigma_um: float = Field(gt=0, default=40.0, description="Spatial smoothness of the mesh, µm.")
    temporal_tau_s: float = Field(gt=0, default=10.0, description="OU temporal correlation time, s (slow).")
    amplitude: float = Field(gt=0, default=0.3, description="Background amplitude relative to cell signal.")
    n_components: int = Field(ge=1, default=3, description="Number of independent diffuse components.")

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.tissue import NeuropilStep

        return NeuropilStep(self, acq, rng)


class Vasculature(StepSpec):
    """Dark absorbing mask × (slow dilation + cardiac). Placeholder no-op for v1."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["vasculature"] = "vasculature"
    enabled: bool = Field(default=False, description="Placeholder; multiplicative absorption lands in v1.1.")

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.tissue import VasculatureStep

        return VasculatureStep(self, acq, rng)


class Bleaching(StepSpec):
    """Global temporal decay of fluorophores (not the additive sensor leakage)."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["bleaching"] = "bleaching"
    model: Literal["mono_exp", "bi_exp"] = Field(
        default="mono_exp",
        description="Decay curve family. Only 'mono_exp' is implemented in v1; 'bi_exp' "
        "is rejected at construction (final_fraction alone underdetermines it).",
    )
    final_fraction: float = Field(
        gt=0, le=1, default=0.65, description="Brightness at the last frame relative to the first."
    )

    @field_validator("model")
    @classmethod
    def _model_implemented(cls, v: str) -> str:
        # Fail fast at construction rather than mid-run: a single final_fraction
        # does not pin a two-component curve, so faking one would be dishonest.
        if v == "bi_exp":
            raise ValueError(
                "model='bi_exp' is not implemented in v1 (mono_exp only); "
                "a single final_fraction does not determine a bi-exponential curve."
            )
        return v

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.tissue import BleachingStep

        return BleachingStep(self, acq, rng)


class BrainMotion(StepSpec):
    """Rigid x,y translation of the whole tissue frame — the tissue→sensor boundary.

    OU/jump and axial focus-drift motion are deferred placeholders.
    """

    domain: ClassVar[str] = "motion"
    kind: Literal["brain_motion"] = "brain_motion"
    trajectory_um: list[tuple[float, float]] | None = Field(
        default=None, description="Explicit per-frame (dy, dx) in µm; else a bounded random walk."
    )
    walk_step_um: float = Field(ge=0, default=0.3, description="Random-walk step size, µm/frame.")
    max_shift_um: float = Field(gt=0, default=5.0, description="Bound on cumulative shift magnitude, µm.")


class Vignette(StepSpec):
    """Static radial illumination falloff (lumped excitation × collection)."""

    domain: ClassVar[str] = "sensor"
    kind: Literal["vignette"] = "vignette"
    falloff: float = Field(
        ge=0, le=1, default=0.6, description="Corner brightness relative to center (1 = none)."
    )
    exponent: float = Field(gt=0, default=2.0, description="Radial falloff exponent.")
    center_offset_um: tuple[float, float] = Field(
        default=(0.0, 0.0), description="(dy, dx) offset of the bright center from FOV center, µm."
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.sensor import VignetteStep

        return VignetteStep(self, acq, rng)


class Leakage(StepSpec):
    """Static additive baseline — what minian's 'glow removal' subtracts."""

    domain: ClassVar[str] = "sensor"
    kind: Literal["leakage"] = "leakage"
    profile: Literal["uniform", "gaussian"] = Field(default="gaussian", description="Spatial baseline shape.")
    level: float = Field(ge=0, default=0.1, description="Additive baseline level.")
    sigma_um: float | None = Field(
        default=None,
        description="Spatial sigma for the gaussian profile, µm; None defaults to a "
        "quarter of the smaller FOV dimension. Ignored by the uniform profile.",
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.sensor import LeakageStep

        return LeakageStep(self, acq, rng)


class Sensor(StepSpec):
    """Photons → e⁻ → Poisson shot + read noise → ×gain → quantize → clip.

    The only step that produces integer-valued counts. The sensor *hardware*
    (QE, read noise, gain, bit depth, pixel pitch) lives on
    ``Acquisition.image_sensor`` and is read from there. The single field below
    is the exposure/flux scale — a scene property, not sensor hardware — which is
    why it stays on the step rather than the image-sensor spec.
    """

    domain: ClassVar[str] = "sensor"
    kind: Literal["sensor"] = "sensor"
    photons_per_unit: float = Field(
        gt=0,
        default=100.0,
        description="Photons per fluorescence intensity unit (exposure/flux scale); sets the "
        "shot-noise regime. A scene/illumination property, not sensor hardware.",
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minian.simulation.steps.sensor import SensorStep

        return SensorStep(self, acq, rng)


# The v1 catalog is closed and known, so AnyStep is a hand-written static union:
# native pydantic, trivially debuggable, no import-order hazards. It is the
# single source of truth for the step catalog — adding a component means
# defining its StepSpec subclass and adding it here. Pydantic's Discriminator
# handles kind→class dispatch for deserialization, so no separate registry is
# needed; if later tooling wants an explicit map, derive it from this union.
AnyStep = Annotated[
    PlaceSomata
    | CellActivity
    | CellOptics
    | Render
    | Neuropil
    | Vasculature
    | Bleaching
    | BrainMotion
    | Vignette
    | Leakage
    | Sensor,
    Discriminator("kind"),
]


# ---------------------------------------------------------------------------
# Top-level Spec + cross-field validation (spec §11)
# ---------------------------------------------------------------------------


class Spec(_Base):
    """A complete, reproducible recording specification.

    ``acquisition`` + ``steps`` (the ordered pipe) + ``seed`` + ``output`` fully
    determine a recording. The cross-field validators below catch genuinely
    invalid configs (raise) and flag unusual-but-legal ones (``SpecWarning``).
    """

    acquisition: Acquisition = Field(default_factory=Acquisition)
    seed: int = Field(default=42, description="RNG seed for full reproducibility.")
    steps: list[AnyStep]
    output: Output = Field(default_factory=Output)

    def cache_key(self) -> str:
        """SHA256 (first 16 hex chars) of the canonical JSON form. Stable across runs."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    @model_validator(mode="after")
    def _validate(self) -> Spec:
        self._check_unique_kinds()  # hard fail; everything below assumes unique kinds
        by_kind = {s.kind: s for s in self.steps}
        self._warn_domain_order()
        self._check_footprint_vs_fov(by_kind)
        self._check_sampling_vs_kinetics(by_kind)
        self._warn_focal_plane(by_kind)
        self._warn_motion_magnitude(by_kind)
        return self

    # -- hard fails ---------------------------------------------------------

    def _check_unique_kinds(self) -> None:
        """Rule 4: each ``kind`` appears at most once — lets sweeps address a step
        by kind and keeps the snapshot dict (keyed by step name) collision-free."""
        dupes = sorted(k for k, n in Counter(s.kind for s in self.steps).items() if n > 1)
        if dupes:
            raise ValueError(f"Duplicate step kind(s) in spec: {dupes}. Each kind must be unique.")

    def _check_footprint_vs_fov(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 2: a soma larger than the entire FOV is a misconfiguration."""
        pc = by_kind.get("place_somata")
        if pc is None:
            return
        min_fov = min(self.acquisition.fov_um)
        if 2 * pc.soma_radius_um > min_fov:
            raise ValueError(
                f"soma_radius_um={pc.soma_radius_um} µm gives a soma diameter larger than "
                f"the FOV ({min_fov:.3g} µm). Reduce the soma or enlarge the FOV."
            )

    def _check_sampling_vs_kinetics(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 3: ``tau_decay_s · fps`` must be ≳ 1, else the decay is unresolvable."""
        act = by_kind.get("cell_activity")
        if act is None:
            return
        samples_per_decay = act.tau_decay_s * self.acquisition.fps
        if samples_per_decay < 1.0:
            raise ValueError(
                f"tau_decay_s={act.tau_decay_s} s at fps={self.acquisition.fps} gives "
                f"{samples_per_decay:.3g} samples per decay (< 1); the calcium decay is "
                "unresolvable. Raise fps or tau_decay_s."
            )

    # -- advisory warnings --------------------------------------------------

    def _warn_domain_order(self) -> None:
        """Rule 5: steps out of cell→tissue→motion→sensor order are legal but unusual."""
        ranks = [_DOMAIN_RANK[type(s).domain] for s in self.steps]
        if any(b < a for a, b in zip(ranks, ranks[1:])):
            order = " → ".join(f"{s.kind}({type(s).domain})" for s in self.steps)
            warnings.warn(
                f"Steps depart from the natural cell→tissue→motion→sensor order: {order}.",
                SpecWarning,
                stacklevel=2,
            )

    def _warn_focal_plane(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 6: a numeric focal plane outside the cell depth range is unusual."""
        focal = self.acquisition.optics.focal_plane_um
        pc = by_kind.get("place_somata")
        if focal == "auto" or pc is None:
            return
        lo, hi = pc.depth_range_um
        if not (lo <= focal <= hi):
            warnings.warn(
                f"focal_plane_um={focal} µm is outside the cell depth range ({lo}, {hi}) µm.",
                SpecWarning,
                stacklevel=2,
            )

    def _warn_motion_magnitude(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 7: shifts beyond ~5% of the FOV likely indicate a misconfig."""
        mot = by_kind.get("brain_motion")
        if mot is None:
            return
        min_fov = min(self.acquisition.fov_um)
        if mot.trajectory_um is not None:
            extent = max((max(abs(dy), abs(dx)) for dy, dx in mot.trajectory_um), default=0.0)
        else:
            extent = mot.max_shift_um
        if extent > 0.05 * min_fov:
            warnings.warn(
                f"Motion extent {extent:.3g} µm exceeds 5% of the FOV ({min_fov:.3g} µm); "
                "likely a misconfiguration.",
                SpecWarning,
                stacklevel=2,
            )
