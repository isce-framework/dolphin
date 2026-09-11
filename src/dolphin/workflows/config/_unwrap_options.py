from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from ._enums import UnwrapMethod

logger = logging.getLogger("dolphin")

__all__ = [
    "PreprocessOptions",
    "SnaphuOptions",
    "SpurtOptions",
    "TophuOptions",
    "UnwrapOptions",
    "WhirlwindOptions",
]


def to_tuple(v: int | tuple[int, int] | None) -> tuple[int, int]:
    """Convert the input value to a tuple of two integers.

    Parameters
    ----------
    v : Union[int, Tuple[int, int], None]
        The value to be converted.

    Returns
    -------
    Tuple[int, int]
        Tuple containing two integers.

    Examples
    --------
    >>> to_tuple(3)
    (3, 3)
    >>> to_tuple((2, 4))
    (2, 4)
    >>> to_tuple(None)
    (1, 1)

    """
    if v is None:
        return (1, 1)
    if isinstance(v, int):
        return (v, v)
    return v


class PreprocessOptions(BaseModel, extra="forbid"):
    alpha: float = Field(
        0.5,
        description="Adaptive phase (Goldstein) filter exponent parameter.",
        ge=0.0,
        le=1.0,
    )
    max_radius: int = Field(
        51,
        ge=0.0,
        description="(for interpolation) Maximum radius to find scatterers.",
    )
    interpolation_cor_threshold: float = Field(
        0.25,
        description=(
            "Threshold on the sliding-window correlation to use for"
            " masking+interpolation.Pixels with less than this value are replaced by a"
            " weighted combination of neighboring pixels."
        ),
        ge=0.0,
        le=1.0,
    )
    interpolation_similarity_threshold: float = Field(
        0.3,
        description=(
            "Threshold on the phase similarity to use for masking+interpolation. "
            "Pixels with less than this value are replaced by a weighted combination of"
            " neighboring pixels."
        ),
        ge=0.0,
        le=1.0,
    )
    zero_correlation_where_interpolating: bool = Field(
        False,
        description=(
            "(for interpolation) Set the pixels where we have masked + interpolated"
            " data to have 0 correlation for the unwrapper. If False, the interpolated"
            " pixels keep their old correlation. If True, the behavior can be unwrapper"
            " dependent (the unwrapper may choose to skip over these pixels when"
            " solving)."
        ),
    )


class SnaphuOptions(BaseModel, extra="forbid"):
    ntiles: tuple[int, int] = Field(
        (1, 1),
        description=(
            "Number of tiles to split the inputs into using SNAPHU's internal tiling."
        ),
    )
    tile_overlap: tuple[int, int] = Field(
        (0, 0),
        description=(
            "Amount of tile overlap (in pixels) along the (row, col) directions."
        ),
    )
    n_parallel_tiles: int = Field(
        1,
        description="Number of tiles to unwrap in parallel for each interferogram.",
    )
    init_method: Literal["mcf", "mst"] = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )
    cost: Literal["defo", "smooth"] = Field(
        "smooth",
        description="Statistical cost mode method for SNAPHU.",
    )
    single_tile_reoptimize: bool = Field(
        False,
        description=(
            "If True, after unwrapping with multiple tiles, an additional"
            " post-processing unwrapping step is performed to re-optimize the unwrapped"
            " phase using a single tile"
        ),
    )

    _to_tuple = field_validator("ntiles", "tile_overlap", mode="before")(to_tuple)


class WhirlwindOptions(BaseModel, extra="forbid"):
    """User-tunable options for the whirlwind (ww) unwrapper.

    ww uses an internal rayon thread pool; pool size is shared across
    concurrent unwraps (``UnwrapOptions.n_parallel_jobs``). Empirically
    ww's MCF is ~40% parallel / 60% serial (Amdahl-limited by the
    primal-dual phase), so each unwrap saturates around 4 threads —
    beyond that, added cores give negligible single-IG speedup but
    concurrent unwraps still benefit from a larger shared pool.
    """

    num_threads: int | None = Field(
        default=None,
        description=(
            "Threads for ww's internal rayon pool. ``None`` = let ww"
            " use whatever ``WHIRLWIND_NUM_THREADS`` / ``RAYON_NUM_THREADS``"
            " is set externally (e.g. via SLURM/``taskset``), falling back"
            " to all CPUs. The pool is *shared* across the"
            " ``n_parallel_jobs`` concurrent unwraps, so the effective"
            " per-IG share is ``num_threads / n_parallel_jobs``."
        ),
        ge=1,
    )

    # --- Spiral persistent-scatterer interpolation pre-pass ------------------
    # Fills low-coherence valid pixels from a Gaussian distance-weighted average
    # of nearby high-coherence phasors before the solve. Like Goldstein, it only
    # informs the MCF: the integer cycle field is transferred back to the
    # original wrapped phase, so per-pixel values are preserved.
    interpolate: bool = Field(
        default=False,
        description=(
            "Enable the spiral PS interpolation pre-pass: every valid pixel"
            " with coherence below ``interp_cutoff`` is filled from nearby"
            " high-coherence pixels before unwrapping."
        ),
    )
    interp_cutoff: float = Field(
        default=0.5,
        description="Coherence below which a valid pixel is interpolated.",
        ge=0.0,
        le=1.0,
    )
    interp_num_neighbors: int = Field(
        default=20,
        description="Nearest high-coherence pixels averaged per interpolated pixel.",
        ge=1,
    )
    interp_max_radius: int = Field(
        default=51,
        description="Maximum search radius (pixels) for the spiral neighbor search.",
        ge=1,
    )
    interp_min_radius: int = Field(
        default=0,
        description="Minimum search radius (pixels); closer neighbors are skipped.",
        ge=0,
    )
    interp_alpha: float = Field(
        default=0.75,
        description="Gaussian distance-weighting falloff for the neighbor average.",
        gt=0.0,
    )

    # --- Connected-component cost / quality knobs ----------------------------
    conncomp_algorithm: Literal["snaphu", "linear"] = Field(
        default="snaphu",
        description=(
            "Connected-component grow algorithm. ``'snaphu'`` uses the default"
            " ambiguity-wiggle reliability grow; ``'linear'`` uses the older"
            " coherence-cost grow."
        ),
    )
    conncomp_min_coherence: float | Literal["auto"] | None = Field(
        default=None,
        description=(
            "For the ``'snaphu'`` grow, label pixels below this coherence as"
            " background. ``None`` (default) disables the floor and uses"
            " ``conncomp_reliability``; ``'auto'`` uses ww's looks-aware floor;"
            " a float in [0, 1] sets it explicitly."
        ),
    )
    conncomp_reliability: float = Field(
        default=0.5,
        description=(
            "Conservativeness of the ``'snaphu'`` grow, in inverse-variance"
            " (``1 / sigma2``) units: an edge becomes a component boundary when"
            " a one-cycle ambiguity flip across it costs no more than this. The"
            " default 0.5 is about a coherence-0.1 floor, dropping decorrelated"
            " water and near-noise while keeping real low-coherence land. 0"
            " labels nearly every unwrapped pixel. Used only when"
            " ``conncomp_min_coherence`` is ``None``."
        ),
        ge=0.0,
    )
    conncomp_thicken: bool = Field(
        default=True,
        description=(
            "SNAPHU ``ThickenCosts`` behavior for the ``'snaphu'`` grow: smooth"
            " each edge's cut strength laterally before cutting, so a one-pixel"
            " reliable bridge through a wide unreliable region no longer"
            " connects the two sides. Production SNAPHU always thickens."
        ),
    )
    # The remaining cost knobs apply only to the legacy `linear` grow. If more
    # than one is set, whirlwind resolves precedence as
    # sigma > cycle_prob > cost_threshold.
    cost_threshold: int = Field(
        default=50,
        description=(
            "For the ``'linear'`` grow, the component-boundary threshold in raw"
            " cost units. Larger makes more boundaries and smaller components."
        ),
        ge=0,
    )
    conncomp_sigma: float | None = Field(
        default=None,
        description=(
            "For the ``'linear'`` grow, set ``cost_threshold`` from a"
            " Gaussian-equivalent noise level (~3.5 reproduces the default 50)."
            " Higher is stricter. Takes precedence over ``cost_threshold`` and"
            " ``conncomp_cycle_prob``."
        ),
        gt=0.0,
    )
    conncomp_cycle_prob: float | None = Field(
        default=None,
        description=(
            "For the ``'linear'`` grow, set ``cost_threshold`` from a target"
            " per-edge one-cycle-correction probability (~2.4e-4 matches the"
            " default). Lower is stricter; ``conncomp_sigma`` takes precedence."
        ),
        gt=0.0,
        lt=1.0,
    )
    min_size_px: int = Field(
        default=100,
        description="Discard connected components smaller than this many pixels.",
        ge=1,
    )
    max_ncomps: int = Field(
        default=1024,
        description="Maximum number of connected components to keep (largest first).",
        ge=1,
    )

    # --- Phase-solve / component knobs ---------------------------------------
    bridge: bool = Field(
        default=True,
        description=(
            "Bridge disjoint connected components across low-coherence gaps so"
            " they share a consistent integer cycle. Disable to keep components"
            " fully independent."
        ),
    )
    connect_gaps: bool = Field(
        default=False,
        description=(
            "Draw phase paths across bounded runs of invalid pixels before"
            " unwrapping, then drop those synthetic pixels from both outputs."
            " The paths let the solver pick a relative 2pi level between"
            " regions that would otherwise be solved independently, which is"
            " what levels the NISAR sub-swaths. Unlike ``interpolate``, it"
            " extrapolates the phase slope on each side of a gap, so a winding"
            " estimate survives the crossing."
        ),
    )
    connect_gaps_max_px: int = Field(
        default=300,
        description=(
            "Widest run of invalid pixels, in pixels, that ``connect_gaps``"
            " will cross. This is a geometric width limit only; it does not"
            " distinguish why the pixels are invalid."
        ),
        ge=1,
    )

    @field_validator("conncomp_min_coherence")
    @classmethod
    def _validate_conncomp_min_coherence(
        cls, value: float | Literal["auto"] | None
    ) -> float | Literal["auto"] | None:
        if isinstance(value, float) and not 0.0 <= value <= 1.0:
            raise ValueError("conncomp_min_coherence must be between 0 and 1")
        return value

    # --- Internal Goldstein pre-filter ---------------------------------------
    # ww applies Goldstein filtering internally (fast Rust). For whirlwind,
    # dolphin routes its ``run_goldstein`` request here instead of running the
    # slower Python pre-process (see unwrap/_unwrap.py), so there is no
    # double-filtering. ``run_goldstein`` (using ``preprocess_options.alpha``)
    # takes precedence over ``goldstein_alpha`` when both are set.
    goldstein_alpha: float = Field(
        default=0.0,
        description=(
            "Goldstein filter strength for ww's internal pre-filter. 0 disables"
            " it. Ignored when ``UnwrapOptions.run_goldstein`` is set (that path"
            " supplies alpha from ``preprocess_options.alpha``)."
        ),
        ge=0.0,
    )
    goldstein_psize: int = Field(
        default=64,
        description="FFT patch size for ww's internal Goldstein filter.",
        ge=1,
    )


class TophuOptions(BaseModel, extra="forbid"):
    ntiles: tuple[int, int] = Field(
        (1, 1),
        description="Number of tiles to split the inputs into",
    )
    downsample_factor: tuple[int, int] = Field(
        (1, 1),
        description="Extra multilook factor to use for the coarse unwrap.",
    )
    init_method: Literal["mcf", "mst"] = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )
    cost: Literal["defo", "smooth"] = Field(
        "smooth",
        description="Statistical cost mode method for SNAPHU.",
    )

    _to_tuple = field_validator("ntiles", "downsample_factor", mode="before")(to_tuple)


class SpurtGeneralSettings(BaseModel):
    use_tiles: bool = Field(default=True, description="Tile up data spatially.")


class SpurtTilerSettings(BaseModel):
    """Class for holding tile generation settings."""

    max_tiles: int = Field(64, description="Maximum number of tiles allowed.", ge=1)
    target_points_for_generation: int = Field(
        120_000,
        description="Number of points used for determining tiles based on density.",
        gt=0,
    )
    target_points_per_tile: int = Field(
        900_000, description="Target points per tile when generating tiles.", gt=0
    )
    dilation_factor: float = Field(
        0.05,
        description=(
            "Dilation factor of non-overlapping tiles. 0.05 would lead to 5 percent"
            " dilation of the tile."
        ),
        ge=0.0,
    )


class SpurtSolverSettings(BaseModel):
    t_worker_count: int = Field(
        default=1,
        alias="t_worker_count",
        description=(
            "Number of workers for temporal unwrapping in parallel. Set value to <=0 to"
            " let workflow use default workers (ncpus - 1)."
        ),
    )
    s_worker_count: int = Field(
        default=1,
        description=(
            "Number of workers for spatial unwrapping in parallel. Set value to <=0"
            " to let workflow use (ncpus - 1)."
        ),
    )
    links_per_batch: int = Field(
        default=150_000,
        description=(
            "Temporal unwrapping operations over spatial links are performed in batches"
            " and each batch is solved in parallel."
        ),
        gt=0,
    )
    t_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant",
        description="Temporal unwrapping costs.",
    )
    t_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for temporal unwrapping.",
        gt=0.0,
    )
    s_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant", description="Spatial unwrapping costs.", alias="s_cost_type"
    )
    s_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for spatial unwrapping.",
        gt=0.0,
    )
    num_parallel_tiles: int = Field(
        default=1,
        description="Number of tiles to process in parallel. Set to 0 for all tiles.",
        ge=0.0,
    )


class SpurtMergerSettings(BaseModel):
    min_overlap_points: int = Field(
        default=25,
        description="Minimum number of overlap pixels to be considered valid.",
    )
    method: Literal["dirichlet"] = Field(
        default="dirichlet", description="Currently, only 'dirichlet' is supported."
    )
    bulk_method: Literal["integer", "L2"] = Field(
        default="L2", description="Method used to estimate bulk offset between tiles."
    )
    num_parallel_ifgs: int = Field(
        default=3,
        ge=0,
        description=(
            "Number of interferograms to merge in one batch. Use zero to merge all"
            " interferograms in a single batch."
        ),
    )


class SpurtOptions(BaseModel, extra="forbid"):
    """Options for running 3D unwrapping on a set of interferograms.

    Uses [`spurt`](https://github.com/isce-framework/spurt) to run the
    temporal/spatial unwrapping. Options are passed through to `spurt`
    library.

    """

    temporal_coherence_threshold: float = Field(
        0.7,
        description="Temporal coherence to pick pixels used on an irregular grid.",
        ge=0.0,
        lt=1.0,
    )
    similarity_threshold: float = Field(
        0.5,
        description=(
            "Similarity to pick pixels used on an irregular grid. Any pixel with"
            " similarity above `similarity_threshold` *or* above the temporal coherence"
            " threshold is chosen."
        ),
        ge=0.0,
        lt=1.0,
    )
    run_ambiguity_interpolation: bool = Field(
        True,
        description=(
            "After running spurt, interpolate the values that were masked during"
            " unwrapping (which are otherwise left as nan)."
        ),
    )
    general_settings: SpurtGeneralSettings = Field(default_factory=SpurtGeneralSettings)
    tiler_settings: SpurtTilerSettings = Field(default_factory=SpurtTilerSettings)
    solver_settings: SpurtSolverSettings = Field(default_factory=SpurtSolverSettings)
    merger_settings: SpurtMergerSettings = Field(default_factory=SpurtMergerSettings)


class UnwrapOptions(BaseModel, extra="forbid"):
    """Options for unwrapping after wrapped phase estimation."""

    run_unwrap: bool = Field(
        True,
        description=(
            "Whether to run the unwrapping step after wrapped phase estimation."
        ),
    )
    run_goldstein: bool = Field(
        False,
        description="Whether to run Goldstein filtering step on wrapped interferogram.",
    )
    run_interpolation: bool = Field(
        False,
        description="Whether to run interpolation step on wrapped interferogram.",
    )
    _directory: Path = PrivateAttr(Path("unwrapped"))
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    n_parallel_jobs: int = Field(
        -1,
        description=(
            "Number of interferograms to unwrap in parallel."
            " ``-1`` (default) auto-selects based on the unwrap method:"
            " ``1`` for snaphu/spurt (which already parallelise"
            " internally — snaphu over tiles, spurt over solver workers),"
            " and ``max(1, cpu_count // 4)`` for whirlwind (its MCF"
            " Amdahl-saturates around 4 threads per IG, so concurrency"
            " gives a ~2x wall-clock speedup at no per-IG cost)."
        ),
        ge=-1,
    )
    zero_where_masked: bool = Field(
        False,
        description=(
            "Set wrapped phase/correlation to 0 where mask is 0 before unwrapping. "
        ),
    )
    preprocess_options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    snaphu_options: SnaphuOptions = Field(default_factory=SnaphuOptions)
    tophu_options: TophuOptions = Field(default_factory=TophuOptions)
    spurt_options: SpurtOptions = Field(default_factory=SpurtOptions)
    whirlwind_options: WhirlwindOptions = Field(default_factory=WhirlwindOptions)
