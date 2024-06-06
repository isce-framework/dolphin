from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from dolphin._log import get_log

from ._enums import UnwrapMethod

logger = get_log(__name__)

__all__ = [
    "UnwrapOptions",
    "PreprocessOptions",
    "SnaphuOptions",
    "TophuOptions",
    "SpurtOptions",
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
        0.5,
        description=(
            "Threshold on the correlation raster to use for interpolation. "
            "Pixels with less than this value are replaced by a weighted "
            "combination of neighboring pixels."
        ),
        ge=0.0,
        le=1.0,
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

    _to_tuple = field_validator("ntiles", "downsample_factor", mode="before")(to_tuple)


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
    _intermediate_folder: Path = PrivateAttr(Path("emcf_tmp"))


class SpurtTilerSettings(BaseModel):
    """Class for holding tile generation settings."""

    max_tiles: int = Field(16, description="Maximum number of tiles allowed.", ge=1)
    target_points_for_generation: int = Field(
        120000,
        description="Number of points used for determining tiles based on density.",
        gt=0,
    )
    target_points_per_tile: int = Field(
        800000, description="Target points per tile when generating tiles.", gt=0
    )
    dilation_factor: float = Field(
        0.05,
        description=(
            "Dilation factor of non-overlapping tiles. "
            "0.05 would lead to 5 percent dilation of the tile."
        ),
        ge=0.0,
    )


class SpurtSolverSettings(BaseModel):
    worker_count: int = Field(
        default=0,
        description=(
            "Number of workers for temporal unwrapping in parallel. "
            "Set value to <=0 to let workflow use default workers (ncpus - 1)."
        ),
    )
    links_per_batch: int = Field(
        default=10000,
        description=(
            "Temporal unwrapping operations over spatial links are performed in "
            "batches and each batch is solved in parallel."
        ),
        gt=0,
    )
    temp_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant", description="Temporal unwrapping costs."
    )
    temp_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for temporal unwrapping.",
        gt=0.0,
    )
    spatial_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant", description="Spatial unwrapping costs."
    )
    spatial_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for spatial unwrapping.",
        gt=0.0,
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


class SpurtOptions(BaseModel, extra="forbid"):
    """Options for running 3D unwrapping on a set of interferograms.

    Uses [`spurt`](https://github.com/isce-framework/spurt) to run the
    temporal/spatial unwrapping. Options are passed through to `spurt`
    library.

    """

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
        description=(
            "Whether to run Goldstein filtering step on wrapped interferogram."
        ),
    )
    run_interpolation: bool = Field(
        False,
        description="Whether to run interpolation step on wrapped interferogram.",
    )
    _directory: Path = PrivateAttr(Path("unwrapped"))
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    n_parallel_jobs: int = Field(
        1, description="Number of interferograms to unwrap in parallel."
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
