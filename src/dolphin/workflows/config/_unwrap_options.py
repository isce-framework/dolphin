from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)
from pydantic.dataclasses import dataclass

from dolphin._log import get_log

from ._enums import UnwrapMethod

logger = get_log(__name__)

__all__ = ["UnwrapOptions", "Unwrap3DOptions"]


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
        description=("Whether to run interpolation step on wrapped interferogram."),
    )
    _directory: Path = PrivateAttr(Path("unwrapped"))
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    n_parallel_jobs: int = Field(
        1, description="Number of interferograms to unwrap in parallel."
    )
    ntiles: tuple[int, int] = Field(
        (1, 1),
        description=(
            "(`snaphu-py` or multiscale unwrapping) Number of tiles to split "
            "the inputs into"
        ),
    )
    downsample_factor: tuple[int, int] = Field(
        (1, 1),
        description=(
            "(for multiscale unwrapping) Extra multilook factor to use for the coarse"
            " unwrap."
        ),
    )
    tile_overlap: tuple[int, int] = Field(
        (0, 0),
        description=(
            "(for use in `snaphu-py`) Amount of tile overlap (in pixels) along the"
            " (row, col) directions."
        ),
    )
    n_parallel_tiles: int = Field(
        1,
        description=(
            "(for snaphu) Number of tiles to unwrap in parallel for each interferogram."
        ),
    )
    init_method: Literal["mcf", "mst"] = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )
    cost: Literal["defo", "smooth"] = Field(
        "smooth",
        description="Statistical cost mode method for SNAPHU.",
    )
    zero_where_masked: bool = Field(
        False,
        description=(
            "Set wrapped phase/correlation to 0 where mask is 0 before unwrapping. "
        ),
    )
    alpha: float = Field(
        0.5,
        description=(
            "(for Goldstein filtering) Power parameter for Goldstein algorithm."
        ),
    )
    max_radius: int = Field(
        51,
        ge=0.0,
        description=("(for interpolation) maximum radius to find scatterers."),
    )
    interpolation_cor_threshold: float = Field(
        0.5,
        description=" Threshold on the correlation raster to use for interpolation. "
        "Pixels with less than this value are replaced by a weighted "
        "combination of neighboring pixels.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("ntiles", "downsample_factor", mode="before")
    @classmethod
    def _to_tuple(cls, v):
        if v is None:
            return (1, 1)
        elif isinstance(v, int):
            return (v, v)
        return v


class Unwrap3DOptions(BaseModel, extra="forbid"):
    """Options for running 3D unwrapping on a set of interferograms.

    Currently used in [`spurt`](https://github.com/isce-framework/spurt) only.

    """

    general_settings: SpurtGeneralSettings
    tiler_settings: SpurtTilerSettings
    solver_settings: SpurtSolverSettings
    merger_settings: SpurtMergerSettings


@dataclass
class SpurtGeneralSettings:
    """Settings associated with breaking data into tiles.

    Parameters
    ----------
    use_tiles: bool
        Tile up data spatially.
    output_folder: str
        Path to output folder.

    """

    use_tiles: bool = True
    intermediate_folder: str = "./emcf_tmp"
    output_folder: str = "./emcf"


@dataclass
class SpurtTilerSettings:
    """Class for holding tile generation settings.

    Parameters
    ----------
    max_tiles: int
        Maximum number of tiles allowed.
    target_points_for_generation: int
        Number of points used for determining tiles based on density.
    target_points_per_tile: int
        Target points per tile when generating tiles.
    dilation_factor: float
        Dilation factor of non-overlapping tiles. 0.05 would lead to
        10 percent dilation of the tile.

    """

    max_tiles: int = 16
    target_points_for_generation: int = 120000
    target_points_per_tile: int = 800000
    dilation_factor: float = 0.05

    def __post_init__(self):
        if self.max_tiles < 1:
            errmsg = f"max_tiles must be atleast 1, got {self.max_tiles}"
            raise ValueError(errmsg)
        if self.dilation_factor < 0.0:
            errmsg = f"dilation_factor must be >= 0., got {self.dilation_factor}"
            raise ValueError(errmsg)
        if self.target_points_for_generation <= 0:
            errmsg = (
                "target_points_for_generation must be > 0,"
                f" got {self.target_points_for_generation}"
            )
            raise ValueError(errmsg)
        if self.target_points_per_tile <= 0.0:
            errmsg = (
                f"target_points_per_tile must be > 0, got {self.target_points_per_tile}"
            )
            raise ValueError(errmsg)


@dataclass
class SpurtSolverSettings:
    """Settings associated with Extended Minimum Cost Flow (EMCF) workflow.

    Parameters
    ----------
    worker_count: int
        Number of workers for temporal unwrapping in parallel. Set value to <=0
        to let workflow use default workers (ncpus - 1).
    links_per_batch: int
        Temporal unwrapping operations over spatial links are performed in batches
        and each batch is solved in parallel.
    t_cost_type: str
        Temporal unwrapping costs. Can be one of 'constant', 'distance',
        'centroid'.
    t_cost_scale: float
        Scale factor used in computing edge costs for temporal unwrapping.
    s_cost_type: str
        Spatial unwrapping costs. Can be one of 'constant', 'distance',
        'centroid'.
    s_cost_scale: float
        Scale factor used in computing edge costs for spatial unwrapping.

    """

    worker_count: int = 0
    links_per_batch: int = 10000
    t_cost_type: str = "constant"
    t_cost_scale: float = 100.0
    s_cost_type: str = "constant"
    s_cost_scale: float = 100.0

    def __post_init__(self):
        cost_types = {"constant", "distance", "centroid"}
        if self.t_cost_type not in cost_types:
            errmsg = f"t_cost_type must be one of {cost_types}, got {self.t_cost_type}"
            raise ValueError(errmsg)
        if self.s_cost_type not in cost_types:
            errmsg = f"s_cost_type must be one of {cost_types}, got {self.s_cost_type}"
            raise ValueError(errmsg)
        if self.links_per_batch <= 0:
            errmsg = f"links_per_batch must be > 0, got {self.links_per_batch}"
            raise ValueError(errmsg)
        if self.t_cost_scale <= 0.0:
            errmsg = f"t_cost_scale must be > 0, got {self.t_cost_scale}"
            raise ValueError(errmsg)
        if self.s_cost_scale <= 0.0:
            errmsg = f"s_cost_scale must be > 0, got {self.s_cost_scale}"
            raise ValueError(errmsg)


@dataclass
class SpurtMergerSettings:
    """Class for holding tile merging settings.

    Parameters
    ----------
    min_overlap_points: int
        Minimum number of pixels in overlap region for it to be considered
        valid.
    method: str
        Currently, only "dirichlet" is supported.
    bulk_method: str
        Method used to estimate bulk offset between tiles.

    """

    min_overlap_points: int = 25
    method: str = "dirichlet"
    bulk_method: str = "L2"

    def __post_init__(self):
        bulk_methods = {"integer", "L2"}
        if self.bulk_method not in bulk_methods:
            errmsg = (
                f"bulk_method must be one of {bulk_methods}. got {self.bulk_method}"
            )
            raise ValueError(errmsg)

        if self.method != "dirichlet":
            errmsg = f"'dirichlet' is the only valid option, got {self.method}"
            raise ValueError(errmsg)
