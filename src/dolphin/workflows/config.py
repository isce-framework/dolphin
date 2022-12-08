from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    BaseSettings,
    DirectoryPath,
    Field,
    root_validator,
    validator,
)

from dolphin import __version__ as _dolphin_version
from dolphin import _show_versions

from ._enums import InterferogramNetworkType, OutputFormat, UnwrapMethod

PathOrStr = Union[Path, str]


def _check_and_make_dir(path: PathOrStr) -> Path:
    """Check for the existence of a directory.

    Create the directory if it doesn't exist.
    Returns `path` as an absolute Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.absolute()


def _move_file_in_dir(path: PathOrStr, values: dict) -> Path:
    # Make sure the file is within `directory`
    p = Path(path)
    d = Path(values.get("directory", "."))
    if not p.parent == d:
        return d / p.name
    else:
        return p


class PsOptions(BaseModel):
    """Options for the PS pixel selection portion of the workflow."""

    directory: Path = Path("PS")
    output_file: Optional[Path] = Path("ps_pixels.tif")
    amp_dispersion_file: Optional[Path] = Path("amp_dispersion.tif")
    amp_mean_file: Optional[Path] = Path("amp_mean.tif")

    amp_dispersion_threshold: float = Field(
        0.42,
        description="Amplitude dispersion threshold to consider a pixel a PS.",
        gt=0.0,
    )

    # validators: Check directory exists, and that outputs are within directory
    _dir_must_exist = validator("directory", allow_reuse=True, always=True, pre=True)(
        _check_and_make_dir
    )
    validator(
        "output_file",
        "amp_dispersion_file",
        "amp_mean_file",
        allow_reuse=True,
        always=True,
    )(_move_file_in_dir)


class HalfWindow(BaseModel):
    """Class to hold half-window size for multi-looking during phase linking."""

    x: int = Field(11, description="Half window size (in pixels) for x direction", gt=0)
    y: int = Field(5, description="Half window size (in pixels) for y direction", gt=0)

    def to_looks(self):
        """Convert (x, y) half-window size to (row, column) looks."""
        return 2 * self.y + 1, 2 * self.x + 1

    @classmethod
    def from_looks(cls, looks: Sequence[int]):
        """Create a half-window from looks."""
        return cls(x=(looks[1] - 1) // 2, y=(looks[0] - 1) // 2)


class PhaseLinkingOptions(BaseModel):
    """Configurable options for wrapped phase estimation."""

    directory: Path = Path("linked_phase")
    ministack_size: int = Field(
        15, description="Size of the ministack for sequential estimator.", gt=1
    )
    half_window = HalfWindow()
    compressed_slc_file: Path = Path("compressed_slc.tif")
    temp_coh_file: Path = Path("temp_coh.tif")

    # validators
    _dir_must_exist = validator("directory", allow_reuse=True, always=True, pre=True)(
        _check_and_make_dir
    )
    validator("compressed_slc_file", "temp_coh_file", allow_reuse=True, always=True)(
        _move_file_in_dir
    )

    @staticmethod
    def _format_date_pair(start: date, end: date, fmt="%Y%m%d") -> str:
        return f"{start.strftime(fmt)}_{end.strftime(fmt)}"


class InterferogramNetwork(BaseModel):
    """Options to determine the type of network for interferogram formation."""

    reference_idx: Optional[int] = Field(
        None,
        description=(
            "For single-reference network: Index of the reference image in the network"
        ),
    )
    max_bandwidth: Optional[int] = Field(
        None,
        description="Max `n` to form the nearest-`n` interferograms by index.",
        gt=1,
    )
    max_temporal_baseline: Optional[int] = Field(
        None,
        description="Maximum temporal baseline of interferograms",
        gt=0,
    )
    network_type: InterferogramNetworkType

    # validation
    @root_validator
    def _check_network_type(cls, values):
        ref_idx = values.get("reference_idx")
        max_bw = values.get("max_bandwidth")
        max_tb = values.get("max_temporal_baseline")
        # Check if more than one has been set:
        if sum([ref_idx is not None, max_bw is not None, max_tb is not None]) > 1:
            raise ValueError(
                "Only one of `reference_idx`, `max_bandwidth`, or"
                " `max_temporal_baseline` can be set."
            )
        if max_tb is not None:
            values["network_type"] = InterferogramNetworkType.TEMPORAL_BASELINE
            return values

        if max_bw is not None:
            values["network_type"] = InterferogramNetworkType.BANDWIDTH
            return values

        if ref_idx is None:
            ref_idx = 0
        values["network_type"] = InterferogramNetworkType.SINGLE_REFERENCE
        return values


class UnwrapOptions(BaseModel):
    """Options for unwrapping after wrapped phase estimation."""

    run_unwrap: bool = False
    directory: Path = Path("unwrap")
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    tiles: Sequence[int] = (1, 1)
    init_method: str = "mcf"


class WorkerSettings(BaseSettings):
    """Settings configurable based on environment variables."""

    gpu_enabled: bool = Field(
        True,
        description="Whether to use GPU for processing (if available)",
    )
    gpu_id: int = Field(
        0,
        description="Index of the GPU to use for processing (if GPU)",
    )
    # n_workers: int = PositiveInt(16)
    n_workers: int = Field(
        16, ge=1, description="Number of cpu cores to use for processing (if CPU)"
    )
    max_ram: int = Field(
        1.0,
        description="Maximum RAM (in GB) to use for processing",
        gt=0.1,
    )

    class Config:
        """Pydantic class configuration for BaseSettings."""

        # https://docs.pydantic.dev/usage/settings/#parsing-environment-variable-values
        env_prefix = "dolphin_"  # e.g. DOLPHIN_N_WORKERS=4 for n_workers
        fields = {
            "gpu_enabled": {"env": ["gpu_enabled", "gpu"]},
        }


class InputOptions(BaseModel):
    """Options specifying input datasets for workflow."""

    cslc_file_list: List[PathOrStr] = Field(
        default_factory=list, description="List of CSLC files"
    )
    cslc_directory: DirectoryPath = Field(None, description="Path to CSLC files")
    cslc_file_ext: str = Field(
        ".nc",
        description="Extension of CSLC files (if providing `cslc_directory`)",
    )

    mask_files: List[str] = Field(
        default_factory=list,
        description=(
            "List of mask files to use, where convention is"
            " 0 for no data/invalid, and 1 for data."
        ),
    )

    @root_validator
    def _check_slc_files_exist(cls, values):
        file_list = values.get("cslc_file_list")
        directory = values.get("cslc_directory")
        if not file_list:
            if not directory:
                raise ValueError("Must specify either cslc_file_list or cslc_directory")
            directory = directory.absolute()
            ext = values.get("cslc_file_ext")
            file_list = sorted(directory.glob(f"*{ext}"))
            values["cslc_file_list"] = file_list
        return values


class OutputOptions(BaseModel):
    """Options for the output format/compressions."""

    output_format: OutputFormat = OutputFormat.NETCDF
    scratch_directory: Path = Path("scratch")
    # TODO: spacing, strides, etc.
    output_directory: Path = Path("output")

    hdf5_creation_options: Dict = Field(
        dict(
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        ),
        description="Options for `create_dataset` with h5py.",
    )
    gtiff_creation_options: List[str] = Field(
        ["TILED=YES", "COMPRESS=DEFLATE", "ZLEVEL=5"],
        description="GDAL creation options for GeoTIFF files",
    )

    # validators
    _dir_must_exist = validator(
        "output_directory", "scratch_directory", allow_reuse=True, always=True
    )(_check_and_make_dir)


class Config(BaseModel):
    """Configuration for the workflow.

    Required fields are in InputOptions:
    Must specify either `cslc_file_list` or `cslc_directory` and
    a file extension.
    """

    input_files: InputOptions
    output_options = OutputOptions()

    worker_settings = WorkerSettings()

    # Options for each step in the workflow
    phase_linking = PhaseLinkingOptions()
    interferogram_network = InterferogramNetwork()
    unwrap_options = UnwrapOptions()
    ps_options = PsOptions()

    # General workflow metadata
    runtime: datetime = Field(default_factory=datetime.utcnow)
    dolphin_version = _dolphin_version
    sys_info: Dict = Field(default_factory=_show_versions._get_sys_info)
