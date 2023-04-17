import re
from datetime import date, datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)

from dolphin import __version__ as _dolphin_version
from dolphin._log import get_log
from dolphin.io import DEFAULT_HDF5_OPTIONS, DEFAULT_TIFF_OPTIONS, format_nc_filename
from dolphin.utils import get_dates, sort_files_by_date

from ._enums import InterferogramNetworkType, UnwrapMethod, WorkflowName
from ._yaml_model import YamlModel

__all__ = [
    "Workflow",
]

logger = get_log(__name__)

# Specific to OPERA CSLC products:
OPERA_DATASET_NAME = "science/SENTINEL1/CSLC/grids/VV"
# for example, t087_185684_iw2
OPERA_BURST_RE = re.compile(
    r"t(?P<track>\d{3})_(?P<burst_id>\d{6})_(?P<subswath>iw[1-3])"
)


class PsOptions(BaseModel, extra=Extra.forbid):
    """Options for the PS pixel selection portion of the workflow."""

    _directory: Path = PrivateAttr(Path("PS"))
    _output_file: Path = PrivateAttr(Path("PS/ps_pixels.tif"))
    _amp_dispersion_file: Path = PrivateAttr(Path("PS/amp_dispersion.tif"))
    _amp_mean_file: Path = PrivateAttr(Path("PS/amp_mean.tif"))

    amp_dispersion_threshold: float = Field(
        0.25,
        description="Amplitude dispersion threshold to consider a pixel a PS.",
        gt=0.0,
    )


class HalfWindow(BaseModel, extra=Extra.forbid):
    """Class to hold half-window size for multi-looking during phase linking."""

    x: int = Field(11, description="Half window size (in pixels) for x direction", gt=0)
    y: int = Field(5, description="Half window size (in pixels) for y direction", gt=0)

    def to_looks(self):
        """Convert (x, y) half-window size to (row, column) looks."""
        return 2 * self.y + 1, 2 * self.x + 1

    @classmethod
    def from_looks(cls, row_looks: int, col_looks: int):
        """Create a half-window from looks."""
        return cls(x=col_looks // 2, y=row_looks // 2)


class PhaseLinkingOptions(BaseModel, extra=Extra.forbid):
    """Configurable options for wrapped phase estimation."""

    _directory: Path = PrivateAttr(Path("linked_phase"))
    ministack_size: int = Field(
        15, description="Size of the ministack for sequential estimator.", gt=1
    )
    half_window = HalfWindow()
    beta: float = Field(
        0.01,
        description=(
            "Beta regularization parameter for correlation matrix inversion. 0 is no"
            " regularization."
        ),
        gt=0.0,
        lt=1.0,
    )


class InterferogramNetwork(BaseModel, extra=Extra.forbid):
    """Options to determine the type of network for interferogram formation."""

    _directory: Path = PrivateAttr(Path("interferograms"))

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
        description="Maximum temporal baseline of interferograms.",
        gt=0,
    )
    indexes: Optional[List[Tuple[int, int]]] = Field(
        None,
        description=(
            "For manual-index network: List of (ref_idx, sec_idx) defining the"
            " interferograms to form."
        ),
    )
    network_type: InterferogramNetworkType = InterferogramNetworkType.SINGLE_REFERENCE

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

        # If nothing else specified, set to a single reference network
        values["network_type"] = InterferogramNetworkType.SINGLE_REFERENCE
        # and make sure the reference index is set
        if ref_idx is None:
            values["reference_idx"] = 0
        return values


class UnwrapOptions(BaseModel, extra=Extra.forbid):
    """Options for unwrapping after wrapped phase estimation."""

    run_unwrap: bool = Field(
        True,
        description=(
            "Whether to run the unwrapping step after wrapped phase estimation."
        ),
    )
    _directory: Path = PrivateAttr(Path("unwrap"))
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    tiles: List[int] = Field(
        [1, 1],
        description="Number of tiles to split the unwrapping into (for Tophu).",
    )
    init_method: str = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )


class WorkerSettings(BaseSettings):
    """Settings configurable based on environment variables."""

    gpu_enabled: bool = Field(
        True,
        description="Whether to use GPU for processing (if available)",
    )
    n_workers: int = Field(
        default_factory=cpu_count,
        ge=1,
        description=(
            "(For non-GPU) Number of cpu cores to use for Python multiprocessing. Uses"
            " `multiprocessing.cpu_count()` if not set."
        ),
    )
    threads_per_worker: int = Field(
        1,
        ge=1,
        description=(
            "Number of threads to use per worker. This sets the OMP_NUM_THREADS"
            " environment variable in each python process."
        ),
    )
    block_size_gb: float = Field(
        1.0,
        description="Size (in GB) of blocks of data to load at a time.",
        gt=0.001,
    )

    class Config:
        """Pydantic class configuration for BaseSettings."""

        extra = Extra.forbid
        # https://docs.pydantic.dev/usage/settings/#parsing-environment-variable-values
        env_prefix = "dolphin_"  # e.g. DOLPHIN_N_WORKERS=4 for n_workers
        fields = {
            "gpu_enabled": {"env": ["dolphin_gpu_enabled", "gpu"]},
        }


class InputOptions(BaseModel, extra=Extra.forbid):
    """Options specifying input datasets for workflow."""

    subdataset: Optional[str] = Field(
        None,
        description=(
            "If passing HDF5/NetCDF files, subdataset to use from CSLC files. "
            f"If not specified, but all `cslc_file_list` looks like {OPERA_BURST_RE}, "
            f" will use {OPERA_DATASET_NAME} as the subdataset."
        ),
    )
    cslc_date_fmt: str = Field(
        "%Y%m%d",
        description="Format of dates contained in CSLC filenames",
    )


class OutputOptions(BaseModel, extra=Extra.forbid):
    """Options for the output size/format/compressions."""

    output_resolution: Optional[Dict[str, int]] = Field(
        # {"x": 20, "y": 20},
        None,
        description="Output (x, y) resolution (in units of input data)",
    )
    strides: Dict[str, int] = Field(
        {"x": 1, "y": 1},
        description=(
            "Alternative to specifying output resolution: Specify the (x, y) strides"
            " (decimation factor) to perform while processing input. For example,"
            " strides of [4, 2] would turn an input resolution of [5, 10] into an"
            " output resolution of [20, 20]."
        ),
    )

    hdf5_creation_options: Dict = Field(
        DEFAULT_HDF5_OPTIONS,
        description="Options for `create_dataset` with h5py.",
    )
    gtiff_creation_options: List[str] = Field(
        list(DEFAULT_TIFF_OPTIONS),
        description="GDAL creation options for GeoTIFF files",
    )

    # validators
    @validator("output_resolution", "strides", pre=True, always=True)
    def _check_resolution(cls, v):
        """Allow the user to specify just one float, applying to both dimensions."""
        if isinstance(v, (int, float)):
            return {"x": v, "y": v}
        return v

    @validator("strides", always=True)
    def _check_strides_against_res(cls, strides, values):
        """Compute the output resolution from the strides."""
        resolution = values.get("output_resolution")
        if strides is not None and resolution is not None:
            raise ValueError("Cannot specify both strides and output_resolution.")
        elif strides is None and resolution is None:
            raise ValueError("Must specify either strides or output_resolution.")

        # Check that the dict has the correct keys
        if strides is not None:
            if not set(strides.keys()) == {"x", "y"}:
                raise ValueError("Strides must be a dict with keys 'x' and 'y'")
            # and that the strides are integers
            if not all([isinstance(v, int) for v in strides.values()]):
                raise ValueError("Strides must be integers")
        if resolution is not None:
            if not set(resolution.keys()) == {"x", "y"}:
                raise ValueError("Resolution must be a dict with keys 'x' and 'y'")
            # and that the resolution is valid, > 0. Can be int or float
            if any([v <= 0 for v in resolution.values()]):
                raise ValueError("Resolutions must be > 0")
            # TODO: compute strides from resolution
            raise NotImplementedError(
                "output_resolution not yet implemented. Use `strides`."
            )
        return strides


class Workflow(YamlModel):
    """Configuration for the workflow."""

    workflow_name: WorkflowName = WorkflowName.STACK

    # Paths to input/output files
    input_options: InputOptions = Field(default_factory=InputOptions)
    cslc_file_list: List[Path] = Field(
        default_factory=list,
        description=(
            "List of CSLC files, or newline-delimited file "
            "containing list of CSLC files."
        ),
    )
    mask_file: Optional[Path] = Field(
        None,
        description=(
            "Byte mask file used to ignore low correlation/bad data (e.g water mask)."
            " Convention is 0 for no data/invalid, and 1 for good data. Dtype must be"
            " uint8."
        ),
    )
    scratch_directory: Path = Field(
        Path("scratch"),
        description="Name of sub-directory to use for scratch files",
    )
    output_directory: Path = Field(
        Path("output"),
        description="Name of sub-directory to use for output files",
    )

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    amplitude_dispersion_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to existing Amplitude Dispersion file (1 per SLC region) for PS"
            " update calculation. If none provided, computed using the input SLC stack."
        ),
    )
    amplitude_mean_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to an existing Amplitude Mean files (1 per SLC region) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )

    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    save_compressed_slc: bool = Field(
        default=False,
        description=(
            "Whether the SAS should output and save the Compressed SLCs in addition to"
            " the standard product output."
        ),
    )

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)
    log_file: Optional[Path] = Field(
        # TODO: Probably more work to make sure log_file is implemented okay
        default=None,
        description="Path to output log file (in addition to logging to `stderr`).",
    )
    creation_time_utc: datetime = Field(
        default_factory=datetime.utcnow, description="Time the config file was created"
    )
    dolphin_version: str = Field(
        _dolphin_version, description="Version of Dolphin used."
    )

    # internal helpers
    # Stores the list of directories to be created by the workflow
    _directory_list: List[Path] = PrivateAttr(default_factory=list)
    _date_list: List[Union[date, List[date]]] = PrivateAttr(default_factory=list)

    class Config:
        extra = Extra.forbid
        schema_extra = {"required": ["cslc_file_list"]}

    @validator("output_directory", "scratch_directory", always=True)
    def _dir_is_absolute(cls, v):
        return v.resolve()

    # validators
    @validator("cslc_file_list", pre=True)
    def _check_input_file_list(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            v_path = Path(v)
            # Check if it's a newline-delimited list of input files
            if v_path.exists() and v_path.is_file():
                filenames = [Path(f) for f in v_path.read_text().splitlines()]
                # If given as relative paths, make them relative to the file
                parent = v_path.parent
                return [parent / f if not f.is_absolute() else f for f in filenames]
            else:
                raise ValueError(
                    f"Input file list {v_path} does not exist or is not a file."
                )

        return [Path(f) for f in v]

    @staticmethod
    def _is_opera_file_list(cslc_file_list):
        return all(
            re.search(OPERA_BURST_RE, str(f)) is not None for f in cslc_file_list
        )

    @root_validator
    def _check_slc_files_exist(cls, values):
        file_list = values.get("cslc_file_list")
        if not file_list:
            raise ValueError("Must specify list of input SLC files.")

        input_options = values.get("input_options")
        date_fmt = input_options.cslc_date_fmt
        # Filter out files that don't have dates in the filename
        file_matching_date = [Path(f) for f in file_list if get_dates(f, fmt=date_fmt)]
        if len(file_matching_date) < len(file_list):
            raise ValueError(
                f"Found {len(file_matching_date)} files with dates in the filename"
                f" out of {len(file_list)} files."
            )

        ext = file_list[0].suffix
        # If they're HDF5/NetCDF files, we need to check that the subdataset exists
        if ext in [".h5", ".nc"]:
            subdataset = input_options.subdataset
            if subdataset is None and cls._is_opera_file_list(file_list):
                # Assume that the user forgot to set the subdataset, and set it to the
                # default OPERA dataset name
                logger.info(
                    "CSLC files look like OPERA files, setting subdataset to"
                    f" {OPERA_DATASET_NAME}."
                )
                subdataset = input_options.subdataset = OPERA_DATASET_NAME

            # gdal formatting function will raise an error if subdataset doesn't exist
            for f in file_list:
                format_nc_filename(f, subdataset)

        # Coerce the file_list to a sorted list of absolute Path objects
        file_list, _ = sort_files_by_date(file_list, file_date_fmt=date_fmt)
        values["cslc_file_list"] = [Path(f).resolve() for f in file_list]
        return values

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """After validation, set up properties for use during workflow run."""
        super().__init__(*args, **kwargs)

        # Ensure outputs from workflow steps are within scratch directory.
        scratch_dir = self.scratch_directory
        # Save all directories as absolute paths
        scratch_dir = scratch_dir.resolve(strict=False)

        # For each workflow step that has an output folder, move it inside
        # the scratch directory (if it's not already inside).
        # They may already be inside if we're loading from a json/yaml file.
        for step in [
            "ps_options",
            "phase_linking",
            "interferogram_network",
            "unwrap_options",
        ]:
            opts = getattr(self, step)
            if not opts._directory.parent == scratch_dir:
                opts._directory = scratch_dir / opts._directory
            opts._directory = opts._directory.resolve(strict=False)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            scratch_dir,
            self.output_directory,
            self.ps_options._directory,
            self.phase_linking._directory,
            self.interferogram_network._directory,
            self.unwrap_options._directory,
        ]
        # Add the output PS files we'll create to the `PS` directory, making
        # sure they're inside the scratch directory
        ps_opts = self.ps_options
        ps_opts._amp_dispersion_file = scratch_dir / ps_opts._amp_dispersion_file
        ps_opts._amp_mean_file = scratch_dir / ps_opts._amp_mean_file
        ps_opts._output_file = scratch_dir / ps_opts._output_file

    def create_dir_tree(self, debug=False):
        """Create the directory tree for the workflow."""
        log = get_log(debug=debug)
        for d in self._directory_list:
            log.debug(f"Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)
