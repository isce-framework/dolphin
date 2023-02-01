import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO, Union

from osgeo import gdal
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)
from ruamel.yaml import YAML

from dolphin import __version__ as _dolphin_version
from dolphin._log import get_log
from dolphin.io import format_nc_filename
from dolphin.utils import get_dates, sort_files_by_date

from ._enums import InterferogramNetworkType, OutputFormat, UnwrapMethod, WorkflowName

gdal.UseExceptions()
PathOrStr = Union[Path, str]

__all__ = [
    "Workflow",
]

logger = get_log()

# Specific to OPERA CSLC products:
OPERA_DATASET_NAME = "science/SENTINEL1/CSLC/grids/VV"
# for example, t087_185684_iw2
OPERA_BURST_RE = re.compile(
    r"t(?P<track>\d{3})_(?P<burst_id>\d{6})_(?P<subswath>iw[1-3])"
)


def _move_file_in_dir(path: PathOrStr, values: dict) -> Path:
    """Make sure the `path` is within `values['directory']`.

    Used for validation in different workflow steps' outputs.
    """
    p = Path(path)
    d = Path(values.get("directory", "."))
    if p.parent != d:
        return d / p.name
    else:
        return p


class PsOptions(BaseModel):
    """Options for the PS pixel selection portion of the workflow."""

    directory: Path = Path("PS")
    output_file: Path = Path("ps_pixels.tif")
    amp_dispersion_file: Path = Path("amp_dispersion.tif")
    amp_mean_file: Path = Path("amp_mean.tif")

    amp_dispersion_threshold: float = Field(
        0.35,
        description="Amplitude dispersion threshold to consider a pixel a PS.",
        gt=0.0,
    )

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

    # validators: Check directory exists, and that outputs are within directory
    _move_in_dir = validator(
        "output_file",
        "amp_dispersion_file",
        "amp_mean_file",
        always=True,
        allow_reuse=True,
    )(_move_file_in_dir)


class HalfWindow(BaseModel):
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


class PhaseLinkingOptions(BaseModel):
    """Configurable options for wrapped phase estimation."""

    directory: Path = Path("linked_phase")
    ministack_size: int = Field(
        15, description="Size of the ministack for sequential estimator.", gt=1
    )
    half_window = HalfWindow()


class InterferogramNetwork(BaseModel):
    """Options to determine the type of network for interferogram formation."""

    directory: Path = Path("interferograms")

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
    network_type = InterferogramNetworkType.SINGLE_REFERENCE

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

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


class UnwrapOptions(BaseModel):
    """Options for unwrapping after wrapped phase estimation."""

    run_unwrap: bool = False
    directory: Path = Path("unwrap")
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    tiles: Sequence[int] = [1, 1]
    init_method: str = "mcf"

    # validators


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
    max_ram_gb: float = Field(
        1.0,
        description="Maximum RAM (in GB) to use for processing",
        gt=0.1,
    )

    class Config:
        """Pydantic class configuration for BaseSettings."""

        # https://docs.pydantic.dev/usage/settings/#parsing-environment-variable-values
        env_prefix = "dolphin_"  # e.g. DOLPHIN_N_WORKERS=4 for n_workers
        fields = {
            "gpu_enabled": {"env": ["dolphin_gpu_enabled", "gpu"]},
        }
        extra = Extra.forbid  # raise error if extra fields passed in


class Inputs(BaseModel):
    """Options specifying input datasets for workflow."""

    cslc_file_list: List[Path] = Field(
        default_factory=list,
        description=(
            "List of CSLC files, or newline-delimited file "
            "containing list of CSLC files."
        ),
    )
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

    mask_files: List[str] = Field(
        default_factory=list,
        description=(
            "List of mask files to use, where convention is"
            " 0 for no data/invalid, and 1 for data."
        ),
    )

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

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

    @validator("subdataset", pre=True, always=True)
    def _check_for_opera(cls, v, values):
        cslc_file_list = values.get("cslc_file_list")
        # if we're not dealing with all OPERA files, just return whatever they gave
        if any(re.search(OPERA_BURST_RE, str(f)) is None for f in cslc_file_list):
            return v
        # Here we're dealing with all OPERA files, so we need to set the subdataset
        if v is None:
            # Assume that the user forgot to set the subdataset, and set it to the
            # default OPERA dataset name
            logger.info(
                "CSLC files look like OPERA files, setting subdataset to"
                f" {OPERA_DATASET_NAME}."
            )
            return OPERA_DATASET_NAME
        return v

    @validator("mask_files", pre=True)
    def _check_mask_files(cls, v):
        if isinstance(v, (str, Path)):
            # If they have passed a single mask file, return it as a list
            return [Path(v)]
        elif v is None:
            return []
        return [Path(f) for f in v]

    @root_validator
    def _check_slc_files_exist(cls, values):
        file_list = values.get("cslc_file_list")
        date_fmt = values.get("cslc_date_fmt")
        if not file_list:
            raise ValueError("Must specify list of input SLC files.")

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
            subdataset = values.get("subdataset")
            # gdal formatting function will raise an error if subdataset doesn't exist
            for f in file_list:
                format_nc_filename(f, subdataset)

        file_list, _ = sort_files_by_date(file_list, file_date_fmt=date_fmt)
        # Coerce the file_list to a list of Path objects, sorted
        values["cslc_file_list"] = [Path(f) for f in file_list]
        return values


class Outputs(BaseModel):
    """Options for the output format/compressions."""

    output_format: OutputFormat = OutputFormat.NETCDF
    scratch_directory: Path = Path("scratch")
    output_directory: Path = Path("output")
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

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

    # validators
    @validator("output_directory", "scratch_directory", always=True)
    def _dir_is_absolute(cls, v):
        return v.absolute()

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


class Workflow(BaseModel):
    """Configuration for the workflow.

    Required fields are in `Inputs`, where you must specify `cslc_file_list`.
    """

    workflow_name: str = WorkflowName.STACK

    inputs: Inputs
    outputs: Outputs = Field(default_factory=Outputs)

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)
    creation_time_utc: datetime = Field(default_factory=datetime.utcnow)
    dolphin_version: str = _dolphin_version

    # internal helpers
    # Stores the list of directories to be created by the workflow
    _directory_list: List[Path] = PrivateAttr(default_factory=list)
    _date_list: List[date] = PrivateAttr(default_factory=list)

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

    # validators
    @root_validator
    def _move_dirs_inside_scratch(cls, values):
        """Ensure outputs from workflow steps are within scratch directory."""
        scratch_dir = values["outputs"].scratch_directory
        # Save all directories as absolute paths
        scratch_dir = scratch_dir.absolute()

        # For each workflow step that has an output folder, move it inside
        # the scratch directory (if it's not already inside).
        # They may already be inside if we're loading from a json/yaml file.
        ps_opts = values["ps_options"]
        if not ps_opts.directory.parent == scratch_dir:
            ps_opts.directory = scratch_dir / ps_opts.directory
        ps_opts.directory = ps_opts.directory.absolute()

        if not ps_opts.amp_dispersion_file.parent.parent == scratch_dir:
            ps_opts.amp_dispersion_file = scratch_dir / ps_opts.amp_dispersion_file
        if not ps_opts.amp_mean_file.parent.parent == scratch_dir:
            ps_opts.amp_mean_file = scratch_dir / ps_opts.amp_mean_file
        if not ps_opts.output_file.parent.parent == scratch_dir:
            ps_opts.output_file = scratch_dir / ps_opts.output_file

        pl_opts = values["phase_linking"]
        if not pl_opts.directory.parent == scratch_dir:
            pl_opts.directory = scratch_dir / pl_opts.directory
        pl_opts.directory = pl_opts.directory.absolute()

        ifg_opts = values["interferogram_network"]
        if not ifg_opts.directory.parent == scratch_dir:
            ifg_opts.directory = scratch_dir / ifg_opts.directory
        ifg_opts.directory = ifg_opts.directory.absolute()

        unw_opts = values["unwrap_options"]
        if not unw_opts.directory.parent == scratch_dir:
            unw_opts.directory = scratch_dir / unw_opts.directory
        unw_opts.directory = unw_opts.directory.absolute()

        return values

    # Extra model exporting options beyond .dict() or .json()
    def to_yaml(self, output_path: Union[PathOrStr, TextIO]):
        """Save workflow configuration as a yaml file.

        Used to record the default-filled version of a supplied yaml.

        Parameters
        ----------
        output_path : Pathlike
            Path to the yaml file to save.
        """
        data = json.loads(self.json())
        y = YAML()
        if hasattr(output_path, "write"):
            y.dump(data, output_path)
        else:
            with open(output_path, "w") as f:
                y.dump(data, f)

    @classmethod
    def from_yaml(cls, yaml_path: PathOrStr):
        """Load a workflow configuration from a yaml file.

        Parameters
        ----------
        yaml_path : Pathlike
            Path to the yaml file to load.

        Returns
        -------
        Config
            Workflow configuration
        """
        y = YAML(typ="safe")
        with open(yaml_path, "r") as f:
            data = y.load(f)
        return cls(**data)

    def __init__(self, **data):
        """After validation, set up properties for use during workflow run."""
        super().__init__(**data)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            self.outputs.scratch_directory,
            self.outputs.output_directory,
            self.ps_options.directory,
            self.phase_linking.directory,
            self.interferogram_network.directory,
            self.unwrap_options.directory,
        ]

    def create_dir_tree(self, debug=False):
        """Create the directory tree for the workflow."""
        log = get_log(debug=debug)
        for d in self._directory_list:
            log.debug(f"Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)
