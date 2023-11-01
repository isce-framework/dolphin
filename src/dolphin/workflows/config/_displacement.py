from __future__ import annotations

from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from dolphin import __version__ as _dolphin_version
from dolphin._log import get_log
from dolphin.utils import get_dates, sort_files_by_date

from ._common import (
    InputOptions,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    UnwrapOptions,
    WorkerSettings,
)
from ._yaml_model import YamlModel

__all__ = [
    "DisplacementWorkflow",
]

logger = get_log(__name__)


# Add a class for troposphere, ionosphere corrections, with geometry files and DEM
class CorrectionOptions(BaseModel, extra="forbid"):
    """Configuration for the auxillary phase corrections."""

    troposphere_files: List[Path] = Field(
        default_factory=list,
        description=(
            "List of weather-model files (one per date) for tropospheric corrections"
        ),
    )
    ionosphere_files: List[Path] = Field(
        default_factory=list,
        description=(
            "List of GNSS-derived TEC maps for ionospheric corrections (one per date)."
            " Source is https://cddis.nasa.gov/archive/gnss/products/ionex/"
        ),
    )
    geometry_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Line-of-sight geometry files for each burst/SLC stack area, for use in"
            " correction computations."
        ),
    )
    dem_file: Optional[Path] = Field(
        None,
        description="DEM file for tropospheric/ topographic phase corrections.",
    )

    @field_validator(
        "troposphere_files", "ionosphere_files", "geometry_files", mode="before"
    )
    @classmethod
    def _to_empty_list(cls, v):
        return v if v is not None else []


class DisplacementWorkflow(YamlModel):
    """Configuration for the workflow."""

    # Paths to input/output files
    input_options: InputOptions = Field(default_factory=InputOptions)
    cslc_file_list: List[Path] = Field(
        default_factory=list,
        description=(
            "list of CSLC files, or newline-delimited file "
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
    work_directory: Path = Field(
        Path("."),
        description="Name of sub-directory to use for writing output files",
        validate_default=True,
    )
    keep_paths_relative: bool = Field(
        False,
        description=(
            "Don't resolve filepaths that are given as relative to be absolute."
        ),
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
    correction_options: CorrectionOptions = Field(default_factory=CorrectionOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)
    log_file: Optional[Path] = Field(
        # TODO: Probably more work to make sure log_file is implemented okay
        default=None,
        description="Path to output log file (in addition to logging to `stderr`).",
    )
    benchmark_log_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Path to directory to write CPU/Memory usage logs. If none passed, will"
            " skip recording"
        ),
    )
    creation_time_utc: datetime = Field(
        default_factory=datetime.utcnow, description="Time the config file was created"
    )
    _dolphin_version: str = PrivateAttr(_dolphin_version)

    # internal helpers
    # Stores the list of directories to be created by the workflow
    _directory_list: List[Path] = PrivateAttr(default_factory=list)
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"required": ["cslc_file_list"]}
    )

    # validators
    @field_validator("cslc_file_list", mode="before")
    @classmethod
    def _check_input_file_list(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            v_path = Path(v)

            # Check if they've passed a glob pattern
            if len(list(glob(str(v)))) > 1:
                v = glob(str(v))
            # Check if it's a newline-delimited list of input files
            elif v_path.exists() and v_path.is_file():
                filenames = [Path(f) for f in v_path.read_text().splitlines()]

                # If given as relative paths, make them relative to the text file
                parent = v_path.parent
                return [parent / f if not f.is_absolute() else f for f in filenames]
            else:
                raise ValueError(
                    f"Input file list {v_path} does not exist or is not a file."
                )

        return list(v)

    @model_validator(mode="after")
    def _check_slc_files_exist(self) -> "DisplacementWorkflow":
        file_list = self.cslc_file_list
        if not file_list:
            raise ValueError("Must specify list of input SLC files.")

        input_options = self.input_options
        date_fmt = input_options.cslc_date_fmt
        # Filter out files that don't have dates in the filename
        files_matching_date = [Path(f) for f in file_list if get_dates(f, fmt=date_fmt)]
        if len(files_matching_date) < len(file_list):
            raise ValueError(
                f"Found {len(files_matching_date)} files with dates like {date_fmt} in"
                f" the filename out of {len(file_list)} files."
            )

        ext = file_list[0].suffix
        # If they're HDF5/NetCDF files, we need to check that the subdataset exists
        if ext in [".h5", ".nc"]:
            subdataset = input_options.subdataset
            if subdataset is None:
                raise ValueError(
                    "Must provide subdataset name for input NetCDF/HDF5 files."
                )

        # Coerce the file_list to a sorted list of Path objects
        self.cslc_file_list = [
            Path(f) for f in sort_files_by_date(file_list, file_date_fmt=date_fmt)[0]
        ]
        return self

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """After validation, set up properties for use during workflow run."""
        super().__init__(*args, **kwargs)

        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Save all directories as absolute paths
            self.work_directory = self.work_directory.resolve(strict=False)
            # Resolve all CSLC paths:
            self.cslc_file_list = [p.resolve(strict=False) for p in self.cslc_file_list]

        work_dir = self.work_directory
        # For each workflow step that has an output folder, move it inside
        # the work directory (if it's not already inside).
        # They may already be inside if we're loading from a json/yaml file.
        for step in [
            "ps_options",
            "phase_linking",
            "interferogram_network",
            "unwrap_options",
        ]:
            opts = getattr(self, step)
            if not opts._directory.parent == work_dir:
                opts._directory = work_dir / opts._directory
            if not self.keep_paths_relative:
                opts._directory = opts._directory.resolve(strict=False)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            work_dir,
            self.ps_options._directory,
            self.phase_linking._directory,
            self.interferogram_network._directory,
            self.unwrap_options._directory,
        ]
        # Add the output PS files we'll create to the `PS` directory, making
        # sure they're inside the work directory
        ps_opts = self.ps_options
        ps_opts._amp_dispersion_file = work_dir / ps_opts._amp_dispersion_file
        ps_opts._amp_mean_file = work_dir / ps_opts._amp_mean_file
        ps_opts._output_file = work_dir / ps_opts._output_file
