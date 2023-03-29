"""Module for creating PGE-compatible run configuration files."""

from pathlib import Path
from typing import ClassVar, List, Optional

from pydantic import Extra, Field

from ._yaml_model import YamlModel
from .config import (
    OPERA_DATASET_NAME,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    UnwrapOptions,
    WorkerSettings,
    Workflow,
)


class InputFileGroup(YamlModel):
    """A group of input files."""

    cslc_file_list: List[Path] = Field(
        default_factory=list,
        description="List of paths to CSLC files.",
    )

    frame_id: int = Field(
        ...,
        description="Frame ID of the bursts contained in `cslc_file_list`.",
    )

    class Config:
        """Pydantic config class."""

        extra = Extra.forbid
        schema_extra = {"required": ["cslc_file_list", "frame_id"]}


class DynamicAncillaryFileGroup(YamlModel, extra=Extra.forbid):
    """A group of dynamic ancillary files."""

    algorithm_parameters_file: Path = Field(  # type: ignore
        default=...,
        description="Path to file containing SAS algorithm parameters.",
    )
    amplitude_dispersion_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to existing Amplitude Dispersion file (1 per burst) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )
    amplitude_mean_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to an existing Amplitude Mean files (1 per burst) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )
    geometry_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to the incidence/azimuth-angle files (1 per burst). If none"
            " provided, corrections using incidence/azimuth-angle are skipped."
        ),
    )
    mask_file: Optional[Path] = Field(
        None,
        description=(
            "Optional Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data. Dtype"
            " must be uint8."
        ),
    )
    dem_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the DEM file covering full frame. If none provided, corrections"
            " using DEM are skipped."
        ),
    )
    # TEC file in IONEX format for ionosphere correction
    tec_files: Optional[List[Path]] = Field(
        default=None,
        description=(
            "List of Paths to TEC files (1 per date) in IONEX format for ionosphere"
            " correction. If none provided, ionosphere corrections are skipped."
        ),
    )

    # Troposphere weather model
    weather_model_files: Optional[List[Path]] = Field(
        default=None,
        description=(
            "List of Paths to troposphere weather model files (1 per date). If none"
            " provided, troposphere corrections are skipped."
        ),
    )


class PrimaryExecutable(YamlModel, extra=Extra.forbid):
    """Group describing the primary executable."""

    product_type: str = Field(
        default="DISP_S1_SINGLE",
        description="Product type of the PGE.",
    )


class ProductPathGroup(YamlModel, extra=Extra.forbid):
    """Group describing the product paths."""

    product_path: Path = Field(  # type: ignore
        default=...,
        description="Directory where PGE will place results",
    )
    scratch_path: Path = Field(
        default=Path("./scratch"),
        description="Path to the scratch directory.",
    )
    output_directory: Path = Field(
        default=Path("./output"),
        description="Path to the SAS output directory.",
        # The alias means that in the YAML file, the key will be "sas_output_path"
        # instead of "output_directory", but the python instance attribute is
        # "output_directory" (to match Workflow)
        alias="sas_output_path",
    )
    product_version: str = Field(
        default="0.1",
        description="Version of the product.",
    )
    save_compressed_slc: bool = Field(
        default=False,
        description=(
            "Whether the SAS should output and save the Compressed SLCs in addition to"
            " the standard product output."
        ),
    )


class AlgorithmParameters(YamlModel, extra=Extra.forbid):
    """Class containing all the other [`Workflow`][dolphin.workflows.config] classes."""

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    subdataset: str = Field(
        default=OPERA_DATASET_NAME,
        description="Name of the subdataset to use in the input NetCDF files.",
    )


class RunConfig(YamlModel, extra=Extra.forbid):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "disp_s1_workflow"

    input_file_group: InputFileGroup
    dynamic_ancillary_file_group: DynamicAncillaryFileGroup
    primary_executable: PrimaryExecutable = Field(default_factory=PrimaryExecutable)
    product_path_group: ProductPathGroup

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)

    log_file: Optional[Path] = Field(
        default=Path("output/disp_s1_workflow.log"),
        description="Path to the output log file in addition to logging to stderr.",
    )

    # Override the constructor to allow recursively construct without validation
    @classmethod
    def construct(cls, **kwargs):
        if "input_file_group" not in kwargs:
            kwargs["input_file_group"] = InputFileGroup._construct_empty()
        if "dynamic_ancillary_file_group" not in kwargs:
            kwargs["dynamic_ancillary_file_group"] = (
                DynamicAncillaryFileGroup._construct_empty()
            )
        if "product_path_group" not in kwargs:
            kwargs["product_path_group"] = ProductPathGroup._construct_empty()
        return super().construct(
            **kwargs,
        )

    def to_workflow(self):
        """Convert to a [`Workflow`][dolphin.workflows.config.Workflow] object."""
        # We need to go to/from the PGE format to our internal Workflow object:
        # Note that the top two levels of nesting can be accomplished by wrapping
        # the normal model export in a dict.
        #
        # The things from the RunConfig that are used in the
        # Workflow are the input files, PS amp mean/disp files,
        # the output directory, and the scratch directory.
        # All the other things come from the AlgorithmParameters.

        workflow_name = self.primary_executable.product_type.replace(
            "DISP_S1_", ""
        ).lower()
        cslc_file_list = self.input_file_group.cslc_file_list
        output_directory = self.product_path_group.output_directory
        scratch_directory = self.product_path_group.scratch_path
        mask_file = self.dynamic_ancillary_file_group.mask_file
        amplitude_mean_files = self.dynamic_ancillary_file_group.amplitude_mean_files
        amplitude_dispersion_files = (
            self.dynamic_ancillary_file_group.amplitude_dispersion_files
        )

        # Load the algorithm parameters from the file
        algorithm_parameters = AlgorithmParameters.from_yaml(
            self.dynamic_ancillary_file_group.algorithm_parameters_file
        )
        param_dict = algorithm_parameters.dict()
        input_options = dict(subdataset=param_dict.pop("subdataset"))

        # This get's unpacked to load the rest of the parameters for the Workflow
        return Workflow(
            workflow_name=workflow_name,
            cslc_file_list=cslc_file_list,
            input_options=input_options,
            mask_file=mask_file,
            output_directory=output_directory,
            scratch_directory=scratch_directory,
            save_compressed_slc=self.product_path_group.save_compressed_slc,
            amplitude_mean_files=amplitude_mean_files,
            amplitude_dispersion_files=amplitude_dispersion_files,
            # These ones directly translate
            worker_settings=self.worker_settings,
            log_file=self.log_file,
            # Finally, the rest of the parameters are in the algorithm parameters
            **param_dict,
        )

    @classmethod
    def from_workflow(
        cls, workflow: Workflow, frame_id: int, algorithm_parameters_file: Path
    ):
        """Convert from a [`Workflow`][dolphin.workflows.config.Workflow] object.

        This is the inverse of the to_workflow method, although there are more
        fields in the PGE version, so it's not a 1-1 mapping.

        Since there's no `frame_id` or `algorithm_parameters_file` in the
        [`Workflow`][dolphin.workflows.config.Workflow] object, we need to pass
        those in as arguments.

        This is mostly used as preliminary setup to further edit the fields.
        """
        # Load the algorithm parameters from the file
        alg_param_dict = workflow.dict(include=AlgorithmParameters.__fields__.keys())
        AlgorithmParameters(**alg_param_dict).to_yaml(algorithm_parameters_file)
        # This get's unpacked to load the rest of the parameters for the Workflow

        return cls(
            input_file_group=InputFileGroup(
                cslc_file_list=workflow.cslc_file_list,
                frame_id=frame_id,
            ),
            dynamic_ancillary_file_group=DynamicAncillaryFileGroup(
                algorithm_parameters_file=algorithm_parameters_file,
                # amplitude_dispersion_files=workflow.amplitude_dispersion_files,
                # amplitude_mean_files=workflow.amplitude_mean_files,
                mask_file=workflow.mask_file,
                # tec_file=workflow.tec_file,
                # weather_model_file=workflow.weather_model_file,
            ),
            primary_executable=PrimaryExecutable(
                product_type=f"DISP_S1_{str(workflow.workflow_name.upper())}",
            ),
            product_path_group=ProductPathGroup(
                product_path=workflow.output_directory,
                scratch_path=workflow.scratch_directory,
                sas_output_path=workflow.output_directory,
            ),
            worker_settings=workflow.worker_settings,
            log_file=workflow.log_file,
        )
