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
    subdataset: str = Field(
        default=OPERA_DATASET_NAME,
        description="Name of the subdataset to use in the input NetCDF files.",
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
    amp_disp_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to existing Amplitude Dispersion file (1 per burst) for PS update"
            " calculation."
        ),
    )
    amp_mean_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to an existing Amplitude Mean files (1 per burst) for PS update"
            " calculation."
        ),
    )
    mask_files: List[Path] = Field(
        default_factory=list,
        description=(
            "List of mask files (e.g water mask), where convention is"
            " 0 for no data/invalid, and 1 for data."
        ),
    )
    # # TEC file in IONEX format for ionosphere correction
    # tec_file: str(required=False)
    tec_file: Optional[Path] = Field(
        default=None,
        description="Path to TEC file in IONEX format for ionosphere correction.",
    )

    # # Troposphere weather model
    # weather_model_file: str(required=False)
    weather_model_file: Optional[Path] = Field(
        default=None,
        description="Path to troposphere weather model file.",
    )


class PrimaryExecutable(YamlModel, extra=Extra.forbid):
    """Group describing the primary executable."""

    product_type: str = Field(
        default="DISP_S1",
        description="Product type of the PGE.",
    )


class ProductPathGroup(YamlModel, extra=Extra.forbid):
    """Group describing the product paths."""

    product_path: Path = Field(  # type: ignore
        # default=None, description="Directory where PGE will place results"
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

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)


class RunConfig(YamlModel, extra=Extra.forbid):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "disp_s1_workflow"

    input_file_group: InputFileGroup
    dynamic_ancillary_file_group: DynamicAncillaryFileGroup
    primary_executable: PrimaryExecutable = Field(default_factory=PrimaryExecutable)
    product_path_group: ProductPathGroup

    log_file: Path = Field(
        default=Path("disp_s1_workflow.log"), description="Path to the output log file."
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

        cslc_file_list = self.input_file_group.cslc_file_list
        output_directory = self.product_path_group.output_directory
        scratch_directory = self.product_path_group.scratch_path
        mask_files = self.dynamic_ancillary_file_group.mask_files
        input_options = dict(subdataset=self.input_file_group.subdataset)

        # Load the algorithm parameters from the file
        algorithm_parameters = AlgorithmParameters.from_yaml(
            self.dynamic_ancillary_file_group.algorithm_parameters_file
        )
        # This get's unpacked to load the rest of the parameters for the Workflow

        return Workflow(
            cslc_file_list=cslc_file_list,
            input_options=input_options,
            mask_files=mask_files,
            output_directory=output_directory,
            scratch_directory=scratch_directory,
            **algorithm_parameters.dict(),
        )
