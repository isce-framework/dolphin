"""Module for creating PGE-compatible run configuration files."""

from pathlib import Path
from typing import ClassVar, List, Optional

from pydantic import BaseModel, Extra, Field

from .config import Workflow


class InputFileGroup(BaseModel, extra=Extra.forbid):
    """A group of input files."""

    input_file_paths: List[Path] = Field(
        default_factory=list,
        description="List of paths ot CSLC files.",
    )


class DynamicAncillaryFileGroup(BaseModel, extra=Extra.forbid):
    """A group of dynamic ancillary files."""

    algorithm_parameters: Path = Field(
        ...,
        description="Path to file containing SAS algorithm parameters.",
    )
    amp_disp_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to an existing Amplitude Dispersion file for PS update calculation."
        ),
    )
    amp_mean_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to an existing Amplitude Mean file for PS update calculation."
        ),
    )
    mask_files: List[Path] = Field(
        default_factory=list,
        description=(
            "List of mask files (e.g water mask), where convention is"
            " 0 for no data/invalid, and 1 for data."
        ),
    )


class PrimaryExecutable(BaseModel, extra=Extra.forbid):
    """Group describing the primary executable."""

    product_type: str = "DISP_S1"


class ProductPathGroup(BaseModel, extra=Extra.forbid):
    """Group describing the product paths."""

    product_path: Path = Field(
        default=..., description="Directory where PGE will place results"
    )
    scratch_path: Path = Field(
        default=Path("scratch"),
        description="Path to the scratch directory.",
    )
    sas_output_path: Path = Field(
        default=Path("sas_output_path"),
        description="Path to the SAS output directory.",
    )
    product_version: str = Field(
        default="0.1",
        description="Version of the product.",
    )


class AlgorithmParameters(Workflow):
    """Override the Workflow class to remove the Inputs and Outputs."""


class RunConfig(BaseModel, extra=Extra.forbid):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "disp_s1_workflow"

    input_file_group: InputFileGroup = Field(default_factory=InputFileGroup)
    dynamic_ancillary_file_group: DynamicAncillaryFileGroup
    primary_executable: PrimaryExecutable = Field(default_factory=PrimaryExecutable)
    product_path_group: ProductPathGroup

    log_file: Path = Field(
        default=Path("disp_s1_workflow.log"), description="Path to the output log file."
    )

    def to_workflow(self):
        """Convert this run configuration to a workflow."""
        # We need to go to/from the PGE format to our internal Workflow object:
        # Note that the top two levels of nesting can be accomplished by wrapping
        # the normal model export in a dict.
        #
        # The only things from the RunConfig that are used in the
        # Workflow are the input files and PS amp mean/disp files.
        # All the other things come from the AlgorithmParameters.

        # take the input file paths, use them for the Inputs
        # input_cslc_files = self.input_file_group.input_file_paths
