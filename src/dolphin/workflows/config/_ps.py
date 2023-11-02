from __future__ import annotations

from pathlib import Path
from typing import Any, List

from pydantic import ConfigDict, Field, field_validator

from dolphin._log import get_log

from ._common import (
    InputOptions,
    OutputOptions,
    PsOptions,
    WorkflowBase,
    _read_file_list_or_glob,
)

__all__ = [
    "PsWorkflow",
]

logger = get_log(__name__)

# def create_ps(
#     *,
#     slc_vrt_file: Filename,
#     output_file: Filename,
#     output_amp_mean_file: Filename,
#     output_amp_dispersion_file: Filename,
#     amp_dispersion_threshold: float = 0.25,
#     existing_amp_mean_file: Optional[Filename] = None,
#     existing_amp_dispersion_file: Optional[Filename] = None,
#     nodata_mask: Optional[np.ndarray] = None,
#     update_existing: bool = False,
#     block_shape: tuple[int, int] = (512, 512),
#     show_progress: bool = True,
# ):


class PsWorkflow(WorkflowBase):
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

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    # internal helpers
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"required": ["cslc_file_list"]}
    )

    # validators
    # reuse the _read_file_list_or_glob
    _check_cslc_file_glob = field_validator("cslc_file_list", mode="before")(
        _read_file_list_or_glob
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """After validation, set up properties for use during workflow run."""
        super().__init__(*args, **kwargs)

        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Resolve all CSLC paths:
            self.cslc_file_list = [p.resolve(strict=False) for p in self.cslc_file_list]

        work_dir = self.work_directory
        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            work_dir,
            self.ps_options._directory,
        ]
        # move the folders inside the work directory
        ps_opts = self.ps_options
        if not ps_opts._directory.parent == work_dir:
            ps_opts._directory = work_dir / ps_opts._directory
        if not self.keep_paths_relative:
            ps_opts._directory = ps_opts._directory.resolve(strict=False)

        # Add the output PS files we'll create to the `PS` directory, making
        # sure they're inside the work directory
        ps_opts._amp_dispersion_file = work_dir / ps_opts._amp_dispersion_file
        ps_opts._amp_mean_file = work_dir / ps_opts._amp_mean_file
        ps_opts._output_file = work_dir / ps_opts._output_file
