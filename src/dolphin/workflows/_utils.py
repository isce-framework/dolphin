from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from .config import DisplacementWorkflow

logger = logging.getLogger(__name__)


def _create_burst_cfg(
    cfg: DisplacementWorkflow,
    burst_id: str,
    grouped_slc_files: dict[str, list[Path]],
    grouped_amp_mean_files: dict[str, list[Path]],
    grouped_amp_dispersion_files: dict[str, list[Path]],
) -> DisplacementWorkflow:
    cfg_temp_dict = cfg.model_dump(exclude={"cslc_file_list"})

    # Just update the inputs and the work directory
    top_level_work = cfg_temp_dict["work_directory"]
    cfg_temp_dict.update({"work_directory": top_level_work / burst_id})
    cfg_temp_dict["cslc_file_list"] = grouped_slc_files[burst_id]
    cfg_temp_dict["amplitude_mean_files"] = grouped_amp_mean_files[burst_id]
    cfg_temp_dict["amplitude_dispersion_files"] = grouped_amp_dispersion_files[burst_id]
    return DisplacementWorkflow(**cfg_temp_dict)


def _remove_dir_if_empty(d: Path) -> None:
    with contextlib.suppress(OSError):
        d.rmdir()
