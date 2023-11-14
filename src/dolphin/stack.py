#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from osgeo import gdal

import dolphin._dates
from dolphin._dates import get_dates
from dolphin._log import get_log
from dolphin._types import DateOrDatetime, Filename

gdal.UseExceptions()
logger = get_log(__name__)


@dataclass
class BaseStack:
    """Base class for mini- and full stack classes."""

    file_list: list[Filename]
    is_compressed: list[bool]
    dates: Optional[list[tuple[DateOrDatetime, ...]]] = None
    file_date_fmt: str = "%Y%m%d"
    output_folder: Optional[Path] = None
    reference_idx: int = 0

    def __post_init__(self):
        if self.dates is None:
            self.dates = [get_dates(f, fmt=self.file_date_fmt) for f in self.file_list]

    @property
    def date_range(self):
        """Get the date range of the ministack as a string, e.g. '20210101_20210202'."""
        d0 = self.dates[0][0]
        d1 = self.dates[-1][-1]
        return dolphin._dates._format_date_pair(d0, d1, fmt=self.file_date_fmt)


@dataclass
class MiniStack(BaseStack):
    """Class for holding attributes about one mini-stack of SLCs.

    Used for planning the processing of a batch of SLCs.
    """

    def get_compressed_slc_name(self):
        """Get the name of the compressed SLC for this ministack."""
        return f"compressed_{self.date_range}.tif"


@dataclass
class CompressedSlc:
    """Class for holding attributes about one compressed SLC."""

    input_files: list[Filename]
    reference_date: DateOrDatetime
    dates: list[tuple[DateOrDatetime]]
    output_folder: Optional[Path] = None

    @property
    def name(self):
        """The filename of the compressed SLC for this ministack."""
        return f"compressed_{self.date_range}.tif"


@dataclass
class StackPlanner(BaseStack):
    """Class for planning the processing of a batch of SLCs."""

    max_num_compressed: int = 5
    output_folders: Optional[list[Path]] = None

    def plan(self):
        output_ministacks: list[MiniStack] = []
        comp_slc_files: list[Path] = []

        # Solve each ministack using the current chunk (and the previous compressed SLCs)
        ministack_starts = range(0, len(self.file_list), self.ministack_size)

        is_compressed = [False] * len(self.file_list)

        for full_stack_idx in ministack_starts:
            cur_slice = slice(full_stack_idx, full_stack_idx + self.ministack_size)
            cur_files = self.file_list[cur_slice].copy()
            cur_dates = self.dates[cur_slice].copy()

            # Add the existing compressed SLC files to the start
            cur_files = comp_slc_files + cur_files
            # limit the num comp slcs `max_num_compressed`
            cur_files = comp_slc_files[-self.max_num_compressed :] + cur_files

            where_compressed = np.where(is_compressed)[0]
            if where_compressed.size > 0:
                # If there are any compressed SLCs, set the reference to the last one
                last_compressed_idx = where_compressed[-1]
            else:
                # Otherwise, set the reference to the first SLC
                last_compressed_idx = 0

            cur_mini_stack = MiniStack(
                comp_slc_files[-self.max_num_compressed :] + cur_files,
                dates=cur_dates,
                is_compressed=is_compressed,
                reference_idx=last_compressed_idx,
                # outfile=cur_output_folder / f"{start_end}.vrt",
            )
            is_compressed = [True] + is_compressed

            output_ministacks.append(cur_mini_stack)
            # comp_slc_files.append(cur_comp_slc_file)
            # Make the current ministack output folder using the start/end dates
            if self.output_folders is None:
                cur_output_folder = None
            else:
                cur_output_folder = self.output_folder / cur_mini_stack.date_range
            self.output_folders.append(cur_output_folder)

        return output_ministacks
