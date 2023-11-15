#!/usr/bin/env python
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime
from os import fspath
from pathlib import Path
from typing import Sequence

import numpy as np

import dolphin._dates
from dolphin._log import get_log
from dolphin._types import DateOrDatetime, Filename

logger = get_log(__name__)

# Sentinel value for when no reference date is provided
# Appeases mypy
_DUMMY_DATE = datetime(1900, 1, 1)


@dataclass(frozen=True)
class BaseStack:
    """Base class for mini- and full stack classes."""

    file_list: Sequence[Filename]
    """List of SLC filenames in the ministack."""
    dates: Sequence[Sequence[DateOrDatetime]]
    """List of date sequences, one for each SLC in the ministack.
    Each item is a list/tuple of datetime.date or datetime.datetime objects,
    as returned by [dolphin._dates.get_dates][].
    """
    is_compressed: Sequence[bool]
    """List of booleans indicating whether each SLC is compressed or real."""
    reference_date: DateOrDatetime = _DUMMY_DATE
    """Reference date to be used for understanding output interferograms.
    Note that this may be different from `dates[reference_idx]` if the ministack
    starts with a compressed SLC which has an earlier "base phase", which
    is used as the phase linking references.
    It will propagate across ministacks when we always use `reference_idx=0`.
    """
    file_date_fmt: str = "%Y%m%d"
    """Format string for the dates/datetimes in the ministack filenames."""
    output_folder: Path = Path("")
    """Folder/location where ministack will write outputs to."""
    reference_idx: int = 0
    """Index of the SLC to use as reference during phase linking"""

    def __post_init__(self):
        super().__init__()
        if len(self.file_list) == 0:
            raise ValueError("Cannot create empty ministack")
        elif len(self.file_list) == 1:
            warnings.warn("Creating ministack with only one SLC")
        if not len(self.file_list) == len(self.is_compressed):
            lengths = f"{len(self.file_list)} and {len(self.is_compressed)}"
            raise ValueError(
                f"file_list and is_compressed must be the same length: Got {lengths}"
            )
        if not len(self.dates) == len(self.file_list):
            lengths = f"{len(self.dates)} and {len(self.file_list)}"
            raise ValueError(
                f"dates and file_list must be the same length. Got {lengths}"
            )

        if isinstance(self.dates[0], (date, datetime)):
            date_tuples = [(d,) for d in self.dates]
            # self.dates = date_tuples
            # Workaround for `frozen=True` dataclass
            object.__setattr__(self, "dates", date_tuples)

        if self.reference_date == _DUMMY_DATE:
            ref_date = self.dates[self.reference_idx][0]
            logger.debug("No reference date provided, using first date: %s", ref_date)
            object.__setattr__(self, "reference_date", ref_date)

    @property
    def full_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Full date range of all SLCs in the ministack."""
        return (self.reference_date, self.dates[-1][-1])

    @property
    def full_date_range_str(self) -> str:
        """Full date range of the ministack as a string, e.g. '20210101_20210202'.

        Includes both compressed + normal SLCs in the range.
        """
        return dolphin._dates._format_date_pair(
            *self.full_date_range, fmt=self.file_date_fmt
        )

    @property
    def first_real_slc_idx(self) -> int:
        """Index of the first real SLC in the ministack."""
        try:
            return np.where(~np.array(self.is_compressed))[0][0]
        except IndexError:
            raise ValueError("No real SLCs in ministack")

    @property
    def real_slc_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return (self.dates[self.first_real_slc_idx][0], self.dates[-1][-1])

    @property
    def real_slc_date_range_str(self) -> str:
        """Date range of the real SLCs in the ministack."""
        return dolphin._dates._format_date_pair(
            *self.real_slc_date_range, fmt=self.file_date_fmt
        )

    @property
    def compressed_slc_file_list(self) -> list[Filename]:
        """List of compressed SLCs for this ministack."""
        return [f for f, is_comp in zip(self.file_list, self.is_compressed) if is_comp]

    @property
    def real_slc_file_list(self) -> list[Filename]:
        """List of real SLCs for this ministack."""
        return [
            f for f, is_comp in zip(self.file_list, self.is_compressed) if not is_comp
        ]

    def get_date_str_list(self) -> list[str]:
        """Get a formated string for each date/date tuple in the ministack."""
        date_strs: list[str] = []
        for d in self.dates:
            if len(d) == 1:
                # normal SLC files will have a single date
                s = d[0].strftime(self.file_date_fmt)
            else:
                # Compressed SLCs will have 2 dates in the name marking the start and end
                s = dolphin._dates._format_date_pair(d[0], d[1], fmt=self.file_date_fmt)
            date_strs.append(s)
        return date_strs

    def __rich_repr__(self):
        yield "file_list", self.file_list
        yield "dates", self.dates
        yield "is_compressed", self.is_compressed
        yield "reference_date", self.reference_date
        yield "file_date_fmt", self.file_date_fmt
        yield "output_folder", self.output_folder
        yield "reference_idx", self.reference_idx


@dataclass(frozen=True)
class CompressedSlcInfo:
    """Class for holding attributes about one compressed SLC."""

    real_slc_file_list: Sequence[Filename]
    real_slc_dates: Sequence[Sequence[DateOrDatetime]]
    compressed_slc_file_list: Sequence[Filename]
    reference_date: DateOrDatetime
    file_date_fmt: str
    output_folder: Path = Path("")

    def __post_init__(self):
        assert len(self.real_slc_file_list) == len(self.real_slc_dates)

    @property
    def real_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return (self.real_slc_dates[0][0], self.real_slc_dates[-1][-1])

    @property
    def filename(self) -> str:
        """The filename of the compressed SLC for this ministack."""
        date_str = dolphin._dates._format_date_pair(
            *self.real_date_range, fmt=self.file_date_fmt
        )
        name = f"compressed_{date_str}.tif"
        return name

    @property
    def path(self) -> Path:
        """The path of the compressed SLC for this ministack."""
        return self.output_folder / self.filename

    @property
    def metadata(self) -> dict[str, str]:
        """Prepare metadata about the compressed SLCs to be stored in the output file.

        The filenames will be stored in GDAL metadata as a list of strings, separated
        by commas, saved into the "DOLPHIN" domain.
        """
        real_names = [Path(p).name for p in self.real_slc_file_list]
        comp_names = [Path(p).name for p in self.compressed_slc_file_list]
        return {
            "input_real_slc_files": ",".join(map(str, real_names)),
            "input_compressed_slc_files": ",".join(map(str, comp_names)),
            "reference_date": self.reference_date.strftime(self.file_date_fmt),
        }

    def __fspath__(self):
        return fspath(self.path)


@dataclass(frozen=True)
class MiniStackInfo(BaseStack):
    """Class for holding attributes about one mini-stack of SLCs.

    Used for planning the processing of a batch of SLCs.
    """

    def get_compressed_slc_info(self) -> CompressedSlcInfo:
        """Get the compressed SLC which will come from this ministack.

        Excludes the existing compressed SLCs during the compression.
        """
        # is_comp_arr = np.array(self.is_compressed)
        # real_slc_idxs = np.where(~is_comp_arr)[0]
        # real_slc_files = np.array(self.file_list)[real_slc_idxs].tolist()
        # real_slc_dates = np.array(self.dates)[real_slc_idxs].tolist()

        # comp_slc_idxs = np.where(is_comp_arr)[0]
        # comp_slc_files = np.array(self.file_list)[comp_slc_idxs].tolist()

        # The above doesn't work when theres some 2-tuples, some one
        # just in python only:

        real_slc_files: list[Filename] = []
        real_slc_dates: list[Sequence[DateOrDatetime]] = []
        comp_slc_files: list[Filename] = []
        for f, d, is_comp in zip(self.file_list, self.dates, self.is_compressed):
            if is_comp:
                comp_slc_files.append(f)
            else:
                real_slc_files.append(f)
                real_slc_dates.append(d)

        return CompressedSlcInfo(
            real_slc_file_list=real_slc_files,
            real_slc_dates=real_slc_dates,
            compressed_slc_file_list=comp_slc_files,
            reference_date=self.reference_date,
            file_date_fmt=self.file_date_fmt,
            output_folder=self.output_folder,
        )


@dataclass(frozen=True)
class MiniStackPlanner(BaseStack):
    """Class for planning the processing of batches of SLCs."""

    max_num_compressed: int = 5

    def __post_init__(self):
        super().__post_init__()
        # Default the reference date for a full stack to the first date
        if self.reference_date is None:
            object.__setattr__(
                self, "reference_date", self.dates[self.reference_idx][0]
            )

    def plan(self, ministack_size: int) -> list[MiniStackInfo]:
        """Create a list of ministacks to be processed."""
        if ministack_size < 2:
            raise ValueError("Cannot create ministacks with size < 2")

        output_ministacks: list[MiniStackInfo] = []
        compressed_slc_infos: list[CompressedSlcInfo] = []

        # Solve each ministack using the current chunk (and the previous compressed SLCs)
        ministack_starts = range(0, len(self.file_list), ministack_size)

        for full_stack_idx in ministack_starts:
            cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
            cur_files = list(self.file_list[cur_slice]).copy()
            cur_dates = list(self.dates[cur_slice]).copy()

            comp_slc_files = [c.path for c in compressed_slc_infos]
            # Add the existing compressed SLC files to the start, but
            # limit the num comp slcs `max_num_compressed`
            cur_comp_slc_files = comp_slc_files[-self.max_num_compressed :]
            combined_files = cur_comp_slc_files + cur_files

            combined_dates = [
                c.real_date_range for c in compressed_slc_infos
            ] + cur_dates

            num_ccslc = len(cur_comp_slc_files)
            combined_is_compressed = num_ccslc * [True] + list(
                self.is_compressed[cur_slice]
            )
            # If there are any compressed SLCs, set the reference to the last one
            try:
                last_compressed_idx = np.where(combined_is_compressed)[0]
                reference_idx = last_compressed_idx[-1]
            except IndexError:
                reference_idx = 0

            # Make the current ministack output folder using the start/end dates
            new_date_str = dolphin._dates._format_date_pair(
                cur_dates[0][0], cur_dates[-1][-1], fmt=self.file_date_fmt
            )
            cur_output_folder = self.output_folder / new_date_str
            cur_ministack = MiniStackInfo(
                combined_files,
                dates=combined_dates,
                is_compressed=combined_is_compressed,
                reference_idx=reference_idx,
                output_folder=cur_output_folder,
                reference_date=self.reference_date,
                # TODO: we'll need to alter logic here if we dont fix
                # reference_idx=0, since this will change the reference date
            )

            output_ministacks.append(cur_ministack)
            cur_comp_slc = cur_ministack.get_compressed_slc_info()
            compressed_slc_infos.append(cur_comp_slc)

        return output_ministacks
        # return output_ministacks, compressed_slc_infos
