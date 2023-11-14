from __future__ import annotations

import datetime
import itertools
import re
from typing import Iterable, Pattern, overload

from ._types import PathLikeT, PathOrStr
from .utils import _get_path_from_gdal_str

DateOrDatetime = datetime.datetime | datetime.date
DEFAULT_DATETIME_FORMAT = "%Y%m%d"

__all__ = [
    "DEFAULT_DATETIME_FORMAT",
    "get_dates",
    "sort_files_by_date",
    "group_by_date",
]


def get_dates(
    filename: PathOrStr,
    fmt: str = DEFAULT_DATETIME_FORMAT,
) -> list[datetime.datetime]:
    """Search for dates/datetimes in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : str or PathLike
        PathOrStr to search for dates.
    fmt : str, optional
        Format of date to search for. Default is "%Y%m%d".

    Returns
    -------
    list[datetime.datetime]
        list of datetimes found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    (datetime.datetime(2019, 12, 31, 0, 0, 0),)
    >>> get_dates("ifg_20190101_20200101.tif")
    (datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.date(2019, 12, 31, 0, 0, 0), datetime.date(2019, 12, 31, 0, 0, 0)]
    >>> get_dates("/not/a/date_named_file.tif")
    []
    """  # noqa: E501
    path = _get_path_from_gdal_str(filename)
    pattern = _date_format_to_regex(fmt)
    date_list = re.findall(pattern, path.stem)
    if not date_list:
        return []
    return list(_parse_datetime(d, fmt) for d in date_list)


def _parse_datetime(datestr: str, fmt: str = "%Y%m%d") -> datetime.datetime:
    return datetime.datetime.strptime(datestr, fmt)


def _date_format_to_regex(date_format) -> Pattern:
    r"""Convert a python date format string to a regular expression.

    Useful for Year, month, date date formats.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. "%Y%m%d"

    Returns
    -------
    re.Pattern
        Regular expression that matches the date format string.

    Examples
    --------
    >>> pat2 = _date_format_to_regex("%Y%m%d").pattern
    >>> pat2 == re.compile(r'\d{4}\d{2}\d{2}').pattern
    True
    >>> pat = _date_format_to_regex("%Y-%m-%d").pattern
    >>> pat == re.compile(r'\d{4}\-\d{2}\-\d{2}').pattern
    True
    """
    # Escape any special characters in the date format string
    date_format = re.escape(date_format)

    # Replace each format specifier with a regular expression that matches it
    date_format = date_format.replace("%Y", r"\d{4}")
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")

    # Return the resulting regular expression
    return re.compile(date_format)


@overload
def sort_files_by_date(
    files: Iterable[str], file_date_fmt: str = DEFAULT_DATETIME_FORMAT
) -> tuple[list[str], list[list[datetime.datetime]]]:
    ...


@overload
def sort_files_by_date(
    files: Iterable[PathLikeT], file_date_fmt: str = DEFAULT_DATETIME_FORMAT
) -> tuple[list[PathLikeT], list[list[datetime.datetime]]]:
    ...


def sort_files_by_date(files, file_date_fmt=DEFAULT_DATETIME_FORMAT):
    """Sort a list of files by date.

    If some files have multiple dates, the files with the most dates are sorted
    first. Within each group of files with the same number of dates, the files
    with the earliest dates are sorted first.

    The multi-date files are placed first so that compressed SLCs are sorted
    before the individual SLCs that make them up.

    Parameters
    ----------
    files : Iterable[str or PathLike]
        list of files to sort.
    file_date_fmt : str, optional
        Datetime format passed to `strptime`, by default "%Y%m%d"

    Returns
    -------
    file_list : list[str] or list[PathLike]
        list of files sorted by date.
    dates : list[list[datetime.date,...]]
        Sorted list, where each entry has all the dates from the corresponding file.
    """

    def sort_key(file_date_tuple):
        # Key for sorting:
        # To sort the files with the most dates first (the compressed SLCs which
        # span a date range), sort the longer date lists first.
        # Then, within each group of dates of the same length, use the date/dates
        _, dates = file_date_tuple
        try:
            return (-len(dates), dates)
        except TypeError:
            return (-1, dates)

    file_date_tuples = [(f, get_dates(f, fmt=file_date_fmt)) for f in files]
    file_dates = sorted([fd_tuple for fd_tuple in file_date_tuples], key=sort_key)

    # Unpack the sorted pairs with new sorted values
    file_list, dates = zip(*file_dates)  # type: ignore
    return list(file_list), list(dates)


@overload
def group_by_date(
    files: Iterable[str], file_date_fmt: str = DEFAULT_DATETIME_FORMAT
) -> dict[tuple[datetime.date, ...], list[str]]:
    ...


@overload
def group_by_date(
    files: Iterable[PathLikeT], file_date_fmt: str = DEFAULT_DATETIME_FORMAT
) -> dict[tuple[datetime.date, ...], list[PathLikeT]]:
    ...


def group_by_date(files, file_date_fmt=DEFAULT_DATETIME_FORMAT):
    """Combine files by date into a dict.

    Parameters
    ----------
    files: Iterable[Filename]
        Path to folder containing files with dates in the filename.
    file_date_fmt: str
        Format of the date in the filename.
        Default is [dolphin.DEFAULT_DATETIME_FORMAT][]

    Returns
    -------
    dict
        key is a list of dates in the filenames.
        Value is a list of Paths on that date.
        E.g.:
        {(datetime.date(2017, 10, 13),
          [Path(...)
            Path(...),
            ...]),
         (datetime.date(2017, 10, 25),
          [Path(...)
            Path(...),
            ...]),
        }
    """
    sorted_file_list, _ = sort_files_by_date(files, file_date_fmt=file_date_fmt)

    # Now collapse into groups, sorted by the date
    grouped_images = {
        dates: list(g)
        for dates, g in itertools.groupby(
            sorted_file_list, key=lambda x: tuple(get_dates(x))
        )
    }
    return grouped_images
