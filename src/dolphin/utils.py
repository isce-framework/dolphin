import datetime
import re
from os import PathLike
from pathlib import Path
from typing import List, Union

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from dolphin._log import get_log

Filename = Union[str, PathLike[str]]
gdal.UseExceptions()
logger = get_log()


def numpy_to_gdal_type(np_dtype):
    """Convert numpy dtype to gdal type."""
    # Wrap in np.dtype in case string is passed
    if isinstance(np_dtype, str):
        np_dtype = np.dtype(np_dtype.lower())
    elif isinstance(np_dtype, type):
        np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    return gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)


def gdal_to_numpy_type(gdal_type):
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type)


def get_dates(filename: Filename, fmt="%Y%m%d") -> List[Union[None, str]]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : str or PathLike
        Filename to search for dates.
    fmt : str, optional
        Format of date to search for. Default is "%Y%m%d".

    Returns
    -------
    list[str] or None
        List of dates found in the stem of `filename` matching `fmt`.
        Returns None if nothing is found.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    ['20191231']
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    ['20191231', '20191231']
    >>> get_dates("/not/a/date_named_file.tif")
    []
    """  # noqa: E501
    pat = _date_format_to_regex(fmt)
    date_list = re.findall(pat, Path(filename).stem)
    if not date_list:
        msg = f"{filename} does not contain date as YYYYMMDD"
        logger.warning(msg)
        return []
    return date_list


def parse_slc_strings(slc_str: Union[Filename, List[Filename]], fmt="%Y%m%d"):
    """Parse a string, or list of strings, matching `fmt` into datetime.date.

    Parameters
    ----------
    slc_str : str or list of str
        String or list of strings to parse.
    fmt : str, optional
        Format of string to parse. Default is "%Y%m%d".

    Returns
    -------
    datetime.date, or list of datetime.date
    """

    def _parse(datestr, fmt="%Y%m%d") -> datetime.date:
        return datetime.datetime.strptime(datestr, fmt).date()

    # The re.search will find YYYYMMDD anywhere in string
    if isinstance(slc_str, str) or hasattr(slc_str, "__fspath__"):
        d_list = get_dates(slc_str, fmt=fmt)
        if not d_list:
            raise ValueError(f"Could not find date of format {fmt} in {slc_str}")
        return _parse(d_list[0], fmt=fmt)
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s, fmt=fmt) for s in slc_str if s]


def _date_format_to_regex(date_format):
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
