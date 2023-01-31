import datetime
import re
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from numpy.typing import DTypeLike
from osgeo import gdal, gdal_array, gdalconst

from dolphin._log import get_log
from dolphin._types import Filename

gdal.UseExceptions()
logger = get_log()


def numpy_to_gdal_type(np_dtype: DTypeLike) -> int:
    """Convert numpy dtype to gdal type.

    Parameters
    ----------
    np_dtype : DTypeLike
        Numpy dtype to convert.

    Returns
    -------
    int
        GDAL type code corresponding to `np_dtype`.

    Raises
    ------
    TypeError
        If `np_dtype` is not a numpy dtype, or if the provided dtype is not
        supported by GDAL (for example, `np.dtype('>i4')`)
    """
    np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    gdal_code = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)
    if gdal_code is None:
        raise TypeError(f"dtype {np_dtype} not supported by GDAL.")
    return gdal_code


def gdal_to_numpy_type(gdal_type: Union[str, int]) -> np.dtype:
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type)


def get_dates(filename: Filename, fmt: str = "%Y%m%d") -> List[datetime.date]:
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
    list[datetime.date]
        List of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.date(2019, 12, 31)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.date(2019, 12, 31), datetime.date(2019, 12, 31)]
    >>> get_dates("/not/a/date_named_file.tif")
    []
    """  # noqa: E501
    pat = _date_format_to_regex(fmt)
    date_list = re.findall(pat, Path(filename).stem)
    if not date_list:
        msg = f"{filename} does not contain date like {fmt}"
        logger.warning(msg)
        return []
    return [_parse_date(d, fmt) for d in date_list]


def _parse_date(datestr: str, fmt: str = "%Y%m%d") -> datetime.date:
    return datetime.datetime.strptime(datestr, fmt).date()


def parse_slc_strings(
    slc_str: Union[Filename, Iterable[Filename]],
    fmt: Union[str, Iterable[str], None] = None,
) -> Union[datetime.date, List[datetime.date]]:
    """Parse a string/Path, or list of strings/Paths, matching `fmt` into dates.

    It is expected that the date is in the filename portion of the string
    (the `.name` part of the Path), and that the date is unique in the filename.

    Note that the date may appear more than once, as in Sentinel-1 SLC product names
    containing the acquisition stop/start datetime.

    Parameters
    ----------
    slc_str : str or Iterable[str]
        String or list of strings to parse for dates.
    fmt : str, or List[str], None. Optional
        Format of string to parse.
        If None (default), searches for "%Y%m%d" or "%Y-%m-%d".

    Returns
    -------
    datetime.date, or List[datetime.date]

    Raises
    ------
    ValueError
        If none of the strings in `slc_str` match `fmt`,
        or if multiple dates are found in the filename portion
        of `slc_str` (we expect only one date per SLC file).
    """
    if fmt is None:
        fmt = ["%Y%m%d", "%Y-%m-%d"]
    elif isinstance(fmt, str):
        fmt = [fmt]

    if isinstance(slc_str, str) or hasattr(slc_str, "__fspath__"):
        # Unpack all returned dates from each format
        d_list = []
        for f in fmt:
            d_list.extend(get_dates(slc_str, fmt=f))  # type: ignore
            if len(d_list) > 0:
                break
        else:  # if we iterate through all formats and don't find any dates
            raise ValueError(f"Could not find date of format {fmt} in {slc_str}")

        unique_dates = np.unique(d_list)
        if len(unique_dates) > 1:
            raise ValueError(
                f"Found multiple dates in {slc_str}: {unique_dates}. "
                "Please specify a date format."
            )
        return unique_dates[0]
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s, fmt=fmt) for s in slc_str if s]  # type: ignore


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
