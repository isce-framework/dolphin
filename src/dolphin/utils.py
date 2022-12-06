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


def get_dates(filename: Filename) -> List[Union[None, str]]:
    """Search for dates (YYYYMMDD) in `filename`, excluding path."""
    date_list = re.findall(r"\d{4}\d{2}\d{2}", Path(filename).stem)
    if not date_list:
        msg = f"{filename} does not contain date as YYYYMMDD"
        logger.warning(msg)
        # raise ValueError(msg)
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
        d_str = get_dates(slc_str)
        if not d_str:
            raise ValueError(f"Could not find date of format {fmt} in {slc_str}")
            # return None
        return _parse(d_str[0], fmt=fmt)
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s, fmt=fmt) for s in slc_str if s]
