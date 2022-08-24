import datetime
import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Union

import numpy as np

Pathlike = Union[PathLike[str], str]

from atlas.log import get_log

logger = get_log()


def get_dates(filename: Pathlike) -> List[Union[None, str]]:
    """Search for dates (YYYYMMDD) in `filename`, excluding path."""
    date_list = re.findall(r"\d{4}\d{2}\d{2}", Path(filename).stem)
    if not date_list:
        raise ValueError(f"{filename} does not contain date as YYYYMMDD")
    return date_list


def copy_projection(src_file: Pathlike, dst_file: Pathlike) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    from osgeo import gdal

    gdal.UseExceptions()

    ds_src = gdal.Open(fspath(src_file))
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        logger.info("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def numpy_to_gdal_type(np_dtype):
    """Convert numpy dtype to gdal type."""
    from osgeo import gdal_array, gdalconst

    # Wrap in np.dtype in case string is passed
    np_dtype = np.dtype(str(np_dtype).lower())
    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    return gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)


def parse_slc_strings(slc_str):
    """Parse a string, or list of strings, with YYYYmmdd as date."""
    # The re.search will find YYYYMMDD anywhere in string
    if isinstance(slc_str, str):
        match = re.search(r"\d{8}", slc_str)
        if not match:
            raise ValueError(f"{slc_str} does not contain date as YYYYMMDD")
        return _parse(match.group())
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s) for s in slc_str if s]


def _parse(datestr):
    return datetime.datetime.strptime(datestr, "%Y%m%d").date()
