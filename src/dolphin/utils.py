from os import PathLike
from typing import Union

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from dolphin.log import get_log

Pathlike = Union[PathLike[str], str]
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
