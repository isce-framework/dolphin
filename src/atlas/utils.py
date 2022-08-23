import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Union

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
    ds_dst = gdal.Open(dst_file, gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None
