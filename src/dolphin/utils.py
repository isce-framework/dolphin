import datetime
import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

Pathlike = Union[PathLike[str], str]

from dolphin.log import get_log

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


def save_arr_like(*, arr, like_filename, output_name, driver="GTiff"):
    """Save an array to a file, copying projection/nodata from `like_filename`."""
    from osgeo import gdal

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    nbands = arr.shape[0]
    gdal.UseExceptions()
    ds = gdal.Open(fspath(like_filename))
    if driver is None:
        driver = ds.GetDriver().ShortName
    drv = gdal.GetDriverByName(driver)
    out_ds = drv.Create(
        fspath(output_name),
        ds.RasterXSize,
        ds.RasterYSize,
        nbands,
        numpy_to_gdal_type(arr.dtype),
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    for i in range(nbands):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[i])
    # TODO: copy other metadata
    ds = out_ds = None


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


def combine_mask_files(
    mask_files: List[Pathlike],
    scratch_dir: Pathlike,
    output_file_name: str = "combined_mask.tif",
    dtype: str = "uint8",
    zero_is_valid: bool = False,
) -> Path:
    """Combine multiple mask files into a single mask file.

    Parameters
    ----------
    mask_files : list of Path or str
        List of mask files to combine.
    scratch_dir : Path or str
        Directory to write output file.
    output_file_name : str
        Name of output file to write into `scratch_dir`
    dtype : str, optional
        Data type of output file. Default is uint8.
    zero_is_valid : bool, optional
        If True, zeros mark the valid pixels (like numpy's masking convention).
        Default is False (matches ISCE convention).

    Returns
    -------
    output_file : Path
    """
    from osgeo import gdal

    gdal.UseExceptions()
    output_file = Path(scratch_dir) / output_file_name

    ds = gdal.Open(fspath(mask_files[0]))
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()

    if projection is None and geotransform is None:
        logger.warning("No projection or geotransform found on file %s", mask_files[0])

    nodata = 1 if zero_is_valid else 0

    # Create output file
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        fspath(output_file),
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        numpy_to_gdal_type(dtype),
    )
    ds_out.SetGeoTransform(geotransform)
    ds_out.SetProjection(projection)
    ds_out.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None

    # Loop through mask files and update the total mask (starts with all valid)
    mask_total = np.ones((ds.RasterYSize, ds.RasterXSize), dtype=bool)
    for mask_file in mask_files:
        ds_input = gdal.Open(fspath(mask_file))
        mask = ds_input.GetRasterBand(1).ReadAsArray().astype(bool)
        if zero_is_valid:
            mask = ~mask
        mask_total = np.logical_and(mask_total, mask)
        ds_input = None

    if zero_is_valid:
        mask_total = ~mask_total
    ds_out.GetRasterBand(1).WriteArray(mask_total.astype(dtype))
    ds_out = None

    return output_file


def get_raster_xysize(filename: Pathlike) -> Tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize
