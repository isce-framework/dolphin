"""stitching.py: utilities for combining interferograms into larger images."""
import datetime
import itertools
import math
from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike
from osgeo import gdal

from dolphin import io, utils
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log()


def merge_by_date(
    image_file_list: List[Filename],
    file_date_fmt: str = io.DEFAULT_DATETIME_FORMAT,
    output_dir: Filename = ".",
    driver: str = "ENVI",
):
    """Group images from the same date and merge into one image per date.

    Parameters
    ----------
    image_file_list : List[Filename]
        list of paths to images.
    file_date_fmt : Optional[str]
        format of the date in the filename. Default is %Y%m%d
    output_dir : Filename
        path to output directory
    driver : str
        GDAL driver to use for output. Default is ENVI.

    Returns
    -------
    dict
        key is the date of the SLC acquisition
        Value is the path to the stitched image

    Notes
    -----
    This function is intended to be used with filenames that contain date pairs
    (from interferograms).
    """
    grouped_images = group_images_by_date(image_file_list, file_date_fmt=file_date_fmt)
    stitched_acq_times = {}

    for date, cur_images in grouped_images.items():
        logger.info(f"Stitching {len(cur_images)} images from {date} into one image")
        stitched_name = _stitch_same_date(
            cur_images,
            date,
            output_dir=output_dir,
            driver=driver,
        )

        stitched_acq_times[date] = stitched_name

    return stitched_acq_times


def group_images_by_date(
    image_file_list: List[Filename], file_date_fmt: str = io.DEFAULT_DATETIME_FORMAT
):
    """Combine Sentinel objects by date.

    Parameters
    ----------
    image_file_list: List[Filename]
        path to folder containing CSLC files
    file_date_fmt: str
        format of the date in the filename.
        Default is [dolphin.io.DEFAULT_DATETIME_FORMAT][]

    Returns
    -------
    dict
        key is the date of the SLC acquisition
        Value is a list of Paths on that date:
        [(datetime.date(2017, 10, 13),
          [Path(...)
            Path(...),
            ...]),
         (datetime.date(2017, 10, 25),
          [Path(...)
            Path(...),
            ...]),
    """
    sorted_file_list, _ = utils.sort_files_by_date(
        image_file_list, file_date_fmt=file_date_fmt
    )

    # Now collapse into groups, sorted by the date
    grouped_images = {
        dates: list(g)
        for dates, g in itertools.groupby(
            sorted_file_list, key=lambda x: tuple(utils.get_dates(x))
        )
    }
    return grouped_images


def _stitch_same_date(
    file_list: List[Filename],
    dates: Tuple[datetime.date, datetime.date],
    output_dir: Filename,
    target_aligned_pixels: bool = False,
    driver: str = "ENVI",
    nodata: Optional[float] = 0,
    out_dtype: Optional[DTypeLike] = None,
) -> Path:
    """Combine multiple SLC images on the same date into one image.

    Parameters
    ----------
    file_list : List[Filename]
        list of raster filenames
    dates : Tuple[datetime.date]
        date(s) of the images
    output_dir : Filename
        path to output directory
    target_aligned_pixels: bool
        if True, adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.
        Default is False.
    driver : str
        GDAL driver to use for output file. Default is ENVI.
    nodata : Optional[float]
        nodata value to use for output file. Default is 0.
    out_dtype : Optional[DTypeLike]
        output data type. Default is None, which will use the data type
        of the first image in the list.

    Returns
    -------
    Path
        path to the stitched SLC file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    new_name = Path(output_dir) / (io._format_date_pair(*dates) + ".int")

    if len(file_list) == 1:
        logger.info("Only one image, no stitching needed")
        logger.info(f"Copying {file_list[0]} to {new_name} and zeroing nodata values.")
        _nodata_to_zero(
            file_list[0],
            outfile=new_name,
            driver=driver,
        )
        return new_name

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    bounds, gt = _get_combined_bounds_gt(
        *file_list, target_aligned_pixels=target_aligned_pixels
    )
    # Get the (dx, dy) resolution from the geotransform
    res = (abs(gt[1]), abs(gt[5]))  # dy is be negative for north-up images

    out_shape = _get_output_shape(bounds, res)
    projection = _get_mode_projection(file_list)
    out_dtype = out_dtype or io.get_dtype(file_list[0])

    io.write_arr(
        arr=None,
        output_name=new_name,
        driver=driver,
        nbands=1,
        shape=out_shape,
        dtype=out_dtype,
        nodata=nodata,
        geotransform=gt,
        projection=projection,
    )

    out_left, out_bottom, out_right, out_top = bounds
    # Now loop through the files and write them to the output
    for f in file_list:
        logger.info(f"Stitching {f} into {new_name}")
        ds_in = gdal.Open(f)
        proj_in = ds_in.GetProjection()
        if proj_in != projection:
            logger.info(
                f"Reprojecting {f} from {proj_in} to match mode projection {projection}"
            )
            ds_in = _get_warped_ds(f, projection, res)
        else:
            ds_in = gdal.Open(f)
        in_left, in_bottom, in_right, in_top = io.get_raster_bounds(ds=ds_in)

        # Get the spatial intersection of input and output
        int_right = min(in_right, out_right)
        int_top = min(in_top, out_top)
        int_left = max(in_left, out_left)
        int_bottom = max(in_bottom, out_bottom)

        # Get the pixel coordinates of the intersection in the input
        row_top, col_right = io.xy_to_rowcol(int_right, int_top, ds=ds_in)
        row_bottom, col_left = io.xy_to_rowcol(int_left, int_bottom, ds=ds_in)
        in_rows, in_cols = ds_in.RasterYSize, ds_in.RasterXSize
        # Read the input data in this window
        arr_in = ds_in.ReadAsArray(
            col_left,
            row_top,
            # Clip the width/height to the raster size
            min(col_right - col_left, in_cols),
            min(row_bottom - row_top, in_rows),
        )
        # TODO: handle nodata and nans

        # Get pixel coordinates of the intersection in the output
        row_top, col_right = io.xy_to_rowcol(int_right, int_top, filename=new_name)
        row_bottom, col_left = io.xy_to_rowcol(int_left, int_bottom, filename=new_name)
        # Write the input data to the output in this window
        io.write_block(
            arr_in,
            filename=new_name,
            row_start=row_top,
            col_start=col_left,
        )

    return Path(new_name)


def _get_warped_ds(input: Filename, projection: str, res: Tuple[float, float]):
    """Get an in-memory warped VRT of the input file.

    Parameters
    ----------
    input : Filename
        Name of the input file.
    projection : str
        The desired projection, as a WKT string or 'EPSG:XXXX' string.
    res : Tuple[float, float]
        The desired [x, y] resolution.

    Returns
    -------
    gdal.Dataset
        The result of gdal.Warp with VRT output format.
    """
    return gdal.Warp(
        "",
        input,
        format="VRT",
        dstSRS=projection,
        targetAlignedPixels=True,
        xRes=res[0],
        yRes=res[1],
    )


def _gdal_merge(*inputs: Filename, output: Filename):
    import subprocess

    cmd = ["gdal_merge.py", "-init", 0, "-n", 0, "-o", output, *inputs]
    subprocess.run(cmd, check=True)  # type: ignore


def _get_combined_bounds_gt(
    *filenames: Filename,
    target_aligned_pixels: bool = False,
) -> Tuple[Tuple[float, float, float, float], List]:
    """Get the bounds and geotransform of the combined image.

    Parameters
    ----------
    filenames : List[Filename]
        list of filenames to combine
    target_aligned_pixels : bool
        if True, adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``.

    Returns
    -------
    bounds : Tuple[float]
        (min_x, min_y, max_x, max_y)
    gt : List[float]
        geotransform of the combined image.
    """
    # scan input files
    xs = []
    ys = []
    projs = set()
    resolutions = set()
    for fn in filenames:
        ds = gdal.Open(fspath(fn))
        left, bottom, right, top = io.get_raster_bounds(fn)
        gt = ds.GetGeoTransform()
        dx, dy = gt[1], gt[5]
        resolutions.add((abs(dx), abs(dy)))  # dy is negative for north-up
        projs.add(ds.GetProjection())

        xs.extend([left, right])
        ys.extend([bottom, top])
    if len(projs) > 1:
        raise ValueError(f"Input files have different projections {projs}")
    if len(resolutions) > 1:
        raise ValueError(f"Input files have different resolutions: {resolutions}")

    bounds = min(xs), min(ys), max(xs), max(ys)
    if target_aligned_pixels:
        bounds = _align_bounds(bounds, resolutions.pop())

    gt_total = [bounds[0], dx, 0, bounds[3], 0, dy]
    return bounds, gt_total


def _get_output_shape(bounds, res):
    """Get the output shape of the combined image."""
    left, bottom, right, top = bounds
    out_width = int(round((right - left) / abs(res[0])))
    out_height = int(round((top - bottom) / abs(res[1])))
    return (out_height, out_width)


def _get_mode_projection(filenames: List[Filename]) -> str:
    """Get the most common projection in the list."""
    projs = [gdal.Open(fspath(fn)).GetProjection() for fn in filenames]
    return max(set(projs), key=projs.count)


def _align_bounds(bounds, res):
    left, bottom, right, top = bounds
    left = math.floor(left / res[0]) * res[0]
    right = math.ceil(right / res[0]) * res[0]
    bottom = math.floor(bottom / res[1]) * res[1]
    top = math.ceil(top / res[1]) * res[1]
    return (left, bottom, right, top)


def _nodata_to_zero(
    infile,
    outfile: Optional[Filename] = None,
    ext: Optional[str] = None,
    in_band: int = 1,
    driver="ENVI",
    creation_options=["SUFFIX=ADD"],
):
    """Make a copy of infile and replace NaNs with 0."""
    in_p = Path(infile)
    if outfile is None:
        if ext is None:
            ext = in_p.suffix
        out_dir: Path = in_p.parent if out_dir is None else Path(out_dir)
        outfile = out_dir / (in_p.stem + "_tmp" + ext)

    ds_in = gdal.Open(fspath(infile))
    drv = gdal.GetDriverByName(driver)
    ds_out = drv.CreateCopy(fspath(outfile), ds_in, options=creation_options)

    bnd = ds_in.GetRasterBand(in_band)
    nodata = bnd.GetNoDataValue()
    arr = bnd.ReadAsArray()
    # also make sure to replace NaNs, even if nodata is not set
    mask = np.logical_or(np.isnan(arr), arr == nodata)
    arr[mask] = 0

    ds_out.GetRasterBand(1).WriteArray(arr)
    ds_out = None

    return outfile
