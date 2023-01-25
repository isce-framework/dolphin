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
from dolphin._types import Filename


def merge_by_date(
    image_file_list: List[Filename],
    output_path: Filename = ".",
    dry_run: bool = False,
):
    """Group images from the same date and merge into one image per date.

    Parameters
    ----------
    image_file_list : List[Filename]
        list of paths to images.
    output_path : Filename
        path to output directory
    dry_run : bool
        if True, do not actually stitch the images, just print.

    Returns
    -------
    str
        path to the stitched image file
    """
    grouped_images = group_images_by_date(image_file_list)
    stitched_acq_times = {}

    for date, cur_images in grouped_images.items():
        print(f"Stitching {len(cur_images)} images from {date} into one image")
        stitched_name = _stitch_same_date(
            cur_images,
            date,
            output_path=output_path,
            dry_run=dry_run,
        )

        # Keep track of the acquisition datetimes for each stitched file
        burst_id_start = cur_images[0].burst_id
        burst_id_end = cur_images[-1].burst_id
        stitched_acq_times[stitched_name] = (date, burst_id_start, burst_id_end)

    return stitched_acq_times


def group_images_by_date(
    image_file_list: List[Filename], file_date_fmt: str = "%Y%m%d"
):
    """Combine Sentinel objects by date.

    Parameters
    ----------
    image_file_list: List[Filename]
        path to folder containing CSLC files
    file_date_fmt: str
        format of the date in the filename. Default is %Y%m%d

    Returns
    -------
    dict
        key is the date of the SLC acquisition
        Value is a list of Paths on that date:
        [(datetime.datetime(2017, 10, 13),
          [Path(...)
            Path(...),
            ...]),
         (datetime.datetime(2017, 10, 25),
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
    date: datetime.date,
    output_path: Filename,
    target_aligned_pixels: bool = False,
    driver: str = "ENVI",
    nodata: Optional[float] = 0,
    out_dtype: Optional[DTypeLike] = None,
    dry_run: bool = False,
):
    """Combine multiple SLC images on the same date into one image.

    Parameters
    ----------
    file_list : List[Filename]
        list of raster filenames
    date : datetime.date
        date of the images
    output_path : Filename
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
    dry_run : bool
        if True, do not actually stitch the images, just print.

    Returns
    -------
    Path
        path to the stitched SLC file
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    new_name = Path(output_path) / f"{date.strftime('%Y%m%d')}.int"

    if dry_run:
        return new_name

    if len(file_list) == 1:
        print("Only one image, no stitching needed")
        print(f"Copying {file_list[0]} to {new_name} and zeroing nodata values.")
        _nodata_to_zero(
            file_list[0],
            outfile=new_name,
            driver=driver,
        )
        return new_name

    # TODO
    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    bounds, gt = _get_combined_bounds_gt(
        *file_list, target_aligned_pixels=target_aligned_pixels
    )
    # Get the resolution from the geotransform
    res = [gt[1], gt[5]]

    out_shape = _get_output_shape(bounds, res)
    projection = _get_mode_projection(file_list)

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

    # in_nodata = io.get_nodata(file_list[0])

    return Path(new_name)


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
        resolutions.add((dx, dy))
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
    out_width = int(round((right - left) / res[0]))
    out_height = int(round((top - bottom) / res[1]))
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
    mask = np.logical_or(np.isnan(arr), arr == nodata)
    arr[mask] = 0
    ds_out.GetRasterBand(1).WriteArray(arr)
    ds_out = None

    return outfile
