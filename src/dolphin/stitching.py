"""stitching.py: utilities for combining interferograms into larger images."""
import datetime
import itertools
import os
from os import fspath
from typing import List, Tuple

from osgeo import gdal

from dolphin import utils
from dolphin._types import Filename
from dolphin.io import get_raster_bounds


def merge_by_date(
    image_file_list: List[Filename],
    output_path: Filename = ".",
    dry_run: bool = False,
):
    """Group images from the same date and merge into one image per date.

    Parameters
    ----------
    image_file_list: List[Filename]
        list of paths to images.
    output_path: Filename
        path to output directory
    dry_run: bool
        if True, do not actually stitch the images, just print.

    Returns
    -------
    str:
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
    dict:
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
    dry_run: bool = False,
):
    """Combine multiple SLC images on the same date into one image.

    Parameters
    ----------
    file_list: List[Filename]
        list of raster filenames
    date: datetime.date
        date of the images
    output_path: Filename
        path to output directory
    dry_run: bool
        if True, do not actually stitch the images, just print.

    Returns
    -------
    str:
        path to the stitched SLC file
    """
    # TODO: what format? How to initialize the file?
    new_name = "{}.h5".format(date.strftime("%Y%m%d"))
    new_name = os.path.join(output_path, new_name)
    if dry_run:
        return new_name

    if len(file_list) == 1:
        print("Only one image, no stitching needed")
        return new_name

    # TODO
    # bounds, gt = get_combined_bounds_gt(*file_list)
    return new_name


def get_combined_bounds_gt(
    *filenames: Filename,
) -> Tuple[Tuple[float, float, float, float], List]:
    """Get the bounds and geotransform of the combined image.

    Parameters
    ----------
    filenames: List[Filename]
        list of filenames to combine

    Returns
    -------
    bounds: Tuple[float]
        (min_x, min_y, max_x, max_y)
    gt: List[float]
        geotransform of the combined image.
    """
    # scan input files
    xs = []
    ys = []
    projs = set()
    resolutions = set()
    for fn in filenames:
        ds = gdal.Open(fspath(fn))
        left, bottom, right, top = get_raster_bounds(fn)
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
    gt_total = [bounds[0], dx, 0, bounds[3], 0, dy]
    return bounds, gt_total
