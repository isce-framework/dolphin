"""stitching.py: utilities for combining interferograms into larger images."""
import itertools
import os
from typing import List

from dolphin._parsers import BurstSlc, parse_opera_cslc
from dolphin._types import Filename
from dolphin.io import get_raster_bounds


def merge_by_date(
    image_file_list: List[Filename],
    output_path: Filename = ".",
    dry_run: bool = False,
    verbose: bool = True,
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
    verbose: bool
        if True, print out the names of the images being stitched.

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
            output_path=output_path,
            dry_run=dry_run,
            verbose=verbose,
        )

        # Keep track of the acquisition datetimes for each stitched file
        burst_id_start = cur_images[0].burst_id
        burst_id_end = cur_images[-1].burst_id
        stitched_acq_times[stitched_name] = (date, burst_id_start, burst_id_end)

    return stitched_acq_times


def group_images_by_date(image_file_list: List[Filename]):
    """Combine Sentinel objects by date.

    Parameters
    ----------
    image_file_list: List[Filename]
        path to folder containing CSLC files

    Returns
    -------
    dict:
        key is the date of the SLC acquisition
        Value is a list of BurstSlc objects on that date:
        [(datetime.datetime(2017, 10, 13),
          [BurstSlc(...)
            BurstSlc(...),
            ...]),
         (datetime.datetime(2017, 10, 25),
          [BurstSlc(...)
            BurstSlc(...),
            ...]),
    """
    burst_images = [parse_opera_cslc(f) for f in image_file_list]
    date_sorted_images = sorted(
        burst_images, key=lambda b: (b.datetime, b.burst_id, b.subswath)
    )

    # Now collapse into groups, sorted by the date
    grouped_images = {
        date: list(g)
        for date, g in itertools.groupby(
            date_sorted_images, key=lambda x: x.datetime.date()
        )
    }
    return grouped_images


def _stitch_same_date(
    slc_file_list: List[BurstSlc],
    output_path: Filename,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Combine multiple SLC images on the same date into one image.

    Parameters
    ----------
    slc_file_list: List[Filename]
        list of BurstSlc objects
    output_path: Filename
        path to output directory
    dry_run: bool
        if True, do not actually stitch the images, just print.
    verbose: bool
        if True, print out the names of the images being stitched.

    Returns
    -------
    str:
        path to the stitched SLC file
    """
    if verbose:
        print("Stitching slcs for %s" % slc_file_list[0].datetime.date())
        for g in slc_file_list:
            print("image:", g.filename, g.datetime)

    g = slc_file_list[0]
    # TODO: what format? How to initialize the file?
    new_name = "{}.h5".format(g.datetime.strftime("%Y%m%d"))
    new_name = os.path.join(output_path, new_name)
    if dry_run:
        return new_name

    if len(slc_file_list) == 1:
        print("Only one image, no stitching needed")
        return new_name

    # TODO
    return new_name


def get_combined_bounds(*filenames: Filename):
    """Get the bounds of the combined image.

    Parameters
    ----------
    filenames: List[Filename]
        list of filenames to combine

    Returns
    -------
    tuple:
        (min_x, min_y, max_x, max_y)

    """
    # scan input files
    xs = []
    ys = []
    for fn in filenames:
        left, bottom, right, top = get_raster_bounds(fn)
        xs.extend([left, right])
        ys.extend([bottom, top])
    return min(xs), min(ys), max(xs), max(ys)
