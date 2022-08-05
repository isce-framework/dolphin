#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Optional, Tuple

from osgeo import gdal

from atlas import utils
from atlas.log import get_log

SENTINEL_WAVELENGTH = 0.05546576

logger = get_log()


def create_stack(
    file_list: list,
    subset_bbox: Optional[Tuple[int, int, int, int]] = None,
    target_extent: Optional[Tuple[float, float, float, float]] = None,
    outfile: str = "slcs_base.vrt",
    use_abs_path: bool = True,
):
    """Create a VRT stack from a list of SLC files.

    Parameters
    ----------
    file_list : list
        Names of files to stack
    subset_bbox : tuple[int], optional
        Desired bounding box (in pixels) of subset as (left, bottom, right, top)
    target_extent : tuple[int], optional
        Target extent: alternative way to subset the stack like the `-te` gdal option:
            (xmin, ymin, xmax, ymax) in units of the SLCs' SRS (e.g. UTM coordinates)
    outfile : str, optional (default = "slcs_base.vrt")
        Name of output file to write
    use_abs_path : bool, optional (default = True)
        Write the filepaths in the VRT as absolute
    """
    if subset_bbox is not None and target_extent is not None:
        raise ValueError("Cannot specify both subset_bbox and target_extent")

    if use_abs_path:
        file_list = [str(Path(f).absolute()) for f in file_list]

    ds = gdal.Open(file_list[0])
    if subset_bbox is not None:
        target_extent = _bbox_to_te(subset_bbox, ds=ds)
    ds = None

    options = gdal.BuildVRTOptions(separate=True, outputBounds=target_extent)
    gdal.BuildVRT(outfile, file_list, options=options)

    # Get the list of files (the first will be the VRT name `outfile`)
    file_list = gdal.Info(outfile, format="json")["files"][1:]
    ds = gdal.Open(outfile, gdal.GA_Update)
    for idx, filename in enumerate(file_list, start=1):
        date = utils.get_dates(filename)[0]
        bnd = ds.GetRasterBand(idx)
        # Set the metadata in the SLC domain
        metadata = {
            "Date": date,
            "Wavelength": SENTINEL_WAVELENGTH,
            "AcquisitionTime": date,
        }
        bnd.SetMetadata(metadata, "slc")
        bnd = None


def _bbox_to_te(subset_bbox, ds=None, filename=None):
    """Convert pixel bounding box to target extent box, in georeferenced coordinates."""
    left, bottom, right, top = subset_bbox  # in pixels
    xmin, ymin = _rowcol_to_xy(bottom, left, ds=ds, filename=filename)
    xmax, ymax = _rowcol_to_xy(top, right, ds=ds, filename=filename)
    return xmin, ymin, xmax, ymax


def _rowcol_to_xy(row, col, ds=None, filename=None):
    """Convert a row and column index to coordinates in the georeferenced space.

    Reference: https://gdal.org/tutorials/geotransforms_tut.html
    """
    if ds is None:
        ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    x = gt[0] + col * gt[1] + row * gt[2]
    y = gt[3] + col * gt[4] + row * gt[5]
    return x, y


def get_cli_args():
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert SLC stack to single VRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in-files",
        nargs="*",
        help="Names of GDAL-readable SLC files to include in stack.",
    )
    parser.add_argument(
        "--in-textfile",
        help=(
            "Newline-delimited text file listing locations of SLC files"
            "Alternative to --in-files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="stack",
        help="Directory where the vrt stack will be stored",
    )
    parser.add_argument(
        "--out-vrt-name",
        default="slcs_base.vrt",
        help="Name of output SLC containing all images",
    )
    parser.add_argument(
        "-b",
        "--subset-bbox",
        type=int,
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        default=None,
        help="Bounding box (in pixels) to subset the stack. None = no subset",
    )
    parser.add_argument(
        "-te",
        "--target-extent",
        type=float,
        nargs=4,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        default=None,
        help=(
            "Target extent (like GDAL's `-te` option) in units of the SLC's SRS"
            " (i.e., in UTM coordinates). An alternative way to subset the stack."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    """Run the command line interface."""
    args = get_cli_args()

    # Get slc list from text file or command line
    if args.in_files is not None:
        file_list = sorted(args.in_files)
    elif args.in_textfile is not None:
        with open(args.in_textfile) as f:
            file_list = sorted(f.read().splitlines())
    else:
        raise ValueError("Need to pass either --in-files or --in-textfile")

    num_slc = len(file_list)
    logger.info(f"Number of SLCs found: {num_slc}")

    # Set up single stack file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outfile = str(out_dir / args.out_vrt_name)
    create_stack(
        file_list,
        outfile=outfile,
        subset_bbox=args.subset_bbox,
        target_extent=args.target_extent,
    )


if __name__ == "__main__":
    main()
