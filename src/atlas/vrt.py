#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Optional, Tuple

from osgeo import gdal

gdal.UseExceptions()

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
    # Use the first file in the stack to get size, transform info
    ds = gdal.Open(file_list[0])
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)  # Should be CFloat32
    # Save these for setting at the end
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    srs = ds.GetSpatialRef()
    ds = None

    xoff, yoff, xsize_sub, ysize_sub = _get_subset_bbox(
        xsize,
        ysize,
        subset_bbox=subset_bbox,
        target_extent=target_extent,
        filename=file_list[0],
    )

    with open(outfile, "w") as fid:
        fid.write(f'<VRTDataset rasterXSize="{xsize_sub}" rasterYSize="{ysize_sub}">\n')

        for idx, filename in enumerate(file_list, start=1):
            filename = str(Path(filename).absolute())
            date = utils.get_dates(filename)[0]
            outstr = f"""    <VRTRasterBand dataType="{dtype}" band="{idx}">
        <SimpleSource>
            <SourceFilename>{filename}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="{dtype}"/>
            <SrcRect xOff="{xoff}" yOff="{yoff}" xSize="{xsize_sub}" ySize="{ysize_sub}"/>
            <DstRect xOff="0" yOff="0" xSize="{xsize_sub}" ySize="{ysize_sub}"/>
        </SimpleSource>
        <Metadata domain="slc">
            <MDI key="Date">{date}</MDI>
            <MDI key="Wavelength">{SENTINEL_WAVELENGTH}</MDI>
            <MDI key="AcquisitionTime">{date}</MDI>
        </Metadata>
    </VRTRasterBand>\n"""  # noqa: E501
            fid.write(outstr)

        fid.write("</VRTDataset>")

    # Set the georeferencing metadata
    ds = gdal.Open(outfile, gdal.GA_Update)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.SetSpatialRef(srs)
    ds = None


def _get_subset_bbox(xsize, ysize, subset_bbox=None, target_extent=None, filename=None):
    """Get the subset bounding box for a given target extent.

    Parameters
    ----------
    xsize : int
        size of the x dimension of the image
    ysize : int
        size of the y dimension of the image
    subset_bbox : tuple[int], optional
        Desired bounding box of subset as (left, bottom, right, top)
    target_extent : tuple[int], optional
        Target extent: alternative way to subset the stack like the `-te` gdal option:
    filename : str, optional
        Name of file to get the bounding box from, if providing `target_extent`

    Returns
    -------
    xoff, yoff, xsize_sub, ysize_sub : tuple[int]
    """
    # If target extent is provided, convert to pixel bounding box
    if target_extent is not None:
        subset_bbox = _te_to_bbox(target_extent, filename=filename)

    if subset_bbox is not None:
        left, bottom, right, top = subset_bbox
        xoff = left
        yoff = top
        xsize_sub = right - left
        ysize_sub = bottom - top

    else:
        xoff, yoff, xsize_sub, ysize_sub = 0, 0, xsize, ysize

    return xoff, yoff, xsize_sub, ysize_sub


def _bbox_to_te(subset_bbox, ds=None, filename=None):
    """Convert pixel bounding box to target extent box, in georeferenced coordinates."""
    left, bottom, right, top = subset_bbox  # in pixels
    xmin, ymin = _rowcol_to_xy(bottom, left, ds=ds, filename=filename)
    xmax, ymax = _rowcol_to_xy(top, right, ds=ds, filename=filename)
    return xmin, ymin, xmax, ymax


def _te_to_bbox(target_extent, ds=None, filename=None):
    """Convert target extent to pixel bounding box, in georeferenced coordinates."""
    xmin, ymin, xmax, ymax = target_extent  # in georeferenced coordinates
    left, bottom = _xy_to_rowcol(xmin, ymin, ds=ds, filename=filename)
    right, top = _xy_to_rowcol(xmax, ymax, ds=ds, filename=filename)
    return left, bottom, right, top


def _rowcol_to_xy(row, col, ds=None, filename=None):
    """Convert a row and column index to coordinates in the georeferenced space."""
    if ds is None:
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        ds = None
    else:
        gt = ds.GetGeoTransform()
    return _apply_gt(gt, col, row)


def _apply_gt(gt, xpixel, ypixel):
    # Reference: https://gdal.org/tutorials/geotransforms_tut.html
    x = gt[0] + xpixel * gt[1] + ypixel * gt[2]
    y = gt[3] + xpixel * gt[4] + ypixel * gt[5]
    return x, y


def _xy_to_rowcol(x, y, ds=None, filename=None):
    """Convert coordinates in the georeferenced space to a row and column index."""
    if ds is None:
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        ds = None
    else:
        gt = ds.GetGeoTransform()
    gt = gdal.InvGeoTransform(ds.GetGeoTransform())
    return _apply_gt(gt, x, y)


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
