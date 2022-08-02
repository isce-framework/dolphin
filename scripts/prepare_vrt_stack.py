#!/usr/bin/env python
import argparse
import os
import re
from typing import Optional, Tuple

from osgeo import gdal

from atlas import utils

SENTINEL_WAVELENGTH = 0.05546576


def create_vrt_stack_manual(
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
    # Use the first file in the stack to get size, transform info
    ds = gdal.Open(file_list[0])
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    # Save the transform info for later
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    srs = ds.GetSpatialRef()
    ds = None

    xoff, yoff, xsize_sub, ysize_sub = 0, 0, xsize, ysize
    print("write vrt file for stack directory")
    # xoff, yoff, xsize_sub, ysize_sub = get_subset_bbox(
    #     xsize, ysize, subset_bbox=subset_bbox
    # )
    with open(outfile, "w") as fid:
        fid.write(f'<VRTDataset rasterXSize="{xsize_sub}" rasterYSize="{ysize_sub}">\n')

        for idx, filename in enumerate(file_list, start=1):
            if use_abs_path:
                filename = os.path.abspath(filename)
            date = _get_date(filename)
            outstr = f"""    <VRTRasterBand dataType="CFloat32" band="{idx}">
        <SimpleSource>
            <SourceFilename>{filename}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="CFloat32"/>
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

    # Set the geotransform and projection
    ds = gdal.Open(outfile, gdal.GA_Update)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.SetSpatialRef(srs)
    xres, yres = gt[1], gt[5]
    ds = None

    if target_extent is None and subset_bbox is None:
        return
    elif target_extent is not None and subset_bbox is not None:
        raise ValueError("Cannot specify both target_extent and subset_bbox")

    # Otherwise, subset the VRT to the target extent using gdal_translate
    ds = gdal.Open(outfile, gdal.GA_Update)
    if target_extent is not None:
        # assert len(target_extent) == 4
        # # projWin is (ulx, uly, lrx, lry), so need to reorder
        # xmin, ymin, xmax, ymax = target_extent
        # projwin = xmin, ymax, xmax, ymin
        # options = gdal.TranslateOptions(projWin=projwin)
        options = gdal.WarpOptions(outputBounds=target_extent, xRes=xres, yRes=yres)
    elif subset_bbox:
        # options = gdal.TranslateOptions(srcWin=_bbox_to_srcwin(subset_bbox))
        options = gdal.WarpOptions(
            outputBounds=subset_bbox, outputBoundsSRS="", xRes=xres, yRes=yres
        )

    temp_file = outfile + ".tmp.vrt"
    # gdal.Translate(temp_file, ds, options=options)
    gdal.Warp(temp_file, ds, options=options)
    ds = None
    ds = gdal.Open(temp_file)
    # os.rename(temp_file, outfile)


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


def create_vrt_stack(
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
        file_list = [os.path.abspath(f) for f in file_list]

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
        date = _get_date(filename)
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


def _bbox_to_srcwin(subset_bbox=None):
    """Convert a gdalwarp -te option to a gdal_translate -srcwin option.

    Parameters
    ----------
    subset_bbox : tuple[int], optional
        Desired bounding box of subset as (left, bottom, right, top)

    Returns
    -------
    xoff, yoff, xsize_sub, ysize_sub : tuple[int]
    """
    # -te xmin ymin xmax ymax
    left, bottom, right, top = subset_bbox
    xoff = left
    yoff = top
    xsize_sub = right - left
    ysize_sub = bottom - top

    return xoff, yoff, xsize_sub, ysize_sub


def _get_date(filename):
    match = re.search(r"\d{4}\d{2}\d{2}", filename)
    if not match:
        raise ValueError(f"{filename} does not contain date as YYYYMMDD")
    return match.group()


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


if __name__ == "__main__":
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
    print("Number of SLCs Used: ", num_slc)

    # Set up single stack file
    utils.mkdir_p(args.out_dir)
    outfile = os.path.join(args.out_dir, args.out_vrt_name)
    create_vrt_stack(
        file_list,
        outfile=outfile,
        subset_bbox=args.subset_bbox,
        target_extent=args.target_extent,
    )
