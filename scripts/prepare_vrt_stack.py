#!/usr/bin/env python
import argparse
import os
import re
from typing import Optional, Tuple

from osgeo import gdal

from atlas import utils

SENTINEL_WAVELENGTH = 0.05546576


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

    if target_extent is None and subset_bbox is None:
        return
    elif target_extent is not None and subset_bbox is not None:
        raise ValueError("Cannot specify both target_extent and subset_bbox")

    # Otherwise, subset the VRT to the target extent using gdal_translate
    # Set the geotransform and projection
    ds = gdal.Open(outfile, gdal.GA_Update)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.SetSpatialRef(srs)
    if target_extent is not None:
        assert len(target_extent) == 4
        # projWin is (ulx, uly, lrx, lry), so need to reorder
        xmin, ymin, xmax, ymax = target_extent
        projwin = xmin, ymax, xmax, ymin
        options = gdal.TranslateOptions(projWin=projwin)
    elif subset_bbox:
        options = gdal.TranslateOptions(srcWin=subset_bbox)

    temp_file = outfile + ".tmp.vrt"
    gdal.Translate(temp_file, ds, options=options)
    ds = None
    os.rename(temp_file, outfile)


def get_subset_bbox(xsize, ysize, subset_bbox=None):
    """Get the subset bounding box for a given target extent.

    Parameters
    ----------
    xsize : int
        size of the x dimension of the image
    ysize : int
        size of the y dimension of the image
    subset_bbox : tuple[int], optional
        Desired bounding box of subset as (left, bottom, right, top)

    Returns
    -------
    xoff, yoff, xsize_sub, ysize_sub : tuple[int]
    """
    if subset_bbox is not None:
        left, bottom, right, top = subset_bbox
        # -te xmin ymin xmax ymax
        xoff = left
        yoff = top
        xsize_sub = right - left
        ysize_sub = bottom - top
    else:
        xoff, yoff, xsize_sub, ysize_sub = 0, 0, xsize, ysize

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
