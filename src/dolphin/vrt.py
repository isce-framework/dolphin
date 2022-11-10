#!/usr/bin/env python
import argparse
from os import fspath
from pathlib import Path
from typing import Optional, Sequence, Tuple

from osgeo import gdal

from dolphin import utils
from dolphin.log import get_log
from dolphin.utils import Pathlike

SENTINEL_WAVELENGTH = 0.05546576

gdal.UseExceptions()
logger = get_log()


class VRTStack:
    """Class for creating VRTs from a list of raster files.

    Attributes
    ----------
    file_list : list
        Names of files to stack
    outfile : Path, optional (default = Path("slc_stack.vrt"))
        Name of output file to write
    use_abs_path : bool, optional (default = True)
        Write the filepaths of the SLCs in the VRT as "relative=0"
    pixel_bbox : tuple[int], optional
        Desired bounding box (in pixels) of subset as (left, bottom, right, top)
    target_extent : tuple[int], optional
        Target extent: alternative way to subset the stack like the `-te` gdal option:
            (xmin, ymin, xmax, ymax) in units of the SLCs' SRS (e.g. UTM coordinates)
    latlon_bbox : tuple[int], optional
        Bounding box in lat/lon coordinates: (left, bottom, right, top)
    sort_files : bool, optional (default = True)
        Sort the files in `file_list`. Assumes that the naming convention
        will sort the files in increasing time order.
    """

    def __init__(
        self,
        file_list: Sequence[Pathlike],
        outfile: Pathlike = "slc_stack.vrt",
        use_abs_path: bool = True,
        pixel_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_extent: Optional[Tuple[float, float, float, float]] = None,
        latlon_bbox: Optional[Tuple[float, float, float, float]] = None,
        sort_files: bool = True,
    ):
        """Initialize a VRTStack object for a list of files, optionally subsetting."""
        if sort_files:
            # Sort by the filename, not the whole path
            file_list = sorted(
                [Path(f) for f in file_list],
                # Make sure compressed slcs are at the beginning of the stack
                key=lambda f: Path(f).stem.replace("compressed_", ""),
            )
        if use_abs_path:
            file_list = [Path(f).resolve() for f in file_list]
        self.file_list = file_list
        self.outfile = Path(outfile).resolve()

        # Use the first file in the stack to get size, transform info
        ds = gdal.Open(fspath(file_list[0]))
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        # Should be CFloat32
        self.dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        # Save these for setting at the end
        self.gt = ds.GetGeoTransform()
        self.proj = ds.GetProjection()
        self.srs = ds.GetSpatialRef()
        ds = None
        # Save the subset info

        self.set_subset(
            pixel_bbox=pixel_bbox,
            target_extent=target_extent,
            latlon_bbox=latlon_bbox,
            filename=file_list[0],
        )

    def write(self, outfile: Optional[Pathlike] = None):
        """Write out the VRT file pointing to the stack of SLCs.

        Uses the `outfile` passed to the constructor, unless `outfile` is provided.
        """
        if outfile is None:
            outfile = self.outfile
        if not outfile:
            raise ValueError("No output file specified")

        with open(self.outfile, "w") as fid:
            fid.write(
                f'<VRTDataset rasterXSize="{self.xsize_sub}"'
                f' rasterYSize="{self.ysize_sub}">\n'
            )

            for idx, filename in enumerate(self.file_list, start=1):
                filename = str(Path(filename).resolve())
                date = utils.get_dates(filename)[0]

                block_size = utils.get_block_size(filename)
                # blocks in a vrt have a min of 16, max of 2**14=16384
                # https://github.com/OSGeo/gdal/blob/2530defa1e0052827bc98696e7806037a6fec86e/frmts/vrt/vrtrasterband.cpp#L339
                if any([b < 16 for b in block_size]) or any(
                    [b > 16384 for b in block_size]
                ):
                    block_str = ""
                else:
                    block_str = (
                        f'blockXSize="{block_size[0]}" blockYSize="{block_size[1]}"'
                    )
                outstr = f"""  <VRTRasterBand dataType="{self.dtype}" band="{idx}" {block_str}>
    <SimpleSource>
      <SourceFilename>{filename}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="{self.xoff}" yOff="{self.yoff}" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
      <DstRect xOff="0" yOff="0" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
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
        ds = gdal.Open(fspath(self.outfile), gdal.GA_Update)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        ds.SetSpatialRef(self.srs)
        ds = None

    def read_stack(self):
        """Read in the SLC stack."""
        # TODO: Implement iterating, reading in chunks, block iterator
        if not self.outfile:
            raise ValueError("No output file specified")
        if not self.outfile.exists():
            self.write()
        ds = gdal.Open(fspath(self.outfile))
        stack = ds.ReadAsArray()
        ds = None
        return stack

    def __fspath__(self):
        # Allows os.fspath() to work on the object, enabling rasterio.open()
        return fspath(self.outfile)

    def add_file(self, new_file):
        """Add a new file to the stack and re-sort."""
        self.file_list = sorted(self.file_list + [new_file])

    def set_subset(
        self, pixel_bbox=None, target_extent=None, latlon_bbox=None, filename=None
    ):
        """Save the subset bounding box for a given target extent.

        Sets the attributes `xoff`, `yoff`, `xsize_sub`, `ysize_sub` for the subset.

        Parameters
        ----------
        pixel_bbox : tuple[int], optional
            Desired bounding box of subset as (left, bottom, right, top)
        target_extent : tuple[int], optional
            Target extent: alternative way to subset the stack like the `-te` gdal option:
        latlon_bbox : tuple[int], optional
            Bounding box in lat/lon coordinates: (left, bottom, right, top)
        filename : str, optional
            Name of file to get the bounding box from, if providing `target_extent`

        """
        if all(
            (pixel_bbox is not None, target_extent is not None, latlon_bbox is not None)
        ):
            raise ValueError(
                "Cannot only specif one of `pixel_bbox` and `latlon_bbox`, and"
                " `target_extent`"
            )
        # If target extent is provided, convert to pixel bounding box
        if latlon_bbox is not None:
            # convert in 2 steps: first lat/lon -> UTM, then UTM -> pixel
            target_extent = VRTStack._latlon_bbox_to_te(latlon_bbox, filename=filename)
        if target_extent is not None:
            # convert UTM -> pixels
            pixel_bbox = VRTStack._te_to_bbox(target_extent, filename=filename)

        if pixel_bbox is not None:
            left, bottom, right, top = pixel_bbox
            self.xoff = left
            self.yoff = top
            self.xsize_sub = right - left
            self.ysize_sub = bottom - top
        else:
            self.xoff, self.yoff = 0, 0
            self.xsize_sub, self.ysize_sub = self.xsize, self.ysize

    @classmethod
    def from_vrt_file(cls, vrt_file, new_outfile="slc_stack.vrt", **kwargs):
        """Create a new VRTStack using an existing VRT file."""
        ds = gdal.Open(fspath(vrt_file))
        # First item is the `vrt_file` itself
        file_list = [Path(f) for f in ds.GetFileList()[1:]]
        ds = None
        return cls(file_list, outfile=new_outfile, **kwargs)

    @staticmethod
    def _latlon_bbox_to_te(
        latlon_bbox,
        filename,
        epsg=None,
    ):
        """Convert a lat/lon bounding box to a target extent.

        Parameters
        ----------
        latlon_bbox : tuple[float]
            Bounding box in lat/lon coordinates: (left, bottom, right, top)
        filename : str
            Name of file to get the destination SRS from
        epsg : int or str, optional
            EPSG code of the destination SRS

        Returns
        -------
        target_extent : tuple[float]
            Target extent: (xmin, ymin, xmax, ymax) in units of `filename`s SRS (e.g. UTM)
        """
        from pyproj import Transformer

        if epsg is None:
            ds = gdal.Open(fspath(filename))
            srs_out = ds.GetSpatialRef()
            epsg = int(srs_out.GetAttrValue("AUTHORITY", 1))
            ds = None
        if int(epsg) == 4326:
            return latlon_bbox
        t = Transformer.from_crs(4326, epsg, always_xy=True)
        left, bottom, right, top = latlon_bbox
        return t.transform(left, bottom) + t.transform(right, top)

    @staticmethod
    def _te_to_bbox(target_extent, ds=None, filename=None):
        """Convert target extent to pixel bounding box, in georeferenced coordinates."""
        xmin, ymin, xmax, ymax = target_extent  # in georeferenced coordinates
        left, bottom = VRTStack._xy_to_rowcol(xmin, ymin, ds=ds, filename=filename)
        right, top = VRTStack._xy_to_rowcol(xmax, ymax, ds=ds, filename=filename)
        return left, bottom, right, top

    @staticmethod
    def _apply_gt(gt, xpixel, ypixel):
        # Reference: https://gdal.org/tutorials/geotransforms_tut.html
        x = gt[0] + xpixel * gt[1] + ypixel * gt[2]
        y = gt[3] + xpixel * gt[4] + ypixel * gt[5]
        return x, y

    @staticmethod
    def _xy_to_rowcol(x, y, ds=None, filename=None):
        """Convert coordinates in the georeferenced space to a row and column index."""
        if ds is None:
            ds = gdal.Open(fspath(filename))
            gt = ds.GetGeoTransform()
            ds = None
        else:
            gt = ds.GetGeoTransform()
        gt = gdal.InvGeoTransform(ds.GetGeoTransform())
        return VRTStack._apply_gt(gt, x, y)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        outname = fspath(self.outfile) if self.outfile else "(not written)"
        return f"VRTStack({len(self.file_list)} bands, outfile={outname})"

    def get_block_shape(self, max_bytes=100e6, default_chunk_size=(None, 256, 256)):
        return utils.get_max_block_shape(
            self.file_list[0],
            len(self),
            max_bytes=max_bytes,
            default_tile_size=default_chunk_size,
        )

    def iter_blocks(
        self,
        overlaps: Tuple[int, int] = (0, 0),
        start_offsets: Tuple[int, int] = (0, 0),
    ):
        yield from utils.iter_blocks(
            self.outfile,
            self.get_block_shape()[-2:],
            overlaps=overlaps,
            start_offsets=start_offsets,
        )


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
        default="slc_stack.vrt",
        help="Name of output SLC containing all images",
    )
    parser.add_argument(
        "-b",
        "--pixel-bbox",
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
    parser.add_argument(
        "-bl",
        "--latlon-bbox",
        type=float,
        nargs=4,
        metavar=("lonmin", "latmin", "lonmax", "latmax"),
        default=None,
        help=(
            "Target extent in longitude/latitude. An alternative way to subset the"
            " stack."
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
    stack = VRTStack(
        file_list,
        outfile=outfile,
        pixel_bbox=args.pixel_bbox,
        target_extent=args.target_extent,
        latlon_bbox=args.latlon_bbox,
    )
    stack.write()


if __name__ == "__main__":
    main()
