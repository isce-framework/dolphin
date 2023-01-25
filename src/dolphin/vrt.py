#!/usr/bin/env python
from math import nan
from os import fspath
from pathlib import Path
from pprint import pformat
from typing import Optional, Sequence, Tuple

from osgeo import gdal

from dolphin import io, utils
from dolphin._log import get_log
from dolphin._types import Filename

SENTINEL_WAVELENGTH = 0.05546576

gdal.UseExceptions()
logger = get_log()


DEFAULT_BLOCK_BYTES = 32e6


class VRTStack:
    """Class for creating VRTs from a list of raster files.

    Attributes
    ----------
    file_list : List[pathlib.Path]
        Names of files to stack
    outfile : pathlib.Path, optional (default = Path("slc_stack.vrt"))
        Name of output file to write
    dates : list[datetime.date]
        List of datetimes matched from the files in file_list
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
    nodata_value : float, optional
        Value to use for nodata. If not specified, will use the nodata value
        from the first file in the stack, or nan if not specified in the raster.
    nodata_mask_file : pathlib.Path, optional
        Path to file containing a mask of pixels containing with nodata
        in every images. Used for skipping the loading of these pixels.
    file_date_fmt : str, optional (default = "%Y%m%d")
        Format string for parsing the dates from the filenames.
        Passed to [dolphin.utils.get_dates][].
    """

    def __init__(
        self,
        file_list: Sequence[Filename],
        outfile: Filename = "slc_stack.vrt",
        use_abs_path: bool = True,
        subdataset: Optional[str] = None,
        pixel_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_extent: Optional[Tuple[float, float, float, float]] = None,
        latlon_bbox: Optional[Tuple[float, float, float, float]] = None,
        sort_files: bool = True,
        nodata_value: Optional[float] = None,
        file_date_fmt: str = "%Y%m%d",
        write_file: bool = True,
    ):
        """Initialize a VRTStack object for a list of files, optionally subsetting."""
        if Path(outfile).exists() and write_file:
            raise FileExistsError(
                f"Output file {outfile} already exists. "
                "Please delete or specify a different output file. "
                "To create from an existing VRT, use the `from_vrt_file` method."
            )

        files = [Path(f) for f in file_list]
        self._use_abs_path = use_abs_path
        if use_abs_path:
            files = [p.resolve() for p in files]
        # Extract the date/datetimes from the filenames
        dates = [utils.parse_slc_strings(f, fmt=file_date_fmt) for f in files]
        if sort_files:
            files, dates = utils.sort_files_by_date(files, file_date_fmt=file_date_fmt)

        # Save the attributes
        self.file_list = list(files)
        self.dates = list(dates)
        # save for future parsing of dates with `add_file`
        self.file_date_fmt = file_date_fmt

        self.outfile = Path(outfile).resolve()
        # Assumes that all files use the same subdataset (if NetCDF)
        self._subdataset = subdataset

        # If we're using .h5 or .nc, get the GDAL-compatible paths to write to the VRT
        # Otherwise this will just be the file_list as is

        self._assert_images_same_size()

        if nodata_value is None:
            self.nodata_value = io.get_nodata(self._gdal_file_strings[0]) or nan
        self.nodata_mask_file = (
            self.outfile.parent / f"{self.outfile.stem}_nodata_mask.tif"
        )

        # Use the first file in the stack to get size, transform info
        ds = gdal.Open(fspath(self._gdal_file_strings[0]))
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

        self._set_subset(
            pixel_bbox=pixel_bbox,
            target_extent=target_extent,
            latlon_bbox=latlon_bbox,
            filename=self.file_list[0],
        )
        if write_file:
            self._write()

    def _write(self):
        """Write out the VRT file pointing to the stack of SLCs, erroring if exists."""
        with open(self.outfile, "w") as fid:
            fid.write(
                f'<VRTDataset rasterXSize="{self.xsize_sub}"'
                f' rasterYSize="{self.ysize_sub}">\n'
            )

            for idx, filename in enumerate(self._gdal_file_strings, start=1):
                block_size = io.get_raster_block_size(filename)
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
  </VRTRasterBand>\n"""  # noqa: E501
                fid.write(outstr)

            fid.write("</VRTDataset>")

        # Set the georeferencing metadata
        ds = gdal.Open(fspath(self.outfile), gdal.GA_Update)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        ds.SetSpatialRef(self.srs)
        ds = None

    def _assert_images_same_size(self):
        """Make sure all files in the stack are the same size."""
        from collections import defaultdict

        size_to_file = defaultdict(list)
        for f in self._gdal_file_strings:
            s = io.get_raster_xysize(f)
            size_to_file[s].append(f)
        if len(size_to_file) > 1:
            size_str = pformat(dict(sorted(size_to_file.items())))
            raise ValueError(f"Not files have same raster (x, y) size:\n{size_str}")

    @property
    def _gdal_file_strings(self):
        return [io.format_nc_filename(f, self._subdataset) for f in self.file_list]

    def read_stack(self, band: Optional[int] = None, subsample_factor: int = 1):
        """Read in the SLC stack."""
        return io.load_gdal(self.outfile, band=band, subsample_factor=subsample_factor)

    def __fspath__(self):
        # Allows os.fspath() to work on the object, enabling rasterio.open()
        return fspath(self.outfile)

    def add_file(self, new_file: Filename, sort_files: bool = True):
        """Append a new file to the stack, and (optionally) re-sort."""
        new_file = Path(new_file)
        if self._use_abs_path:
            new_file = new_file.resolve()
        self.file_list.append(new_file)

        # Parse the new date, and add it to the list
        new_date = utils.parse_slc_strings(new_file, fmt=self.file_date_fmt)
        self.dates.append(new_date)
        if sort_files:
            self.file_list, self.dates = utils.sort_files_by_date(
                self.file_list, file_date_fmt=self.file_date_fmt
            )

        self._write()

    def _set_subset(
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
    def from_vrt_file(cls, vrt_file, new_outfile=None, **kwargs):
        """Create a new VRTStack using an existing VRT file."""
        file_list, subdataset = cls._parse_vrt_file(vrt_file)
        if new_outfile is None:
            # Point to the same, if none provided
            new_outfile = vrt_file

        return cls(
            file_list,
            outfile=new_outfile,
            subdataset=subdataset,
            write_file=False,
            **kwargs,
        )

    @staticmethod
    def _parse_vrt_file(vrt_file):
        """Extract the filenames, and possible subdatasets, from a .vrt file.

        Note that we are parsing the XML, not using `GetFileList`, because the
        function does not seem to work when using HDF5 files. E.g.

            <SourceFilename ="1">NETCDF:20220111.nc:SLC/VV</SourceFilename>

        This would not get added to the result of `GetFileList`

        Parameters
        ----------
        vrt_file : Filename
            Path to the VRT file to read.

        Returns
        -------
        filepaths
        """
        file_strings = []
        with open(vrt_file) as f:
            for line in f:
                if "<SourceFilename" not in line:
                    continue
                # Get the middle part of < >filename</ >
                fn = line.split(">")[1].strip().split("<")[0]
                file_strings.append(fn)
        # double check we got the same count
        ds = gdal.Open(fspath(vrt_file))
        count = ds.RasterCount
        ds = None
        if count != len(file_strings):
            raise ValueError(
                f"Found {len(file_strings)} parsing {vrt_file}, but file has"
                f" {count} bands."
            )
        testname = file_strings[0].upper()
        if testname.startswith("HDF5:") or testname.startswith("NETCDF:"):
            name_triplets = [name.split(":") for name in file_strings]
            prefixes, filepaths, subdatasets = zip(*name_triplets)
            # Remove quoting if it was present
            filepaths = [f.strip('"').strip("'") for f in filepaths]
            if len(set(subdatasets)) > 1:
                raise NotImplementedError("Only 1 subdataset name is supported")
            sds = subdatasets[0].lstrip("/")  # Only returning one subdataset name
        else:
            # If no prefix, the file_strings are actually paths
            filepaths, sds = file_strings, None
        return list(filepaths), sds

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
        row_bottom, col_left = io.xy_to_rowcol(xmin, ymin, ds=ds, filename=filename)
        row_top, col_right = io.xy_to_rowcol(xmax, ymax, ds=ds, filename=filename)
        return col_left, row_bottom, col_right, row_top

    def iter_blocks(
        self,
        overlaps: Tuple[int, int] = (0, 0),
        start_offsets: Tuple[int, int] = (0, 0),
        block_shape: Optional[Tuple[int, int]] = None,
        max_bytes: Optional[float] = DEFAULT_BLOCK_BYTES,
        return_slices: bool = False,
        skip_empty: bool = True,
        use_nodata_mask: bool = True,
    ):
        """Iterate over blocks of the stack.

        Loads all images for one window at a time into memory.

        Parameters
        ----------
        overlaps : Tuple[int, int], optional
            Pixels to overlap each block by (rows, cols)
            By default (0, 0)
        start_offsets : Tuple[int, int], optional
            (row, col) number of pixels to offset initial block
            By default (0, 0)
        block_shape : Optional[Tuple[int, int]], optional
            If provided, force the blocks to load in the given shape.
            Otherwise, calculates how much blocks are possible to load
            while staying under `max_bytes` that align wit the data's
            internal chunking/tiling structure.
        max_bytes : Optional[int], optional
            RAM size (in Bytes) to attempt to stay under with each loaded block.
        return_slices : bool, optional (default False)
            return (row_slice, col_slice) indicating the position of the current block.
        skip_empty : bool, optional (default True)
            Skip blocks that are entirely empty (all NaNs)
        use_nodata_mask : bool, optional (default True)
            Use the nodata mask to determine if a block is empty.
            Not used if `skip_empty` is False.

        Yields
        ------
        Tuple[Tuple[int, int], Tuple[int, int]]
            Iterator of ((row_start, row_stop), (col_start, col_stop))
        """
        if block_shape is None:
            block_shape = self._get_block_shape(max_bytes=max_bytes)
        ndm = None
        if skip_empty and use_nodata_mask:
            ndm = self._get_nodata_mask(nodata=self.nodata_value, buffer_pixels=100)

        yield from io.iter_blocks(
            self.outfile,
            block_shape=block_shape,
            overlaps=overlaps,
            start_offsets=start_offsets,
            return_slices=return_slices,
            skip_empty=skip_empty,
            nodata=self.nodata_value,
            nodata_mask=ndm,
        )

    def _get_block_shape(self, max_bytes=DEFAULT_BLOCK_BYTES):
        test_file = self._get_non_vrt_file(self._gdal_file_strings[0])

        return io.get_max_block_shape(
            # Note that we're using the actual first file, not the VRT
            # since the VRT always has the same block size.
            test_file,
            len(self),
            max_bytes=max_bytes,
        )

    def _get_num_blocks(
        self,
        max_bytes=DEFAULT_BLOCK_BYTES,
        overlaps: Tuple[int, int] = (0, 0),
        start_offsets: Tuple[int, int] = (0, 0),
    ):
        """Get the number of blocks that will be loaded when iterating over the stack.

        Assumes no empty blocks will be skipped.
        """
        block_shape = self._get_block_shape(max_bytes=max_bytes)
        return len(
            list(
                io.slice_iterator(
                    arr_shape=self.shape[-2:],
                    block_shape=block_shape,
                    overlaps=overlaps,
                    start_offsets=start_offsets,
                )
            )
        )

    def _get_nodata_mask(self, nodata=nan, buffer_pixels=100):
        if self.nodata_mask_file.exists():
            return io.load_gdal(self.nodata_mask_file).astype(bool)
        # TODO: Write the code to grab the pre-computed polygon, rather
        # than the loading data.
        else:
            return io.get_stack_nodata_mask(
                self.outfile,
                output_file=self.nodata_mask_file,
                nodata=nodata,
                buffer_pixels=buffer_pixels,
            )

    @staticmethod
    def _get_non_vrt_file(filename: Filename):
        """Get one of the files within a VRT.

        If the file is not a VRT, return the file itself.
        Will traverse nested VRTs.
        """
        if Path(filename).suffix == ".vrt":
            file_list = gdal.Info(fspath(filename), format="json")["files"]
            if len(file_list) <= 1:
                raise ValueError(f"VRT file {filename} contains no files")
            return VRTStack._get_non_vrt_file(file_list[1])
        return filename

    @property
    def shape(self):
        """Get the 3D shape of the stack."""
        xsize, ysize = io.get_raster_xysize(self._gdal_file_strings[0])
        return (len(self.file_list), ysize, xsize)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        outname = fspath(self.outfile) if self.outfile else "(not written)"
        return f"VRTStack({len(self.file_list)} bands, outfile={outname})"

    def __eq__(self, other):
        if not isinstance(other, VRTStack):
            return False
        return (
            self._gdal_file_strings == other._gdal_file_strings
            and self.outfile == other.outfile
        )
