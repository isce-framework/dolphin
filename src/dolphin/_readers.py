from os import fspath
from pathlib import Path
from typing import Generator, Optional, Sequence

import numpy as np
from osgeo import gdal

from dolphin import io, utils
from dolphin._dates import get_dates, sort_files_by_date
from dolphin._types import Bbox, Filename
from dolphin.stack import logger


class VRTStack:
    """Class for creating a virtual stack of raster files.

    Attributes
    ----------
    file_list : list[Filename]
        Paths or GDAL-compatible strings (NETCDF:...) for paths to files.
    outfile : pathlib.Path, optional (default = Path("slc_stack.vrt"))
        Name of output file to write
    dates : list[list[DateOrDatetime]]
        list, where each entry is all dates matched from the corresponding file
        in `file_list`. This is used to sort the files by date.
        Each entry is a list because some files (compressed SLCs) may have
        multiple dates in the filename.
    use_abs_path : bool, optional (default = True)
        Write the filepaths of the SLCs in the VRT as "relative=0"
    subdataset : str, optional
        Subdataset to use from the files in `file_list`, if using NetCDF files.
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
    nodata_mask_file : pathlib.Path, optional
        Path to file containing a mask of pixels containing with nodata
        in every images. Used for skipping the loading of these pixels.
    file_date_fmt : str, optional (default = "%Y%m%d")
        Format string for parsing the dates from the filenames.
        Passed to [dolphin._dates.get_dates][].
    """

    def __init__(
        self,
        file_list: Sequence[Filename],
        outfile: Filename = "slc_stack.vrt",
        use_abs_path: bool = True,
        subdataset: Optional[str] = None,
        pixel_bbox: Optional[tuple[int, int, int, int]] = None,
        target_extent: Optional[tuple[float, float, float, float]] = None,
        latlon_bbox: Optional[Bbox] = None,
        sort_files: bool = True,
        file_date_fmt: str = "%Y%m%d",
        write_file: bool = True,
        fail_on_overwrite: bool = False,
        skip_size_check: bool = False,
    ):
        """Initialize a VRTStack object for a list of files, optionally subsetting."""
        if Path(outfile).exists() and write_file:
            if fail_on_overwrite:
                raise FileExistsError(
                    f"Output file {outfile} already exists. "
                    "Please delete or specify a different output file. "
                    "To create from an existing VRT, use the `from_vrt_file` method."
                )
            else:
                logger.info(f"Overwriting {outfile}")

        files: list[Filename] = [Path(f) for f in file_list]
        self._use_abs_path = use_abs_path
        if use_abs_path:
            files = [utils._resolve_gdal_path(p) for p in files]
        # Extract the date/datetimes from the filenames
        dates = [get_dates(f, fmt=file_date_fmt) for f in files]
        if sort_files:
            files, dates = sort_files_by_date(  # type: ignore
                files, file_date_fmt=file_date_fmt
            )

        # Save the attributes
        self.file_list = files
        self.dates = dates

        self.outfile = Path(outfile).resolve()
        # Assumes that all files use the same subdataset (if NetCDF)
        self.subdataset = subdataset

        if not skip_size_check:
            io._assert_images_same_size(self._gdal_file_strings)

        # Use the first file in the stack to get size, transform info
        ds = gdal.Open(fspath(self._gdal_file_strings[0]))
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        # Should be CFloat32
        self.gdal_dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        # Save these for setting at the end
        self.gt = ds.GetGeoTransform()
        self.proj = ds.GetProjection()
        self.srs = ds.GetSpatialRef()
        ds = None
        # Save the subset info

        self.xoff, self.yoff = 0, 0
        self.xsize_sub, self.ysize_sub = self.xsize, self.ysize

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
                chunk_size = io.get_raster_chunk_size(filename)
                # chunks in a vrt have a min of 16, max of 2**14=16384
                # https://github.com/OSGeo/gdal/blob/2530defa1e0052827bc98696e7806037a6fec86e/frmts/vrt/vrtrasterband.cpp#L339
                if any([b < 16 for b in chunk_size]) or any(
                    [b > 16384 for b in chunk_size]
                ):
                    chunk_str = ""
                else:
                    chunk_str = (
                        f'blockXSize="{chunk_size[0]}" blockYSize="{chunk_size[1]}"'
                    )
                outstr = f"""  <VRTRasterBand dataType="{self.gdal_dtype}" band="{idx}" {chunk_str}>
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

    @property
    def _gdal_file_strings(self):
        """Get the GDAL-compatible paths to write to the VRT.

        If we're not using .h5 or .nc, this will just be the file_list as is.
        """
        return [io.format_nc_filename(f, self.subdataset) for f in self.file_list]

    def read_stack(
        self,
        band: Optional[int] = None,
        subsample_factor: int = 1,
        rows: Optional[slice] = None,
        cols: Optional[slice] = None,
        masked: bool = False,
    ):
        """Read in the SLC stack."""
        return io.load_gdal(
            self.outfile,
            band=band,
            subsample_factor=subsample_factor,
            rows=rows,
            cols=cols,
            masked=masked,
        )

    def __fspath__(self):
        # Allows os.fspath() to work on the object, enabling rasterio.open()
        return fspath(self.outfile)

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

        Note that we are parsing the XML, not using `GetFilelist`, because the
        function does not seem to work when using HDF5 files. E.g.

            <SourceFilename ="1">NETCDF:20220111.nc:SLC/VV</SourceFilename>

        This would not get added to the result of `GetFilelist`

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

        testname = file_strings[0].upper()
        if testname.startswith("HDF5:") or testname.startswith("NETCDF:"):
            name_triplets = [name.split(":") for name in file_strings]
            prefixes, filepaths, subdatasets = zip(*name_triplets)
            # Remove quoting if it was present
            filepaths = [f.replace('"', "").replace("'", "") for f in filepaths]
            if len(set(subdatasets)) > 1:
                raise NotImplementedError("Only 1 subdataset name is supported")
            sds = (
                subdatasets[0].replace('"', "").replace("'", "").lstrip("/")
            )  # Only returning one subdataset name
        else:
            # If no prefix, the file_strings are actually paths
            filepaths, sds = file_strings, None
        return list(filepaths), sds

    def iter_blocks(
        self,
        overlaps: tuple[int, int] = (0, 0),
        block_shape: tuple[int, int] = (512, 512),
        skip_empty: bool = True,
        nodata_mask: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> Generator[tuple[np.ndarray, tuple[slice, slice]], None, None]:
        """Iterate over blocks of the stack.

        Loads all images for one window at a time into memory.

        Parameters
        ----------
        overlaps : tuple[int, int], optional
            Pixels to overlap each block by (rows, cols)
            By default (0, 0)
        block_shape : tuple[int, int], optional
            2D shape of blocks to load at a time.
            Loads all dates/bands at a time with this shape.
        skip_empty : bool, optional (default True)
            Skip blocks that are entirely empty (all NaNs)
        nodata_mask : bool, optional
            Optional mask indicating nodata values. If provided, will skip
            blocks that are entirely nodata.
            1s are the nodata values, 0s are valid data.
        show_progress : bool, default=True
            If true, displays a `rich` ProgressBar.

        Yields
        ------
        tuple[np.ndarray, tuple[slice, slice]]
            Iterator of (data, (slice(row_start, row_stop), slice(col_start, col_stop))

        """
        self._loader = io.EagerLoader(
            self.outfile,
            block_shape=block_shape,
            overlaps=overlaps,
            nodata_mask=nodata_mask,
            skip_empty=skip_empty,
            show_progress=show_progress,
        )
        yield from self._loader.iter_blocks()

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

    # To allow VRTStack to be passed to `dask.array.from_array`, we need:
    # .shape, .ndim, .dtype and support numpy-style slicing.
    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            return self.read_stack(band=index + 1)

        # TODO: raise an error if they try to skip like [::2, ::2]
        # or pass it to read_stack... but I dont think I need to support it.
        n, rows, cols = index
        if isinstance(rows, int):
            rows = slice(rows, rows + 1)
        if isinstance(cols, int):
            cols = slice(cols, cols + 1)
        if isinstance(n, int):
            if n < 0:
                n = len(self) + n
            return self.read_stack(band=n + 1, rows=rows, cols=cols)

        bands = list(range(1, 1 + len(self)))[n]
        if len(bands) == len(self):
            # This will use gdal's ds.ReadAsRaster, no iteration needed
            data = self.read_stack(band=None, rows=rows, cols=cols)
        else:
            data = np.stack(
                [self.read_stack(band=i, rows=rows, cols=cols) for i in bands], axis=0
            )
        return data.squeeze()

    @property
    def dtype(self):
        return io.get_raster_dtype(self._gdal_file_strings[0])
