"""Functions for reading from and writing to raster files.

This module heavily relies on GDAL and provides many convenience/
wrapper functions to write/iterate over blocks of large raster files.
"""
import math
from datetime import date
from os import fspath
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from osgeo import gdal
from pyproj import CRS

from dolphin._background import _DEFAULT_TIMEOUT, BackgroundReader, BackgroundWriter
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.utils import gdal_to_numpy_type, numpy_to_gdal_type, progress

gdal.UseExceptions()

__all__ = [
    "load_gdal",
    "write_arr",
    "write_block",
    "EagerLoader",
]


DEFAULT_TILE_SIZE = [128, 128]
DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    f"BLOCKXSIZE={DEFAULT_TILE_SIZE[1]}",
    f"BLOCKYSIZE={DEFAULT_TILE_SIZE[0]}",
)
DEFAULT_ENVI_OPTIONS = ("SUFFIX=ADD",)
DEFAULT_HDF5_OPTIONS = dict(
    # https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    chunks=DEFAULT_TILE_SIZE,
    compression="gzip",
    compression_opts=4,
    shuffle=True,
)
DEFAULT_DATETIME_FORMAT = "%Y%m%d"

logger = get_log(__name__)


def load_gdal(
    filename: Filename,
    *,
    band: Optional[int] = None,
    subsample_factor: int = 1,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
    masked: bool = False,
):
    """Load a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int, optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.
    rows : slice, optional
        Rows to load. Default is None (load all rows).
    cols : slice, optional
        Columns to load. Default is None (load all columns).
    masked : bool, optional
        If True, return a masked array using the raster's `nodata` value.
        Default is False.

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.
    """
    ds = gdal.Open(fspath(filename))
    nrows, ncols = ds.RasterYSize, ds.RasterXSize
    # Make an output object of the right size
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    if rows is not None and cols is not None:
        xoff, yoff = cols.start, rows.start
        row_stop = min(rows.stop, nrows)
        col_stop = min(cols.stop, ncols)
        xsize, ysize = col_stop - cols.start, row_stop - rows.start
        if xsize <= 0 or ysize <= 0:
            raise IndexError(
                f"Invalid row/col slices: {rows}, {cols} for file {filename} of size"
                f" {nrows}x{ncols}"
            )
        nrows_out, ncols_out = ysize // subsample_factor, xsize // subsample_factor
    else:
        xoff, yoff = 0, 0
        xsize, ysize = ncols, nrows
        nrows_out, ncols_out = nrows // subsample_factor, ncols // subsample_factor
    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty((count, nrows_out, ncols_out), dtype=dt)
        ds.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((nrows_out, ncols_out), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)

    if not masked:
        return out
    # Get the nodata value
    nd = get_raster_nodata(filename)
    if nd is not None and np.isnan(nd):
        return np.ma.masked_invalid(out)
    else:
        return np.ma.masked_equal(out, nd)


def format_nc_filename(filename: Filename, ds_name: Optional[str] = None) -> str:
    """Format an HDF5/NetCDF filename with dataset for reading using GDAL.

    If `filename` is already formatted, or if `filename` is not an HDF5/NetCDF
    file (based on the file extension), it is returned unchanged.

    Parameters
    ----------
    filename : str or PathLike
        Filename to format.
    ds_name : str, optional
        Dataset name to use. If not provided for a .h5 or .nc file, an error is raised.

    Returns
    -------
    str
        Formatted filename.

    Raises
    ------
    ValueError
        If `ds_name` is not provided for a .h5 or .nc file.
    """
    # If we've already formatted the filename, return it
    if str(filename).startswith("NETCDF:") or str(filename).startswith("HDF5:"):
        return str(filename)

    if not (fspath(filename).endswith(".nc") or fspath(filename).endswith(".h5")):
        return fspath(filename)

    # Now we're definitely dealing with an HDF5/NetCDF file
    if ds_name is None:
        raise ValueError("Must provide dataset name for HDF5/NetCDF files")

    return f'NETCDF:"{filename}":"//{ds_name.lstrip("/")}"'


def copy_projection(src_file: Filename, dst_file: Filename) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    ds_src = gdal.Open(fspath(src_file))
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        logger.info("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def get_raster_xysize(filename: Filename) -> Tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def get_raster_nodata(filename: Filename, band: int = 1) -> Optional[float]:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.
    """
    ds = gdal.Open(fspath(filename))
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    return nodata


def get_raster_crs(filename: Filename) -> CRS:
    """Get the CRS from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    CRS
        CRS.
    """
    ds = gdal.Open(fspath(filename))
    crs = CRS.from_wkt(ds.GetProjection())
    return crs


def get_raster_gt(filename: Filename) -> List[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        Geotransform.
    """
    ds = gdal.Open(fspath(filename))
    gt = ds.GetGeoTransform()
    return gt


def get_raster_dtype(filename: Filename) -> np.dtype:
    """Get the data type from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    np.dtype
        Data type.
    """
    ds = gdal.Open(fspath(filename))
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)
    return dt


def get_raster_driver(filename: Filename) -> str:
    """Get the GDAL driver `ShortName` from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    str
        Driver name.
    """
    ds = gdal.Open(fspath(filename))
    driver = ds.GetDriver().ShortName
    return driver


def get_raster_bounds(
    filename: Optional[Filename] = None, ds: Optional[gdal.Dataset] = None
) -> Tuple[float, float, float, float]:
    """Get the (left, bottom, right, top) bounds of the image."""
    if ds is None:
        if filename is None:
            raise ValueError("Must provide either `filename` or `ds`")
        ds = gdal.Open(fspath(filename))

    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    left, top = _apply_gt(gt=gt, x=0, y=0)
    right, bottom = _apply_gt(gt=gt, x=xsize, y=ysize)

    return (left, bottom, right, top)


def rowcol_to_xy(
    row: int,
    col: int,
    ds: Optional[gdal.Dataset] = None,
    filename: Optional[Filename] = None,
) -> Tuple[float, float]:
    """Convert indexes in the image space to georeferenced coordinates."""
    return _apply_gt(ds, filename, col, row)


def xy_to_rowcol(
    x: float,
    y: float,
    ds: Optional[gdal.Dataset] = None,
    filename: Optional[Filename] = None,
    do_round=True,
) -> Tuple[int, int]:
    """Convert coordinates in the georeferenced space to a row and column index."""
    col, row = _apply_gt(ds, filename, x, y, inverse=True)
    # Need to convert to int, otherwise we get a float
    if do_round:
        # round up to the nearest pixel, instead of banker's rounding
        row = int(math.floor(row + 0.5))
        col = int(math.floor(col + 0.5))
    return int(row), int(col)


def _apply_gt(
    ds=None, filename=None, x=None, y=None, inverse=False, gt=None
) -> Tuple[float, float]:
    """Read the (possibly inverse) geotransform, apply to the x/y coordinates."""
    if gt is None:
        if ds is None:
            ds = gdal.Open(fspath(filename))
            gt = ds.GetGeoTransform()
            ds = None
        else:
            gt = ds.GetGeoTransform()

    if inverse:
        gt = gdal.InvGeoTransform(gt)
    # Reference: https://gdal.org/tutorials/geotransforms_tut.html
    x = gt[0] + x * gt[1] + y * gt[2]
    y = gt[3] + x * gt[4] + y * gt[5]
    return x, y


def compute_out_shape(
    shape: Tuple[int, int], strides: Dict[str, int]
) -> Tuple[int, int]:
    """Calculate the output size for an input `shape` and row/col `strides`.

    Parameters
    ----------
    shape : Tuple[int, int]
        Input size: (rows, cols)
    strides : Dict[str, int]
        {"x": x strides, "y": y strides}

    Returns
    -------
    out_shape : Tuple[int, int]
        Size of output after striding

    Notes
    -----
    If there is not a full window (of size `strides`), the end
    will get cut off rather than padded with a partial one.
    This should match the output size of `[dolphin.utils.take_looks][]`.

    As a 1D example, in array of size 6 with `strides`=3 along this dim,
    we could expect the pixels to be centered on indexes
    `[1, 4]`.

        [ 0  1  2   3  4  5]

    So the output size would be 2, since we have 2 full windows.
    If the array size was 7 or 8, we would have 2 full windows and 1 partial,
    so the output size would still be 2.
    """
    rows, cols = shape
    rs, cs = strides["y"], strides["x"]
    return (rows // rs, cols // cs)


def write_arr(
    *,
    arr: Optional[ArrayLike],
    output_name: Filename,
    like_filename: Optional[Filename] = None,
    driver: Optional[str] = "GTiff",
    options: Optional[Sequence] = None,
    nbands: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
    dtype: Optional[DTypeLike] = None,
    geotransform: Optional[Sequence[float]] = None,
    strides: Optional[Dict[str, int]] = None,
    projection: Optional[Any] = None,
    nodata: Optional[Union[float, str]] = None,
):
    """Save an array to `output_name`.

    If `like_filename` if provided, copies the projection/nodata.
    Options can be overridden by passing `driver`/`nbands`/`dtype`.

    If arr is None, create an empty file with the same x/y shape as `like_filename`.

    Parameters
    ----------
    arr : ArrayLike, optional
        Array to save. If None, create an empty file.
    output_name : str or Path
        Path to save the file to.
    like_filename : str or Path, optional
        Path to a file to copy raster shape/metadata from.
    driver : str, optional
        GDAL driver to use. Default is "GTiff".
    options : list, optional
        List of options to pass to the driver. Default is DEFAULT_TIFF_OPTIONS.
    nbands : int, optional
        Number of bands to save. Default is 1.
    shape : tuple, optional
        (rows, cols) of desired output file.
        Overrides the shape of the output file, if using `like_filename`.
    dtype : DTypeLike, optional
        Data type to save. Default is `arr.dtype` or the datatype of like_filename.
    geotransform : List, optional
        Geotransform to save. Default is the geotransform of like_filename.
        See https://gdal.org/tutorials/geotransforms_tut.html .
    strides : dict, optional
        If using `like_filename`, used to change the pixel size of the output file.
        {"x": x strides, "y": y strides}
    projection : str or int, optional
        Projection to save. Default is the projection of like_filename.
        Possible values are anything parse-able by ``pyproj.CRS.from_user_input``
        (including EPSG ints, WKT strings, PROJ strings, etc.)
    nodata : float or str, optional
        Nodata value to save.
        Default is the nodata of band 1 of `like_filename` (if provided), or None.

    """
    if like_filename is not None:
        ds_like = gdal.Open(fspath(like_filename))
    else:
        ds_like = None

    xsize = ysize = gdal_dtype = None
    if arr is not None:
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        ysize, xsize = arr.shape[-2:]
        gdal_dtype = numpy_to_gdal_type(arr.dtype)
    else:
        # If not passing an array to write, get shape/dtype from like_filename
        if shape is not None:
            ysize, xsize = shape
        else:
            xsize, ysize = ds_like.RasterXSize, ds_like.RasterYSize
            # If using strides, adjust the output shape
            if strides is not None:
                ysize, xsize = compute_out_shape((ysize, xsize), strides)

        if dtype is not None:
            gdal_dtype = numpy_to_gdal_type(dtype)
        else:
            gdal_dtype = ds_like.GetRasterBand(1).DataType

    if any(v is None for v in (xsize, ysize, gdal_dtype)):
        raise ValueError("Must specify either `arr` or `like_filename`")

    if nodata is None and ds_like is not None:
        b = ds_like.GetRasterBand(1)
        nodata = b.GetNoDataValue()

    if nbands is None:
        if arr is not None:
            nbands = arr.shape[0]
        elif ds_like is not None:
            nbands = ds_like.RasterCount
        else:
            nbands = 1

    if driver is None:
        if str(output_name).endswith(".tif"):
            driver = "GTiff"
        else:
            if not ds_like:
                raise ValueError("Must specify `driver` if `like_filename` is None")
            driver = ds_like.GetDriver().ShortName
    if options is None and driver == "GTiff":
        options = list(DEFAULT_TIFF_OPTIONS)

    drv = gdal.GetDriverByName(driver)
    ds_out = drv.Create(
        fspath(output_name),
        xsize,
        ysize,
        nbands,
        gdal_dtype,
        options=options or [],
    )

    # If not provided, attempt to get projection/geotransform from like_filename
    if projection is None and ds_like is not None:
        projection = ds_like.GetProjection()
    if geotransform is None and ds_like is not None:
        geotransform = ds_like.GetGeoTransform()
        # If we're using strides, adjust the geotransform
        if strides is not None:
            geotransform = list(geotransform)
            geotransform[1] *= strides["x"]
            geotransform[5] *= strides["y"]

    # Set the geo/proj information
    if projection:
        # Make sure we're got a correct format for the projection
        # this still works if we're passed a WKT string
        projection = CRS.from_user_input(projection).to_wkt()
        ds_out.SetProjection(projection)

    if geotransform is not None:
        ds_out.SetGeoTransform(geotransform)

    # Write the actual data
    if arr is not None:
        for i in range(nbands):
            logger.debug(f"Writing band {i+1}/{nbands}")
            bnd = ds_out.GetRasterBand(i + 1)
            bnd.WriteArray(arr[i])

    # Set the nodata value for each band
    if nodata is not None:
        for i in range(nbands):
            logger.debug(f"Setting nodata for band {i+1}/{nbands}")
            bnd = ds_out.GetRasterBand(i + 1)
            bnd.SetNoDataValue(nodata)

    ds_out.FlushCache()
    ds_like = ds_out = None


def write_block(
    cur_block: ArrayLike,
    filename: Filename,
    row_start: int,
    col_start: int,
):
    """Write out an ndarray to a subset of the pre-made `filename`.

    Parameters
    ----------
    cur_block : ArrayLike
        2D or 3D data array
    filename : Filename
        List of output files to save to, or (if cur_block is 2D) a single file.
    row_start : int
        Row index to start writing at.
    col_start : int
        Column index to start writing at.

    Raises
    ------
    ValueError
        If length of `output_files` does not match length of `cur_block`.
    """
    if cur_block.ndim == 2:
        # Make into 3D array shaped (1, rows, cols)
        cur_block = cur_block[np.newaxis, ...]
    # filename must be pre-made
    if not Path(filename).exists():
        raise ValueError(f"File {filename} does not exist")

    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    for b_idx, cur_image in enumerate(cur_block, start=1):
        bnd = ds.GetRasterBand(b_idx)
        # only need offset for write:
        # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Band.WriteArray
        bnd.WriteArray(cur_image, col_start, row_start)
        bnd.FlushCache()
        bnd = None
    ds = None


class Writer(BackgroundWriter):
    """Class to write data to files in a background thread."""

    def __init__(self, max_queue: int = 0, **kwargs):
        super().__init__(nq=max_queue, name="Writer", **kwargs)

    def write(
        self, data: ArrayLike, filename: Filename, row_start: int, col_start: int
    ):
        """Write out an ndarray to a subset of the pre-made `filename`.

        Parameters
        ----------
        data : ArrayLike
            2D or 3D data array to save.
        filename : Filename
            List of output files to save to, or (if cur_block is 2D) a single file.
        row_start : int
            Row index to start writing at.
        col_start : int
            Column index to start writing at.

        Raises
        ------
        ValueError
            If length of `output_files` does not match length of `cur_block`.
        """
        write_block(data, filename, row_start, col_start)

    @property
    def num_queued(self):
        """Number of items waiting in the queue to be written."""
        return self._work_queue.qsize()


class EagerLoader(BackgroundReader):
    """Class to pre-fetch data chunks in a background thread."""

    def __init__(
        self,
        filename: Filename,
        block_shape: Tuple[int, int],
        overlaps: Tuple[int, int] = (0, 0),
        skip_empty: bool = True,
        nodata_mask: Optional[ArrayLike] = None,
        queue_size: int = 1,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        super().__init__(nq=queue_size, timeout=timeout, name="EagerLoader")
        self.filename = filename
        # Set up the generator of ((row_start, row_end), (col_start, col_end))
        xsize, ysize = get_raster_xysize(filename)
        # convert the slice generator to a list so we have the size
        self.slices = list(
            _slice_iterator(
                arr_shape=(ysize, xsize),
                block_shape=block_shape,
                overlaps=overlaps,
            )
        )
        self._queue_size = queue_size
        self._skip_empty = skip_empty
        self._nodata_mask = nodata_mask
        self._block_shape = block_shape
        self._nodata = get_raster_nodata(filename)
        if self._nodata is None:
            self._nodata = np.nan

    def read(self, rows: slice, cols: slice) -> Tuple[np.ndarray, Tuple[slice, slice]]:
        logger.debug(f"EagerLoader reading {rows}, {cols}")
        cur_block = load_gdal(self.filename, rows=rows, cols=cols)
        return cur_block, (rows, cols)

    def iter_blocks(
        self,
    ) -> Generator[Tuple[np.ndarray, Tuple[slice, slice]], None, None]:
        # Queue up all slices to the work queue
        for rows, cols in self.slices:
            self.queue_read(rows, cols)

        s_iter = range(len(self.slices))
        desc = f"Processing {self._block_shape} sized blocks..."
        with progress() as p:
            for _ in p.track(s_iter, description=desc):
                cur_block, (rows, cols) = self.get_data()
                logger.debug(f"got data for {rows, cols}: {cur_block.shape}")

                if self._skip_empty and self._nodata_mask is not None:
                    if self._nodata_mask[rows, cols].all():
                        continue

                if self._skip_empty:
                    # Otherwise look at the actual block we loaded
                    if np.isnan(self._nodata):
                        block_nodata = np.isnan(cur_block)
                    else:
                        block_nodata = cur_block == self._nodata
                    if np.all(block_nodata):
                        continue
                yield cur_block, (rows, cols)

        self.notify_finished()


def _slice_iterator(
    arr_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
    overlaps: Tuple[int, int] = (0, 0),
    start_offsets: Tuple[int, int] = (0, 0),
):
    """Create a generator to get indexes for accessing blocks of a raster.

    Parameters
    ----------
    arr_shape : Tuple[int, int]
        (num_rows, num_cols), full size of array to access
    block_shape : Tuple[int, int]
        (height, width), size of accessing blocks
    overlaps : Tuple[int, int]
        (row_overlap, col_overlap), number of pixels to re-include
        after sliding the block (default (0, 0))
    start_offsets : Tuple[int, int]
        Offsets to start reading from (default (0, 0))

    Yields
    ------
    Tuple[slice, slice]
        Iterator of (slice(row_start, row_stop), slice(col_start, col_stop))

    Examples
    --------
        >>> list(_slice_iterator((180, 250), (100, 100)))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(100, 200, None)), (slice(0, 100, None), slice(200, 250, None)), \
(slice(100, 180, None), slice(0, 100, None)), (slice(100, 180, None), \
slice(100, 200, None)), (slice(100, 180, None), slice(200, 250, None))]
        >>> list(_slice_iterator((180, 250), (100, 100), overlaps=(10, 10)))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]
    """
    rows, cols = arr_shape
    row_off, col_off = start_offsets
    row_overlap, col_overlap = overlaps
    height, width = block_shape

    if height is None:
        height = rows
    if width is None:
        width = cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height:
        raise ValueError(f"row_overlap {row_overlap} must be less than {height}")
    if col_overlap >= width:
        raise ValueError(f"col_overlap {col_overlap} must be less than {width}")
    while row_off < rows:
        while col_off < cols:
            row_end = min(row_off + height, rows)  # Dont yield something OOB
            col_end = min(col_off + width, cols)
            yield (slice(row_off, row_end), slice(col_off, col_end))

            col_off += width
            if col_off < cols:  # dont bring back if already at edge
                col_off -= col_overlap

        row_off += height
        if row_off < rows:
            row_off -= row_overlap
        col_off = 0


def get_max_block_shape(
    filename: Filename, nstack: int, max_bytes: float = 64e6
) -> Tuple[int, int]:
    """Find a block shape to load from `filename` with memory size < `max_bytes`.

    Attempts to get an integer number of chunks ("tiles" for geotiffs) from the
    file to avoid partial tiles.

    Parameters
    ----------
    filename : str
        GDAL-readable file name containing 3D dataset.
    nstack: int
        Number of bands in dataset.
    max_bytes : float, optional
        Target size of memory (in Bytes) for each block.
        Defaults to 64e6.

    Returns
    -------
    Tuple[int, int]:
        (num_rows, num_cols) shape of blocks to load from `vrt_file`
    """
    chunk_cols, chunk_rows = get_raster_chunk_size(filename)
    xsize, ysize = get_raster_xysize(filename)
    # If it's written by line, load at least 16 lines at a time
    chunk_cols = min(max(16, chunk_cols), xsize)
    chunk_rows = min(max(16, chunk_rows), ysize)

    ds = gdal.Open(fspath(filename))
    shape = (ds.RasterYSize, ds.RasterXSize)
    # get the size of the data type from the raster
    nbytes = gdal_to_numpy_type(ds.GetRasterBand(1).DataType).itemsize
    return _increment_until_max(
        max_bytes=max_bytes,
        file_chunk_size=[chunk_rows, chunk_cols],
        shape=shape,
        nstack=nstack,
        bytes_per_pixel=nbytes,
    )


def get_raster_chunk_size(filename: Filename) -> List[int]:
    """Get size the raster's chunks on disk.

    This is called blockXsize, blockYsize by GDAL.
    """
    ds = gdal.Open(fspath(filename))
    block_size = ds.GetRasterBand(1).GetBlockSize()
    for i in range(2, ds.RasterCount + 1):
        if block_size != ds.GetRasterBand(i).GetBlockSize():
            logger.warning(f"Warning: {filename} bands have different block shapes.")
            break
    return block_size


def _format_date_pair(start: date, end: date, fmt=DEFAULT_DATETIME_FORMAT) -> str:
    return f"{start.strftime(fmt)}_{end.strftime(fmt)}"


def _increment_until_max(
    max_bytes: float,
    file_chunk_size: Sequence[int],
    shape: Tuple[int, int],
    nstack: int,
    bytes_per_pixel: int = 8,
) -> Tuple[int, int]:
    """Find size of 3D chunk to load while staying at ~`max_bytes` bytes of RAM."""
    chunk_rows, chunk_cols = file_chunk_size

    # How many chunks can we fit in max_bytes?
    chunks_per_block = max_bytes / (
        (nstack * chunk_rows * chunk_cols) * bytes_per_pixel
    )
    num_chunks = [1, 1]
    cur_block_shape = [chunk_rows, chunk_cols]

    idx = 1  # start incrementing cols
    while chunks_per_block > 1 and tuple(cur_block_shape) != tuple(shape):
        # Alternate between adding a row and column chunk by flipping the idx
        chunk_idx = idx % 2
        nc = num_chunks[chunk_idx]
        chunk_size = file_chunk_size[chunk_idx]

        cur_block_shape[chunk_idx] = min(nc * chunk_size, shape[chunk_idx])

        chunks_per_block = max_bytes / (
            nstack * np.prod(cur_block_shape) * bytes_per_pixel
        )
        num_chunks[chunk_idx] += 1
        idx += 1
    return cur_block_shape[0], cur_block_shape[1]
