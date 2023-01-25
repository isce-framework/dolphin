import copy
from datetime import date
from os import fspath
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from osgeo import gdal
from pyproj import CRS

from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.utils import gdal_to_numpy_type, numpy_to_gdal_type

gdal.UseExceptions()
logger = get_log()


DEFAULT_TILE_SIZE = (128, 128)
DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    f"BLOCKXSIZE={DEFAULT_TILE_SIZE[1]}",
    f"BLOCKYSIZE={DEFAULT_TILE_SIZE[0]}",
)
DEFAULT_HDF5_OPTIONS = dict(
    # https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    chunks=DEFAULT_TILE_SIZE,
    compression="gzip",
    compression_opts=4,
    shuffle=True,
)
DEFAULT_DATETIME_FORMAT = "%Y%m%d"
# Specific to opera CSLC products:
OPERA_DATASET_NAME = "science/SENTINEL1/CSLC/grids/VV"


def get_raster_xysize(filename: Filename) -> Tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def load_gdal(
    filename: Filename, band: Optional[int] = None, subsample_factor: int = 1
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

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.
    """
    ds = gdal.Open(fspath(filename))
    rows, cols = ds.RasterYSize, ds.RasterXSize
    # Make an output object of the right size
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty(
            (count, rows // subsample_factor, cols // subsample_factor), dtype=dt
        )
        ds.ReadAsArray(buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((rows // subsample_factor, cols // subsample_factor), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(buf_obj=out, resample_alg=resamp)
    return out


def format_nc_filename(filename: Filename, ds_name: Optional[str] = None) -> str:
    """Format an HDF5/NetCDF filename, with dataset for reading using GDAL."""
    # If we've already formatted the filename, return it
    if str(filename).startswith("NETCDF:") or str(filename).startswith("HDF5:"):
        return str(filename)

    if not (fspath(filename).endswith(".nc") or fspath(filename).endswith(".h5")):
        return fspath(filename)

    if ds_name is None:
        return _guess_gdal_dataset(filename)
    else:
        return f'NETCDF:"{filename}":"//{ds_name.lstrip("/")}"'


def _guess_gdal_dataset(filename: Filename) -> str:
    """Guess the GDAL dataset from a NetCDF/HDF5 filename.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.

    Returns
    -------
    ds : str
        GDAL dataset.
    """
    logger.debug(
        "No dataset name specified for %s, guessing from file contents", filename
    )
    info = gdal.Info(fspath(filename), format="json")
    if len(info["bands"]) > 0:
        # This means that gdal already found bands to read with just `filename`
        # so try that
        return fspath(filename)
    # Otherwise, check if it found subdatasets
    sds = info.get("metadata", {}).get("SUBDATASETS", {})
    if not sds:
        raise ValueError(f"No subdatasets found in {filename}")
    # {'SUBDATASET_1_NAME': 'HDF5:"t087_185682_iw2_20180306_VV.h5"://SLC/VV',
    #  'SUBDATASET_1_DESC': '[4720x20220] //SLC/VV (complex, 32-bit floating-point)', ...
    for i in range(len(sds) // 2):
        k = f"SUBDATASET_{i+1}_NAME"
        d = f"SUBDATASET_{i+1}_DESC"
        if "complex" in sds[d]:
            return sds[k].replace("HDF5:", "NETCDF:")
    raise ValueError(f"No complex subdatasets found in {filename}")


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


def get_nodata(filename: Filename) -> Optional[float]:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.
    """
    ds = gdal.Open(fspath(filename))
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    return nodata


def get_dtype(filename: Filename) -> np.dtype:
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
        row = round(row)
        col = round(col)
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
    arr: Optional[np.ndarray],
    output_name: Filename,
    like_filename: Optional[Filename] = None,
    driver: Optional[str] = "GTiff",
    options: Optional[List] = None,
    nbands: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
    dtype: Optional[Union[str, np.dtype, type]] = None,
    geotransform: Optional[Sequence[float]] = None,
    projection: Optional[Any] = None,
    nodata: Optional[Union[float, str]] = None,
):
    """Save an array to `output_name`.

    If `like_filename` if provided, copies the projection/nodata.
    Options can be overridden by passing `driver`/`nbands`/`dtype`.

    If arr is None, create an empty file with the same x/y shape as `like_filename`.

    Parameters
    ----------
    arr : np.ndarray, optional
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
    dtype : str or np.dtype or type, optional
        Data type to save. Default is `arr.dtype` or the datatype of like_filename.
    geotransform : List, optional
        Geotransform to save. Default is the geotransform of like_filename.
        See https://gdal.org/tutorials/geotransforms_tut.html .
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
        if shape is not None:
            ysize, xsize = shape
        else:
            xsize, ysize = ds_like.RasterXSize, ds_like.RasterYSize
        if dtype is not None:
            gdal_dtype = numpy_to_gdal_type(dtype)
        else:
            gdal_dtype = ds_like.GetRasterBand(1).DataType

    if any(v is None for v in (xsize, ysize, gdal_dtype)):
        raise ValueError("Must specify either `arr` or `like_filename`")

    if nodata is None and ds_like is not None:
        b = ds_like.GetRasterBand(1)
        nodata = b.GetNoDataValue()

    nbands = nbands or (ds_like.RasterCount if ds_like else arr.shape[0])

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
            print(f"Writing band {i+1}/{nbands}")
            bnd = ds_out.GetRasterBand(i + 1)
            bnd.WriteArray(arr[i])
            if nodata is not None:
                bnd.SetNoDataValue(nodata)

    ds_out.FlushCache()
    ds_like = ds_out = None


def write_block(
    cur_block: np.ndarray,
    filename: Filename,
    row_start: int,
    col_start: int,
):
    """Write out an ndarray to a subset of the pre-made `filename`.

    Parameters
    ----------
    cur_block : np.ndarray
        Array of shape (n_bands, block_rows, block_cols)
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


def _format_date_pair(start: date, end: date, fmt=DEFAULT_DATETIME_FORMAT) -> str:
    return f"{start.strftime(fmt)}_{end.strftime(fmt)}"


def get_stack_nodata_mask(
    stack_filename: Filename,
    output_file: Optional[Filename] = None,
    compute_bands: Optional[List[int]] = None,
    buffer_pixels: int = 100,
    nodata: float = np.nan,
):
    """Get a mask of pixels that are nodata in all bands of `slc_stack_vrt`.

    Parameters
    ----------
    stack_filename : Path or str
        File containing the SLC stack as separate bands.
    output_file : Path or str, optional
        Name of file to save to., by default None
    compute_bands : List[int], optional
        List of bands in vrt_stack to read.
        If None, reads in the first, middle, and last images.
    buffer_pixels : int, optional
        Number of pixels to expand the good-data area, by default 100
    nodata : float, optional
        Value of no data in the vrt_stack, by default np.nan

    Returns
    -------
    mask : np.ndarray[bool]
        Array where True indicates all bands are nodata.
    """
    ds = gdal.Open(fspath(stack_filename))
    if compute_bands is None:
        count = ds.RasterCount
        # Get the first and last file
        compute_bands = list(sorted(set([1, count])))

    # Start with ones, then only keep pixels that are nodata
    # in all the bands we check (reducing using logical_and)
    out_mask = np.ones((ds.RasterYSize, ds.RasterXSize), dtype=bool)

    # cap buffer pixel length to be no more the image size
    buffer_pixels = min(buffer_pixels, min(ds.RasterXSize, ds.RasterYSize))
    for b in compute_bands:
        print(f"Computing mask for band {b}")
        bnd = ds.GetRasterBand(b)
        arr = bnd.ReadAsArray()
        if np.isnan(nodata):
            nodata_mask = np.isnan(arr)
        else:
            nodata_mask = arr == nodata

        # Expand the region with a convolution
        if buffer_pixels > 0:
            print(f"Padding mask with {buffer_pixels} pixels")
            out_mask &= _erode_nodata(nodata_mask, buffer_pixels)
        else:
            out_mask &= nodata_mask

    if output_file:
        write_arr(
            arr=out_mask,
            output_name=output_file,
            like_filename=stack_filename,
            nbands=1,
            dtype="Byte",
        )
    return out_mask


def _erode_nodata(nd_mask, buffer_pixels=25):
    """Erode the nodata mask by `buffer_pixels`.

    This makes the nodata mask more conservative:
    there will be fewer pixels marked as nodata after.

    Parameters
    ----------
    nd_mask : np.ndarray[bool]
        Array where True indicates nodata.
    buffer_pixels : int, optional
        Size (in pixels) of erosion structural element to use.
        By default 25.

    Returns
    -------
    np.ndarray[bool]
        Same size as `nd_mask`, with no data pixels shrunk
        after erosion.
    """
    # invert so that good pixels are 1
    # we want to expand the area that is considered "good"
    # since we're being conservative with what we completely ignore
    out = (~nd_mask).astype("float32").copy()
    strel = np.ones(buffer_pixels)
    for i in range(out.shape[0]):
        o = np.convolve(out[i, :], strel, mode="same")
        out[i, :] = o
    for j in range(out.shape[1]):
        o = np.convolve(out[:, j], strel, mode="same")
        out[:, j] = o
    # convert back to binary mask, and re-invert
    return ~(out > 1e-3)


def iter_blocks(
    filename,
    block_shape: Tuple[int, int],
    band=None,
    overlaps: Tuple[int, int] = (0, 0),
    start_offsets: Tuple[int, int] = (0, 0),
    return_slices: bool = False,
    skip_empty: bool = True,
    nodata: float = np.nan,
    nodata_mask: Optional[np.ndarray] = None,
):
    """Read blocks of a raster as a generator.

    Parameters
    ----------
    filename : str or Path
        path to raster file
    block_shape : tuple[int, int]
        (height, width), size of accessing blocks (default (None, None))
    band : int, optional
        band to read (default None, whole stack)
    overlaps : tuple[int, int], optional
        (row_overlap, col_overlap), number of pixels to re-include
        after sliding the block (default (0, 0))
    start_offsets : tuple[int, int], optional
        (row_start, col_start), start reading from this offset
    return_slices : bool, optional (default False)
        return the (row, col) slice indicating the position of the current block
    skip_empty : bool, optional (default True)
        Skip blocks that are entirely empty (all NaNs)
    nodata : float, optional (default np.nan)
        Value to use for nodata to determine if a block is empty.
        Not used if `skip_empty` is False.
    nodata_mask : ndarray, optional
        A boolean mask of the same shape as the raster, where True indicates
        nodata. Ignored if `skip_empty` is False.
        If provided, `nodata` is ignored.

    Yields
    ------
    ndarray:
        Current block being loaded

    tuple[slice, slice]:
        ((row_start, row_end), (col_start, col_end)) slice indicating
        the position of the current block.
        (Only returned if return_slices is True)
    """
    ds = gdal.Open(fspath(filename))
    if band is None:
        # Read all bands
        read_func = ds.ReadAsArray
    else:
        # Read from single band
        read_func = ds.GetRasterBand(band).ReadAsArray

    # Set up the generator of ((row_start, row_end), (col_start, col_end))
    slice_gen = slice_iterator(
        (ds.RasterYSize, ds.RasterXSize),
        block_shape,
        overlaps=overlaps,
        start_offsets=start_offsets,
    )
    for rows, cols in slice_gen:
        # Skip empty blocks before reading if we have a nodata mask
        if skip_empty and nodata_mask is not None:
            if nodata_mask[rows, cols].all():
                continue
        xoff = cols.start
        yoff = rows.start
        xsize = cols.stop - cols.start
        ysize = rows.stop - rows.start
        cur_block = read_func(
            xoff,
            yoff,
            xsize,
            ysize,
        )
        if skip_empty:
            # Otherwise look at the actual block we loaded
            if np.isnan(nodata):
                block_nodata = np.isnan(cur_block)
            else:
                block_nodata = cur_block == nodata
            if np.all(block_nodata):
                continue

        if return_slices:
            yield cur_block, (rows, cols)
        else:
            yield cur_block
    ds = None


def slice_iterator(
    arr_shape,
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
        >>> list(slice_iterator((180, 250), (100, 100)))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(100, 200, None)), (slice(0, 100, None), slice(200, 250, None)), \
(slice(100, 180, None), slice(0, 100, None)), (slice(100, 180, None), \
slice(100, 200, None)), (slice(100, 180, None), slice(200, 250, None))]
        >>> list(slice_iterator((180, 250), (100, 100), overlaps=(10, 10)))
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
    filename, nstack: int, max_bytes: float = 64e6
) -> Tuple[int, int]:
    """Find shape to load from GDAL-readable `filename` with memory size < `max_bytes`.

    Attempts to get an integer number of tiles from the file to avoid partial tiles.

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
    tuple[int]:
        (num_rows, num_cols) shape of blocks to load from `vrt_file`
    """
    blockX, blockY = get_raster_block_size(filename)
    xsize, ysize = get_raster_xysize(filename)
    # If it's written by line, load at least 16 lines at a time
    blockX = min(max(16, blockX), xsize)
    blockY = min(max(16, blockY), ysize)

    ds = gdal.Open(fspath(filename))
    shape = (ds.RasterYSize, ds.RasterXSize)
    # get the data type from the raster
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)
    # get the size of the data type
    nbytes = np.dtype(dt).itemsize

    full_shape = [nstack, *shape]
    chunk_size_3d = [nstack, blockY, blockX]

    # Find size of 3D chunk to load while staying at ~`max_bytes` bytes of RAM
    chunks_per_block = max_bytes / (np.prod(chunk_size_3d) * nbytes)
    row_chunks, col_chunks = 1, 1
    cur_block_shape = list(copy.copy(chunk_size_3d))
    while chunks_per_block > 1:
        # First keep incrementing the number of columns we grab at once time
        if col_chunks * chunk_size_3d[2] < full_shape[2]:
            col_chunks += 1
            cur_block_shape[2] = min(col_chunks * chunk_size_3d[2], full_shape[2])
        # Then increase the row size if still haven't hit `max_bytes`
        elif row_chunks * chunk_size_3d[1] < full_shape[1]:
            row_chunks += 1
            cur_block_shape[1] = min(row_chunks * chunk_size_3d[1], full_shape[1])
        else:
            break
        chunks_per_block = max_bytes / (np.prod(cur_block_shape) * nbytes)
    rows, cols = cur_block_shape[1:]
    return (rows, cols)


def get_raster_block_size(filename):
    """Get the raster's (blockXsize, blockYsize) on disk."""
    ds = gdal.Open(fspath(filename))
    block_size = ds.GetRasterBand(1).GetBlockSize()
    for i in range(2, ds.RasterCount + 1):
        if block_size != ds.GetRasterBand(i).GetBlockSize():
            print(f"Warning: {filename} bands have different block shapes.")
            break
    return block_size
