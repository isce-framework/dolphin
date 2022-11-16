import copy
import datetime
import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from dolphin.log import get_log

Pathlike = Union[PathLike[str], str]
gdal.UseExceptions()
logger = get_log()


def get_dates(filename: Pathlike) -> List[Union[None, str]]:
    """Search for dates (YYYYMMDD) in `filename`, excluding path."""
    date_list = re.findall(r"\d{4}\d{2}\d{2}", Path(filename).stem)
    if not date_list:
        raise ValueError(f"{filename} does not contain date as YYYYMMDD")
    return date_list


def numpy_to_gdal_type(np_dtype):
    """Convert numpy dtype to gdal type."""
    # Wrap in np.dtype in case string is passed
    if isinstance(np_dtype, str):
        np_dtype = np.dtype(np_dtype.lower())
    elif isinstance(np_dtype, type):
        np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    return gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)


def gdal_to_numpy_type(gdal_type):
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type)


def parse_slc_strings(slc_str):
    """Parse a string, or list of strings, with YYYYmmdd as date."""
    # The re.search will find YYYYMMDD anywhere in string
    if isinstance(slc_str, str):
        match = re.search(r"\d{8}", slc_str)
        if not match:
            raise ValueError(f"{slc_str} does not contain date as YYYYMMDD")
        return _parse(match.group())
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s) for s in slc_str if s]


def _parse(datestr):
    return datetime.datetime.strptime(datestr, "%Y%m%d").date()


def combine_mask_files(
    mask_files: List[Pathlike],
    scratch_dir: Pathlike,
    output_file_name: str = "combined_mask.tif",
    dtype: str = "uint8",
    zero_is_valid: bool = False,
) -> Path:
    """Combine multiple mask files into a single mask file.

    Parameters
    ----------
    mask_files : list of Path or str
        List of mask files to combine.
    scratch_dir : Path or str
        Directory to write output file.
    output_file_name : str
        Name of output file to write into `scratch_dir`
    dtype : str, optional
        Data type of output file. Default is uint8.
    zero_is_valid : bool, optional
        If True, zeros mark the valid pixels (like numpy's masking convention).
        Default is False (matches ISCE convention).

    Returns
    -------
    output_file : Path
    """
    output_file = Path(scratch_dir) / output_file_name

    ds = gdal.Open(fspath(mask_files[0]))
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()

    if projection is None and geotransform is None:
        logger.warning("No projection or geotransform found on file %s", mask_files[0])

    nodata = 1 if zero_is_valid else 0

    # Create output file
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        fspath(output_file),
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        numpy_to_gdal_type(dtype),
    )
    ds_out.SetGeoTransform(geotransform)
    ds_out.SetProjection(projection)
    ds_out.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None

    # Loop through mask files and update the total mask (starts with all valid)
    mask_total = np.ones((ds.RasterYSize, ds.RasterXSize), dtype=bool)
    for mask_file in mask_files:
        ds_input = gdal.Open(fspath(mask_file))
        mask = ds_input.GetRasterBand(1).ReadAsArray().astype(bool)
        if zero_is_valid:
            mask = ~mask
        mask_total = np.logical_and(mask_total, mask)
        ds_input = None

    if zero_is_valid:
        mask_total = ~mask_total
    ds_out.GetRasterBand(1).WriteArray(mask_total.astype(dtype))
    ds_out = None

    return output_file


def get_raster_xysize(filename: Pathlike) -> Tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def full_suffix(filename: Pathlike):
    """Get the full suffix of a filename, including multiple dots.

    Parameters
    ----------
    filename : str or Path
        path to file

    Returns
    -------
    str
        The full suffix, including multiple dots.

    Examples
    --------
    >>> full_suffix('test.tif')
    '.tif'
    >>> full_suffix('test.tar.gz')
    '.tar.gz'
    """
    fpath = Path(filename)
    return "".join(fpath.suffixes)


def half_window_to_full(half_window):
    """Convert a half window to a full window."""
    return (2 * half_window[0] + 1, 2 * half_window[1] + 1)


def get_array_module(arr):
    """Get the array module (numpy or cupy) for the given array.

    References
    ----------
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code
    """
    try:
        import cupy as cp

        xp = cp.get_array_module(arr)
    except ImportError:
        logger.debug("cupy not installed, falling back to numpy")
    return xp


def take_looks(
    arr, row_looks, col_looks, row_stride=None, col_stride=None, func_type="nansum"
):
    """Downsample a numpy matrix by summing blocks of (row_looks, col_looks).

    Parameters
    ----------
    arr : np.array
        2D array of an image
    row_looks : int
        the reduction rate in row direction
    col_looks : int
        the reduction rate in col direction
    row_stride : int, optional
        the sliding rate in row direction. If None, equal to row_looks
    col_stride : int, optional
        the sliding rate in col direction. If None, equal to col_looks
    func_type : str, optional
        the numpy function to use for downsampling, by default "nansum"

    Returns
    -------
    ndarray
        The downsampled array, shape = ceil(rows / row_looks, cols / col_looks)

    Notes
    -----
    Cuts off values if the size isn't divisible by num looks.
    Will use cupy if available and if `arr` is a cupy array on the GPU.
    """
    xp = get_array_module(arr)

    if row_looks == 1 and col_looks == 1:
        return arr

    if row_stride is None:
        row_stride = row_looks
    if col_stride is None:
        col_stride = col_looks
    if (row_stride, col_stride) != (row_looks, col_looks):
        if xp != np:
            raise NotImplementedError(
                "Sliding looks is not implemented for cupy arrays yet."
            )
        return take_looks_bn(
            arr, row_looks, col_looks, row_stride, col_stride, func_type
        )

    if arr.ndim >= 3:
        return xp.stack(
            [
                take_looks(a, row_looks, col_looks, row_stride, col_stride, func_type)
                for a in arr
            ]
        )
    rows, cols = arr.shape
    new_rows = rows // row_looks
    new_cols = cols // col_looks

    row_cutoff = rows % row_looks
    col_cutoff = cols % col_looks

    if row_cutoff != 0:
        arr = arr[:-row_cutoff, :]
    if col_cutoff != 0:
        arr = arr[:, :-col_cutoff]

    func = getattr(xp, func_type)
    return func(
        xp.reshape(arr, (new_rows, row_looks, new_cols, col_looks)), axis=(3, 1)
    )


def take_looks_bn(
    arr, row_looks, col_looks, row_stride=None, col_stride=None, func_type="sum"
):
    """Multi-look window with different step sizes than look sizes."""
    import bottleneck as bn

    if row_stride is None:
        row_stride = row_looks
    if col_stride is None:
        col_stride = col_looks

    func_name = func_type.replace("nan", "")  # bottleneck always ignores nans
    if func_name not in ("sum", "mean", "median", "max"):
        raise ValueError(f"func_type {func_type} not supported")

    func = getattr(bn, f"move_{func_name}")

    # Pad at the end so we can get a centered mean
    r_pad = row_looks - row_looks // 2 - 1
    c_pad = col_looks - col_looks // 2 - 1
    # dont pad the earlier dimensions, just the last two we'll be multilooking
    pad_widths = (arr.ndim - 2) * ((0, 0),) + ((0, r_pad), (0, c_pad))
    arr0 = np.pad(arr, pad_width=pad_widths, mode="constant", constant_values=np.nan)
    # bottleneck doesn't support multi-axis, so we have to do it in two steps:
    # across cols
    a1 = func(arr0, col_looks, axis=-1, min_count=1)[..., :, c_pad:]
    # then rows
    a2 = func(a1, row_looks, axis=-2, min_count=1)[..., r_pad:, :]
    # note: if there are less than min_count non-nan values in the window, returns nan

    # if we dont pad:
    # return a2[..., (row_stride - 1) :: row_stride, (col_stride - 1) :: col_stride]
    r_start = row_stride // 2
    c_start = col_stride // 2
    return a2[..., r_start::row_stride, c_start::col_stride]


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
            if nodata_mask is not None:
                cur_mask = nodata_mask[rows, cols]
            elif np.isnan(nodata):
                cur_mask = np.isnan(cur_block)
            else:
                cur_mask = cur_block == nodata
            if np.all(cur_mask):
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
        number of bands in dataset.
    max_bytes : float, optional)
        target size of memory (in Bytes) for each block.
        Defaults to 64e6.

    Returns
    -------
    tuple[int]:
        (num_rows, num_cols) shape of blocks to load from `vrt_file`
    """
    blockX, blockY = get_block_size(filename)
    # If it's written by line, load at least 16 lines at a time
    blockX = max(16, blockX)
    blockY = max(16, blockY)

    ds = gdal.Open(fspath(filename))
    shape = (ds.RasterYSize, ds.RasterXSize)
    # get the data type from the raster
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)
    # get the size of the data type
    nbytes = np.dtype(dt).itemsize

    full_shape = [nstack, *shape]
    chunk_3d = [nstack, blockY, blockX]
    return _get_stack_block_shape(full_shape, chunk_3d, nbytes, max_bytes)


def _get_stack_block_shape(full_shape, chunk_size, nbytes, max_bytes):
    """Find size of 3D chunk to load while staying at ~`max_bytes` bytes of RAM."""
    chunks_per_block = max_bytes / (np.prod(chunk_size) * nbytes)
    row_chunks, col_chunks = 1, 1
    cur_block_shape = list(copy.copy(chunk_size))
    while chunks_per_block > 1:
        # First keep incrementing the number of columns we grab at once time
        if col_chunks * chunk_size[2] < full_shape[2]:
            col_chunks += 1
            cur_block_shape[2] = min(col_chunks * chunk_size[2], full_shape[2])
        # Then increase the row size if still haven't hit `max_bytes`
        elif row_chunks * chunk_size[1] < full_shape[1]:
            row_chunks += 1
            cur_block_shape[1] = min(row_chunks * chunk_size[1], full_shape[1])
        else:
            break
        chunks_per_block = max_bytes / (np.prod(cur_block_shape) * nbytes)
    return tuple(cur_block_shape[-2:])


def get_block_size(filename):
    """Get the raster's (blockXsize, blockYsize) on disk."""
    ds = gdal.Open(fspath(filename))
    block_size = ds.GetRasterBand(1).GetBlockSize()
    for i in range(2, ds.RasterCount + 1):
        if block_size != ds.GetRasterBand(i).GetBlockSize():
            print(f"Warning: {filename} bands have different block shapes.")
            break
    return block_size
