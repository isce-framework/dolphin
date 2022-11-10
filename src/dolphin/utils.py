import copy
import datetime
import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Tuple, Union

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


def copy_projection(src_file: Pathlike, dst_file: Pathlike) -> None:
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


def save_arr_like(*, arr, like_filename, output_name, driver="GTiff"):
    """Save an array to a file, copying projection/nodata from `like_filename`."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    ysize, xsize = arr.shape[-2:]
    nbands = arr.shape[0]
    gdal.UseExceptions()
    ds = gdal.Open(fspath(like_filename))
    if driver is None:
        driver = ds.GetDriver().ShortName
    drv = gdal.GetDriverByName(driver)
    out_ds = drv.Create(
        fspath(output_name),
        xsize,
        ysize,
        nbands,
        numpy_to_gdal_type(arr.dtype),
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    for i in range(nbands):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[i])
    # TODO: copy other metadata
    ds = out_ds = None


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


def load_gdal(filename, band=None):
    """Load a gdal file into a numpy array."""
    ds = gdal.Open(fspath(filename))
    return ds.ReadAsArray() if band is None else ds.GetRasterBand(band).ReadAsArray()


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


def take_looks(arr, row_looks, col_looks, func_type="nansum"):
    """Downsample a numpy matrix by summing blocks of (row_looks, col_looks).

    Parameters
    ----------
    arr : np.array
        2D array of an image
    row_looks : int
        the reduction rate in row direction
    col_looks : int
        the reduction rate in col direction
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
    try:
        import cupy as cp

        xp = cp.get_array_module(arr)
    except ImportError:
        print("cupy not installed, using numpy")
        xp = np

    if row_looks == 1 and col_looks == 1:
        return arr
    if arr.ndim >= 3:
        return xp.stack([take_looks(a, row_looks, col_looks) for a in arr])

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


def iter_blocks(
    filename,
    block_shape: Tuple[int, int],
    band=None,
    overlaps: Tuple[int, int] = (0, 0),
    start_offsets: Tuple[int, int] = (0, 0),
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

    Yields
    ------
    Iterator of data from slices
    """
    import rasterio as rio
    from rasterio.windows import Window

    with rio.open(filename) as src:
        block_iter = slice_iterator(
            src.shape,
            block_shape,
            overlaps=overlaps,
            start_offsets=start_offsets,
        )
        for win_slice in block_iter:
            window = Window.from_slices(*win_slice)
            yield src.read(band, window=window)


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
    Iterator of ((row_start, row_stop), (col_start, col_stop))

    Examples
    --------
    >>> list(slice_iterator((180, 250), (100, 100)))
    [((0, 100), (0, 100)), ((0, 100), (100, 200)), ((0, 100), (200, 250)), \
((100, 180), (0, 100)), ((100, 180), (100, 200)), ((100, 180), (200, 250))]
    >>> list(slice_iterator((180, 250), (100, 100), overlaps=(10, 10)))
    [((0, 100), (0, 100)), ((0, 100), (90, 190)), ((0, 100), (180, 250)), \
((90, 180), (0, 100)), ((90, 180), (90, 190)), ((90, 180), (180, 250))]
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
            yield ((row_off, row_end), (col_off, col_end))

            col_off += width
            if col_off < cols:  # dont bring back if already at edge
                col_off -= col_overlap

        row_off += height
        if row_off < rows:
            row_off -= row_overlap
        col_off = 0


def get_max_block_shape(filename, nstack: int, max_bytes=100e6):
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
        Defaults to 100e6.

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
    return cur_block_shape[-2:]


def get_block_size(filename):
    """Get the raster's (blockXsize, blockYsize) on disk."""
    ds = gdal.Open(fspath(filename))
    block_size = ds.GetRasterBand(1).GetBlockSize()
    for i in range(2, ds.RasterCount + 1):
        if block_size != ds.GetRasterBand(i).GetBlockSize():
            print(f"Warning: {filename} bands have different block shapes.")
            break
    return block_size
