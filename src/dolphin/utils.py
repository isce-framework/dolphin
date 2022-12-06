import datetime
import re
from os import PathLike, fspath
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from dolphin._log import get_log

Filename = Union[str, PathLike[str]]
gdal.UseExceptions()
logger = get_log()


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


def get_dates(filename: Filename) -> List[Union[None, str]]:
    """Search for dates (YYYYMMDD) in `filename`, excluding path."""
    date_list = re.findall(r"\d{4}\d{2}\d{2}", Path(filename).stem)
    if not date_list:
        msg = f"{filename} does not contain date as YYYYMMDD"
        logger.warning(msg)
        # raise ValueError(msg)
    return date_list


def parse_slc_strings(slc_str: Union[Filename, List[Filename]], fmt="%Y%m%d"):
    """Parse a string, or list of strings, matching `fmt` into datetime.date.

    Parameters
    ----------
    slc_str : str or list of str
        String or list of strings to parse.
    fmt : str, optional
        Format of string to parse. Default is "%Y%m%d".

    Returns
    -------
    datetime.date, or list of datetime.date
    """

    def _parse(datestr, fmt="%Y%m%d") -> datetime.date:
        return datetime.datetime.strptime(datestr, fmt).date()

    # The re.search will find YYYYMMDD anywhere in string
    if isinstance(slc_str, str) or hasattr(slc_str, "__fspath__"):
        d_str = get_dates(slc_str)
        if not d_str:
            raise ValueError(f"Could not find date of format {fmt} in {slc_str}")
            # return None
        return _parse(d_str[0], fmt=fmt)
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s, fmt=fmt) for s in slc_str if s]


def rowcol_to_xy(row, col, ds=None, filename=None):
    """Convert indexes in the image space to georeferenced coordinates."""
    return _apply_gt(ds, filename, col, row)


def xy_to_rowcol(x, y, ds=None, filename=None):
    """Convert coordinates in the georeferenced space to a row and column index."""
    return _apply_gt(ds, filename, x, y, inverse=True)


def _apply_gt(ds=None, filename=None, x=None, y=None, inverse=False):
    """Read the (possibly inverse) geotransform, apply to the x/y coordinates."""
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


def combine_mask_files(
    mask_files: List[Filename],
    scratch_dir: Filename,
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


def get_raster_xysize(filename: Filename) -> Tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def full_suffix(filename: Filename):
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


def half_window_to_full(half_window: Union[List, Tuple]) -> Tuple[int, int]:
    """Convert a half window size to a full window size."""
    return (2 * half_window[0] + 1, 2 * half_window[1] + 1)


def check_gpu_available() -> bool:
    """Check if a GPU is available."""
    try:
        # cupy not available on Mac m1
        import cupy as cp  # noqa: F401
        from numba import cuda

    except ImportError:
        logger.debug("numba/cupy installed, but GPU not available")
        return False
    try:
        cuda_version = cuda.runtime.get_version()
        logger.debug(f"CUDA version: {cuda_version}")
    except OSError as e:
        logger.debug(f"CUDA runtime version error: {e}")
        return False
    try:
        n_gpu = len(cuda.gpus)
    except cuda.CudaSupportError as e:
        logger.debug(f"CUDA support error {e}")
        return False
    if n_gpu < 1:
        logger.debug("No available GPUs found")
        return False
    return True


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
        xp = np
    return xp


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
    xp = get_array_module(arr)

    if row_looks == 1 and col_looks == 1:
        return arr

    if arr.ndim >= 3:
        return xp.stack([take_looks(a, row_looks, col_looks, func_type) for a in arr])
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
