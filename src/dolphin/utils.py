import datetime
import re
import warnings
from os import fspath
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from osgeo import gdal, gdal_array, gdalconst

from dolphin._log import get_log
from dolphin._types import Filename

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


def get_dates(filename: Filename, fmt="%Y%m%d") -> List[Union[None, str]]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : str or PathLike
        Filename to search for dates.
    fmt : str, optional
        Format of date to search for. Default is "%Y%m%d".

    Returns
    -------
    list[str] or None
        List of dates found in the stem of `filename` matching `fmt`.
        Returns None if nothing is found.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    ['20191231']
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    ['20191231', '20191231']
    >>> get_dates("/not/a/date_named_file.tif")
    []
    """  # noqa: E501
    pat = _date_format_to_regex(fmt)
    date_list = re.findall(pat, Path(filename).stem)
    if not date_list:
        msg = f"{filename} does not contain date as YYYYMMDD"
        logger.warning(msg)
        return []
    return date_list


def parse_slc_strings(slc_str: Union[Filename, Sequence[Filename]], fmt=None):
    """Parse a string, or list of strings, matching `fmt` into datetime.date.

    Parameters
    ----------
    slc_str : str or list of str
        String or list of strings to parse.
    fmt : str, or List[str]. Optional
        Format of string to parse.
        If None (default), searches for "%Y%m%d" or "%Y-%m-%d".

    Returns
    -------
    datetime.date, or list of datetime.date
    """

    def _parse(datestr, fmt="%Y%m%d") -> datetime.date:
        return datetime.datetime.strptime(datestr, fmt).date()

    if fmt is None:
        fmt = ["%Y%m%d", "%Y-%m-%d"]
    elif isinstance(fmt, str):
        fmt = [fmt]

    if isinstance(slc_str, str) or hasattr(slc_str, "__fspath__"):
        # Unpack all returned dates from each format
        d_list = []
        fmt_found = None
        for f in fmt:
            d_list.extend(get_dates(slc_str, fmt=f))
            if len(d_list) > 0:
                fmt_found = f
                break
        else:  # if we iterate through all formats and don't find any dates
            raise ValueError(f"Could not find date of format {fmt} in {slc_str}")

        # if len(unique_dates) > 1:
        #     raise ValueError(
        #         f"Found multiple dates in {slc_str}: {unique_dates}. "
        #         "Please specify a date format."
        #     )
        return _parse(d_list[0], fmt=fmt_found)
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slc_strings(s, fmt=fmt) for s in slc_str if s]


def _date_format_to_regex(date_format):
    r"""Convert a python date format string to a regular expression.

    Useful for Year, month, date date formats.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. "%Y%m%d"

    Returns
    -------
    re.Pattern
        Regular expression that matches the date format string.

    Examples
    --------
    >>> pat2 = _date_format_to_regex("%Y%m%d").pattern
    >>> pat2 == re.compile(r'\d{4}\d{2}\d{2}').pattern
    True
    >>> pat = _date_format_to_regex("%Y-%m-%d").pattern
    >>> pat == re.compile(r'\d{4}\-\d{2}\-\d{2}').pattern
    True
    """
    # Escape any special characters in the date format string
    date_format = re.escape(date_format)

    # Replace each format specifier with a regular expression that matches it
    date_format = date_format.replace("%Y", r"\d{4}")
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")

    # Return the resulting regular expression
    return re.compile(date_format)


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


def gpu_is_available() -> bool:
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


def take_looks(arr, row_looks, col_looks, func_type="nansum", edge_strategy="cutoff"):
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
    edge_strategy : str, optional
        how to handle edges, by default "cutoff"
        Choices are "cutoff", "pad"

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

    arr = _make_dims_multiples(arr, row_looks, col_looks, how=edge_strategy)

    rows, cols = arr.shape
    new_rows = rows // row_looks
    new_cols = cols // col_looks

    func = getattr(xp, func_type)
    with warnings.catch_warnings():
        # ignore the warning about nansum of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(
            xp.reshape(arr, (new_rows, row_looks, new_cols, col_looks)), axis=(3, 1)
        )


def _make_dims_multiples(arr, row_looks, col_looks, how="cutoff"):
    """Pad or cutoff an array to make the dimensions multiples of the looks."""
    rows, cols = arr.shape
    row_cutoff = rows % row_looks
    col_cutoff = cols % col_looks
    if how == "cutoff":
        if row_cutoff != 0:
            arr = arr[:-row_cutoff, :]
        if col_cutoff != 0:
            arr = arr[:, :-col_cutoff]
        return arr
    elif how == "pad":
        pad_rows = (row_looks - row_cutoff) % row_looks
        pad_cols = (col_looks - col_cutoff) % col_looks
        pad_val = False if arr.dtype == bool else np.nan
        if pad_rows > 0 or pad_cols > 0:
            arr = np.pad(
                arr,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=pad_val,
            )
        return arr
    else:
        raise ValueError(f"Invalid edge strategy: {how}")


def upsample_nearest(
    arr: np.ndarray,
    output_shape: Tuple[int, int],
    looks: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Upsample a numpy matrix by repeating blocks of (row_looks, col_looks).

    Parameters
    ----------
    arr : np.array
        2D or 3D downsampled array.
    output_shape : Tuple[int, int]
        The desired output shape.
    looks : Tuple[int, int]
        The number of looks in the row and column directions.
        If not provided, will be calculated from `output_shape`.

    Returns
    -------
    ndarray
        The upsampled array, shape = `output_shape`.

    Notes
    -----
    Will use cupy if available and if `arr` is a cupy array on the GPU.
    """
    xp = get_array_module(arr)
    in_rows, in_cols = arr.shape[-2:]
    out_rows, out_cols = output_shape[-2:]
    if (in_rows, in_cols) == (out_rows, out_cols):
        return arr

    if looks is None:
        row_looks = out_rows // in_rows
        col_looks = out_cols // in_cols
    else:
        row_looks, col_looks = looks

    arr_up = xp.repeat(xp.repeat(arr, row_looks, axis=-2), col_looks, axis=-1)
    # This may be larger than the original array, or it may be smaller, depending
    # on whether it was padded or cutoff
    out_r = min(out_rows, arr_up.shape[-2])
    out_c = min(out_cols, arr_up.shape[-1])

    shape = (len(arr), out_rows, out_cols) if arr.ndim == 3 else (out_rows, out_cols)
    arr_out = xp.zeros(shape=shape, dtype=arr.dtype)
    arr_out[..., :out_r, :out_c] = arr_up[..., :out_r, :out_c]
    return arr_out
