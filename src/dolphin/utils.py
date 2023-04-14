import datetime
import re
import resource
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from osgeo import gdal, gdal_array, gdalconst
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

from dolphin._log import get_log
from dolphin._types import Filename

gdal.UseExceptions()
logger = get_log(__name__)


def progress():
    """Create a Progress bar context manager.

    Usage
    -----
    >>> with progress() as p:
    ...     for i in p.track(range(10)):
    ...         pass
    10/10 Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    """
    return Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns()[:-1],  # Skip the ETA column
        TimeElapsedColumn(),
    )


def numpy_to_gdal_type(np_dtype: DTypeLike) -> int:
    """Convert numpy dtype to gdal type.

    Parameters
    ----------
    np_dtype : DTypeLike
        Numpy dtype to convert.

    Returns
    -------
    int
        GDAL type code corresponding to `np_dtype`.

    Raises
    ------
    TypeError
        If `np_dtype` is not a numpy dtype, or if the provided dtype is not
        supported by GDAL (for example, `np.dtype('>i4')`)
    """
    np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    gdal_code = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)
    if gdal_code is None:
        raise TypeError(f"dtype {np_dtype} not supported by GDAL.")
    return gdal_code


def gdal_to_numpy_type(gdal_type: Union[str, int]) -> np.dtype:
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def get_dates(filename: Filename, fmt: str = "%Y%m%d") -> List[datetime.date]:
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
    list[datetime.date]
        List of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.date(2019, 12, 31)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.date(2019, 12, 31), datetime.date(2019, 12, 31)]
    >>> get_dates("/not/a/date_named_file.tif")
    []
    """  # noqa: E501
    path = _get_path_from_gdal_str(filename)
    pattern = _date_format_to_regex(fmt)
    date_list = re.findall(pattern, path.stem)
    if not date_list:
        return []
    return [_parse_date(d, fmt) for d in date_list]


def _parse_date(datestr: str, fmt: str = "%Y%m%d") -> datetime.date:
    return datetime.datetime.strptime(datestr, fmt).date()


def _get_path_from_gdal_str(name: Filename) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        p = s.split(":")[1].strip('"').strip("'")
    else:
        return Path(name)
    return Path(p)


def _resolve_gdal_path(gdal_str: Filename) -> Filename:
    """Resolve the file portion of a gdal-openable string to an absolute path."""
    s = str(gdal_str)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        file_part = s.split(":")[-1]
        is_gdal_str = True
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":slc_var
        file_part = s.split(":")[1]
        is_gdal_str = True
    else:
        file_part = s
        is_gdal_str = False

    # strip quotes to add back in after
    file_part = file_part.strip('"').strip("'")
    file_part_resolved = Path(file_part).resolve()
    resolved = s.replace(file_part, str(file_part_resolved))
    return Path(resolved) if not is_gdal_str else resolved


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


def sort_files_by_date(
    files: Iterable[Filename], file_date_fmt: str = "%Y%m%d"
) -> Tuple[List[Filename], List[List[datetime.date]]]:
    """Sort a list of files by date.

    If some files have multiple dates, the files with the most dates are sorted
    first. Within each group of files with the same number of dates, the files
    with the earliest dates are sorted first.

    The multi-date files are placed first so that compressed SLCs are sorted
    before the individual SLCs that make them up.

    Parameters
    ----------
    files : Iterable[Filename]
        List of files to sort.
    file_date_fmt : str, optional
        Datetime format passed to `strptime`, by default "%Y%m%d"

    Returns
    -------
    file_list : List[Filename]
        List of files sorted by date.
    dates : List[List[datetime.date,...]]
        Sorted list, where each entry has all the dates from the corresponding file.
    """

    def sort_key(file_date_tuple):
        # Key for sorting:
        # To sort the files with the most dates first (the compressed SLCs which
        # span a date range), sort the longer date lists first.
        # Then, within each group of dates of the same length, use the date/dates
        _, dates = file_date_tuple
        try:
            return (-len(dates), dates)
        except TypeError:
            return (-1, dates)

    date_lists = [get_dates(f, fmt=file_date_fmt) for f in files]
    file_dates = sorted([fd_tuple for fd_tuple in zip(files, date_lists)], key=sort_key)

    # Unpack the sorted pairs with new sorted values
    file_list, dates = zip(*file_dates)  # type: ignore
    return list(file_list), list(dates)


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
        return xp.stack(
            [
                take_looks(
                    a,
                    row_looks,
                    col_looks,
                    func_type=func_type,
                    edge_strategy=edge_strategy,
                )
                for a in arr
            ]
        )

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


def decimate(arr: ArrayLike, strides: Dict[str, int]) -> ArrayLike:
    """Decimate an array by strides in the x and y directions.

    Output will match [`io.compute_out_shape`][dolphin.io.compute_out_shape]

    Parameters
    ----------
    arr : ArrayLike
        2D or 3D array to decimate.
    strides : Dict[str, int]
        The strides in the x and y directions.

    Returns
    -------
    ArrayLike
        The decimated array.

    """
    xs, ys = strides["x"], strides["y"]
    rows, cols = arr.shape[-2:]
    start_r = ys // 2
    start_c = xs // 2
    end_r = (rows // ys) * ys + 1
    end_c = (cols // xs) * xs + 1
    return arr[..., start_r:end_r:ys, start_c:end_c:xs]


def get_max_memory_usage(units: str = "GB", children: bool = True) -> float:
    """Get the maximum memory usage of the current process.

    Parameters
    ----------
    units : str, optional, choices=["GB", "MB", "KB", "byte"]
        The units to return, by default "GB".
    children : bool, optional
        Whether to include the memory usage of child processes, by default True

    Returns
    -------
    float
        The maximum memory usage in the specified units.

    Raises
    ------
    ValueError
        If the units are not recognized.

    References
    ----------
    1. https://stackoverflow.com/a/7669279/4174466
    2. https://unix.stackexchange.com/a/30941/295194
    3. https://manpages.debian.org/bullseye/manpages-dev/getrusage.2.en.html

    """
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if children:
        max_mem += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if units.lower().startswith("g"):
        factor = 1e9
    elif units.lower().startswith("m"):
        factor = 1e6
    elif units.lower().startswith("k"):
        factor = 1e3
    elif units.lower().startswith("byte"):
        factor = 1.0
    else:
        raise ValueError(f"Unknown units: {units}")
    if sys.platform.startswith("linux"):
        # on linux, ru_maxrss is in kilobytes, while on mac, ru_maxrss is in bytes
        factor /= 1e3

    return max_mem / factor


def get_gpu_memory(pid: Optional[int] = None, gpu_id: int = 0) -> float:
    """Get the memory usage (in GiB) of the GPU for the current pid."""
    try:
        from pynvml.smi import nvidia_smi
    except ImportError:
        raise ImportError("Please install pynvml through pip or conda")

    def get_mem(process):
        used_mem = process["used_memory"] if process else 0
        if process["unit"] == "MiB":
            multiplier = 1 / 1024
        else:
            logger.warning(f"Unknown unit: {process['unit']}")
        return used_mem * multiplier

    nvsmi = nvidia_smi.getInstance()
    processes = nvsmi.DeviceQuery()["gpu"][gpu_id]["processes"]
    if not processes:
        return 0.0

    if pid is None:
        # Return sum of all processes
        return sum(get_mem(p) for p in processes)
    else:
        procs = [p for p in processes if p["pid"] == pid]
        return get_mem(procs[0]) if procs else 0.0


def moving_window_mean(
    image: ArrayLike, size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """Calculate the mean of a moving window of size `size`.

    Parameters
    ----------
    image : ndarray
        input image
    size : int or tuple of int
        Window size. If a single int, the window is square.
        If a tuple of (row_size, col_size), the window can be rectangular.

    Returns
    -------
    ndarray
        image the same size as `image`, where each pixel is the mean
        of the corresponding window.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError("size must be a single int or a tuple of 2 ints")
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        raise ValueError("size must be odd in both dimensions")

    row_size, col_size = size
    row_pad = row_size // 2
    col_pad = col_size // 2

    # Pad the image with zeros
    image_padded = np.pad(
        image, ((row_pad + 1, row_pad), (col_pad + 1, col_pad)), mode="constant"
    )

    # Calculate the cumulative sum of the image
    integral_img = np.cumsum(np.cumsum(image_padded, axis=0), axis=1)
    if not np.iscomplexobj(integral_img):
        integral_img = integral_img.astype(float)

    # Calculate the mean of the moving window
    # Uses the algorithm from https://en.wikipedia.org/wiki/Summed-area_table
    window_mean = (
        integral_img[row_size:, col_size:]
        - integral_img[:-row_size, col_size:]
        - integral_img[row_size:, :-col_size]
        + integral_img[:-row_size, :-col_size]
    )
    window_mean /= row_size * col_size
    return window_mean
