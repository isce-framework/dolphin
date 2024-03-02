from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm

from dolphin import DateOrDatetime, io
from dolphin._types import PathOrStr
from dolphin.utils import flatten, format_dates

__all__ = [
    "invert_network",
    "invert_network_stack",
    "get_incidence_matrix",
    "estimate_velocity",
    "create_velocity",
    "create_temporal_average",
    "invert_unw_network",
]

T = TypeVar("T")


@jit
def invert_network(
    A: ArrayLike,
    dphi: np.ndarray,
) -> Array:
    """Solve the SBAS problem for a list of ifg pairs and phase differences.

    Parameters
    ----------
    A : Arraylike
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    dphi : np.array 1D
        The phase differences between the ifg pairs

    Returns
    -------
    phi : np.array 1D
        The estimated phase for each SAR acquisition

    """
    phi = jnp.linalg.lstsq(A, dphi, rcond=None)[0]
    # Add 0 for the reference date to the front
    return jnp.concatenate([jnp.array([0]), phi])


# vectorize the solve function to work on 2D and 3D arrays
# We are not vectorizing over the A matrix, only the dphi vector
# Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
invert_network_2d = vmap(invert_network, in_axes=(None, 1), out_axes=1)
# Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
invert_network_stack = vmap(invert_network_2d, in_axes=(None, 2), out_axes=2)


def get_incidence_matrix(
    ifg_pairs: Sequence[tuple[T, T]], sar_idxs: Sequence[T] | None = None
) -> np.ndarray:
    """Build the indicator matrix from a list of ifg pairs (index 1, index 2).

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[T, T]]
        List of ifg pairs represented as tuples of (day 1, day 2)
        Can be ints, datetimes, etc.
    sar_idxs : Sequence[T], optional
        If provided, used as the total set of indexes which `ifg_pairs`
        were formed from.
        Otherwise, created from the unique entries in `ifg_pairs`.
        Only provide if there are some dates which are not present in `ifg_pairs`.

    Returns
    -------
    A : np.array 2D
        The incident-like matrix for the system: A*phi = dphi
        Each row corresponds to an ifg, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
        Shape: (n_ifgs, n_sar_dates - 1)

    """
    if sar_idxs is None:
        sar_idxs = sorted(set(flatten(ifg_pairs)))

    M = len(ifg_pairs)
    N = len(sar_idxs) - 1
    A = np.zeros((M, N))

    # Create a dictionary mapping sar dates to matrix columns
    # We take the first SAR acquisition to be time 0, leave out of matrix
    date_to_col = {date: i for i, date in enumerate(sar_idxs[1:])}
    # Populate the matrix
    for i, (early, later) in enumerate(ifg_pairs):
        if early in date_to_col:
            A[i, date_to_col[early]] = -1
        if later in date_to_col:
            A[i, date_to_col[later]] = +1

    return A


# """vmapping polyfit
# In [138]: def fit1w(x, y, w):
#      ...:     return jnp.polyfit(x, y, deg=1, w=w)
#      ...:
# In [140]: weights2d = np.ones_like(yy)

# In [142]: vfitw = vmap(fit1w, in_axes=(None, -1, -1), out_axes=(-1))
# In [143]: vfit1w(xx, yy.reshape(100, 1, -1), weights2d.reshape(100, 1, -1))
# """


@jit
def estimate_velocity_pixel(x: ArrayLike, y: ArrayLike, w: ArrayLike) -> Array:
    """Estimate the velocity from a single pixel's time series.

    Parameters
    ----------
    x : np.array 1D
        The time values
    y : np.array 1D
        The unwrapped phase values
    w : np.array 1D
        The weights for each time value

    Returns
    -------
    velocity : np.array, 0D
        The estimated velocity in (unw unit) / day.

    """
    # Need to reshape w so that it can be broadcast with x and y
    # Jax polyfit will grab the first *2* dimensions of y to solve in a batch
    return jnp.polyfit(x, y.reshape(-1, 1), deg=1, w=w.reshape(-1, 1), rcond=None)[0]


@jit
def estimate_velocity(
    x_arr: ArrayLike, unw_stack: ArrayLike, weight_stack: ArrayLike
) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.
    weight_stack : ArrayLike
        Array of weights for each pixel, shape=`(n_time, n_rows, n_cols)`.
        If not provided, all weights are set to 1.

    Returns
    -------
    velocity : np.array 2D
        The estimated velocity in (unw unit) / day.
        E.g. if the unwrapped phase is in radians, the velocity is in rad/day.

    """
    # TODO: weighted least squares using correlation?
    n_time, n_rows, n_cols = unw_stack.shape

    # We use the same x inputs for all output pixels
    unw_pixels = unw_stack.reshape(n_time, -1)
    weights_pixels = weight_stack.reshape(n_time, -1)

    # coeffs = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)
    coeffs = vmap(estimate_velocity_pixel, in_axes=(None, 0, 0))(
        x_arr, unw_pixels, weights_pixels
    )
    velos = coeffs[0]
    return velos.reshape(n_rows, n_cols)


def datetime_to_float(dates: Sequence[DateOrDatetime]) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[0]
    return date_arr.astype(float) / sec_per_day


class BlockProcessor(Protocol):
    """Protocol for a block-wise processing function.

    Reads a block from each reader, processes it, and returns the result.
    """

    def __call__(
        self, readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> ArrayLike:
        ...


def process_blocks(
    readers: Sequence[io.StackReader],
    writer: io.DatasetWriter,
    func: BlockProcessor,
    # output_files: Sequence[PathOrStr],
    # like_filename: PathOrStr | None = None,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform block-wise processing over blocks in `readers`, writing to `writer`."""
    # reader = RasterStackReader.from_file_list(file_list=file_list)
    # writer = io.GdalStackWriter(
    #     file_list=output_files, like_filename=like_filename or file_list[0]
    # )
    # shape = io.get_raster_xysize(file_list[0])[::-1]
    shape = readers[0].shape[-2:]
    slices = list(io.iter_blocks(shape, block_shape=block_shape))

    pbar = tqdm(total=len(slices))

    def write_callback(fut: Future):
        rows, cols, data = fut.result()
        writer[..., rows, cols] = data
        pbar.update()

    with ThreadPoolExecutor(num_threads) as exc:
        for rows, cols in slices:
            future = exc.submit(func, readers=readers, rows=rows, cols=cols)
            future.add_done_callback(write_callback)


def create_velocity(
    unw_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    date_list: Sequence[DateOrDatetime] | None = None,
    cor_file_list: Sequence[PathOrStr] | None = None,
    cor_threshold: float = 0.2,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform pixel-wise (weighted) linear regression to estimate velocity."""
    if date_list is None:
        date_list = get_dates(unw_file_list)
    x_arr = datetime_to_float(date_list)

    def read_and_fit(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        if len(readers) == 2:
            unw_reader, cor_reader = readers
            unw_stack = unw_reader[:, rows, cols]
            weights = cor_reader[:, rows, cols]
            weights[weights < cor_threshold] = 0
        else:
            unw_stack = readers[0][:, rows, cols]
            weights = np.ones_like(unw_stack)
        # Fit a line to each pixel with weighted least squares
        return (
            rows,
            cols,
            estimate_velocity(x_arr=x_arr, stack=unw_stack, weight_stack=weights),
        )

    # Note: For some reason, the `RasterStackReader` is much slower than the VRT:
    # ~300 files takes >2 min to open, >2 min to read each block
    # VRTStack seems to take ~30 secs to open, 1 min to read
    # Very possible there's a tuning param/rasterio config to fix, but not sure.
    # unw_reader = io.RasterStackReader.from_file_list(file_list=unw_file_list)
    # cor_reader = io.RasterStackReader.from_file_list(file_list=cor_file_list)
    with NamedTemporaryFile(mode="w", suffix=".vrt") as f1, NamedTemporaryFile(
        mode="w", suffix=".vrt"
    ) as f2:
        unw_reader = io.VRTStack(
            file_list=unw_file_list, outfile=f1.name, skip_size_check=True
        )
        if cor_file_list is not None:
            cor_reader = io.VRTStack(
                file_list=cor_file_list, outfile=f2.name, skip_size_check=True
            )
            readers = [unw_reader, cor_reader]
        else:
            readers = [unw_reader]

        writer = io.BackgroundRasterWriter(output_file, like_filename=unw_file_list[0])
        return process_blocks(
            # file_list=unw_file_list,
            # output_files=[output_file],
            readers=readers,
            writer=writer,
            func=read_and_fit,
            block_shape=block_shape,
            num_threads=num_threads,
        )


def create_temporal_average(
    file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Average all images in `reader` to create a 2D image in `output_file`."""

    def read_and_average(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        return rows, cols, np.nanmean(readers[0][:, rows, cols], axis=0)

    with NamedTemporaryFile(mode="w", suffix=".vrt") as f:
        reader = io.VRTStack(file_list=file_list, outfile=f.name, skip_size_check=True)
        writer = io.BackgroundRasterWriter(output_file, like_filename=file_list[0])

        return process_blocks(
            # file_list=file_list,
            # output_files=[output_file],
            readers=[reader],
            writer=writer,
            func=read_and_average,
            block_shape=block_shape,
            num_threads=num_threads,
        )


def invert_unw_network(
    unw_file_list: Sequence[PathOrStr],
    output_dir: PathOrStr,
    ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform pixel-wise inversion of unwrapped network to get phase per date."""
    if ifg_date_pairs is None:
        ifg_date_pairs = get_dates(unw_file_list)

    ifg_tuples = [tuple(pair) for pair in ifg_date_pairs]
    if not all(len(pair) == 2 for pair in ifg_tuples):
        raise ValueError("Each item in `ifg_date_pairs` must be a sequence of length 2")

    sar_dates = sorted(set(flatten(ifg_tuples)))
    A = get_incidence_matrix(ifg_pairs=ifg_tuples, sar_idxs=sar_dates)
    # Make the names of the output files from the SAR dates to solve for
    ref_date = sar_dates[0]
    out_paths = [Path(output_dir) / format_dates(ref_date, d) for d in sar_dates]

    def read_and_solve(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        stack = readers[0][:, rows, cols]
        # TODO: possible second for weights
        return rows, cols, invert_network_stack(A, stack)

    with NamedTemporaryFile(mode="w", suffix=".vrt") as f:
        reader = io.VRTStack(
            file_list=unw_file_list, outfile=f.name, skip_size_check=True
        )
        writer = io.BackgroundStackWriter(out_paths, like_filename=unw_file_list[0])

    return process_blocks(
        readers=[reader],
        writer=writer,
        func=read_and_solve,
        block_shape=block_shape,
        num_threads=num_threads,
    )
