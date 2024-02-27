from concurrent.futures import Future, ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from typing import Callable, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm

from dolphin import DateOrDatetime, io
from dolphin._types import PathOrStr
from dolphin.utils import flatten

T = TypeVar("T")


@jit
def solve(
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
solve_2d = vmap(solve, in_axes=(None, 1), out_axes=1)
solve_3d = vmap(solve_2d, in_axes=(None, 2), out_axes=2)


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


@jit
def estimate_velocity(unw_stack: ArrayLike, x_arr: ArrayLike) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.

    Returns
    -------
    velocity : np.array 2D
        The estimated velocity in (unw unit) / day.
        E.g. if the unwrapped phase is in radians, the velocity is in rad/day.

    """

    def fit_line(time, y, axis=0):
        # TODO: weighted least squares using correlation?
        return jnp.polyfit(time, y, deg=1, axis=axis, rcond=None)[0]

    # We use the same x inputs for all output pixels, so only vmap over y
    fit_3d = vmap(vmap(fit_line, in_axes=(None, 0)), in_axes=(None, 0))
    return fit_3d(x_arr, unw_stack)


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


def process_blocks(
    file_list: Sequence[PathOrStr],
    output_files: Sequence[PathOrStr],
    func: Callable[[io.StackReader, slice, slice], ArrayLike],
    like_filename: PathOrStr | None = None,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform block-wise processing over `file_list` to create `output_file`."""
    # reader = RasterStackReader.from_file_list(file_list=file_list)
    shape = io.get_raster_xysize(file_list[0])[::-1]
    slices = list(io.iter_blocks(shape, block_shape=block_shape))

    writer = io.GdalStackWriter(
        file_list=output_files, like_filename=like_filename or file_list[0]
    )
    pbar = tqdm(total=len(slices))

    def write_callback(fut: Future):
        rows, cols, data = fut.result()
        writer.queue_write(data, rows.start, cols.start)
        pbar.update()

    with NamedTemporaryFile(mode="w", suffix=".vrt") as f, ThreadPoolExecutor(
        num_threads
    ) as exc:
        reader = io.VRTStack(file_list=file_list, outfile=f.name)
        for rows, cols in slices:
            future = exc.submit(func, reader, rows, cols)
            future.add_done_callback(write_callback)
    writer.notify_finished()


def create_velocity(
    unw_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    date_list: Sequence[DateOrDatetime] | None = None,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform pixel-wise linear regression to estimate velocity."""
    if date_list is None:
        date_list = get_dates(unw_file_list)
    x_arr = datetime_to_float(date_list)

    def read_and_fit(
        reader: io.StackReader, rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        stack = reader[:, rows, cols]
        # Fit a line to each pixel
        return rows, cols, estimate_velocity(stack, x_arr)

    return process_blocks(
        file_list=unw_file_list,
        output_files=[output_file],
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
        reader: io.StackReader, rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        return rows, cols, np.nanmean(reader[:, rows, cols], axis=0)

    return process_blocks(
        file_list=file_list,
        output_files=[output_file],
        func=read_and_average,
        block_shape=block_shape,
        num_threads=num_threads,
    )
