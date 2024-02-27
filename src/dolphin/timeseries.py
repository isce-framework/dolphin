from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
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


@jit
def estimate_velocity(x_arr: ArrayLike, unw_stack: ArrayLike) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.

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
    coeffs = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)
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
        return rows, cols, estimate_velocity(x_arr=x_arr, stack=stack)

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
        reader: io.StackReader, rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        stack = reader[:, rows, cols]
        return rows, cols, invert_network_stack(A, stack)

    return process_blocks(
        file_list=unw_file_list,
        output_files=out_paths,
        func=read_and_solve,
        block_shape=block_shape,
        num_threads=num_threads,
    )
