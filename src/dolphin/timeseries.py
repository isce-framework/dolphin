import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Protocol, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from numpy.typing import ArrayLike
from opera_utils import get_dates
from tqdm.auto import tqdm

from dolphin import DateOrDatetime, io
from dolphin._types import PathOrStr, ReferencePoint
from dolphin.utils import DummyProcessPoolExecutor, flatten, format_dates

__all__ = [
    "weighted_lstsq_single",
    "invert_stack",
    "get_incidence_matrix",
    "estimate_velocity",
    "create_velocity",
    "create_temporal_average",
    "invert_unw_network",
]

T = TypeVar("T")

logger = logging.getLogger(__name__)


@jit
def weighted_lstsq_single(
    A: ArrayLike,
    b: ArrayLike,
    weights: ArrayLike,
) -> Array:
    r"""Perform weighted least for one data vector.

    Minimizes the weighted 2-norm of the residual vector:

    \[
        || b - A x ||^2_W
    \]

    where \(W\) is a diagonal matrix of weights.

    Parameters
    ----------
    A : Arraylike
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    b : ArrayLike, 1D
        The phase differences between the ifg pairs
    weights : ArrayLike, 1D, optional
        The weights for each element of `b`.

    Returns
    -------
    x : np.array 1D
        The estimated phase for each SAR acquisition
    residuals : np.array 1D
        Sums of squared residuals: Squared Euclidean 2-norm for `b - A @ x`
        For a 1D `b`, this is a scalar.

    """
    # scale both A and b by sqrt so we are minimizing
    sqrt_weights = jnp.sqrt(weights)
    # Multiply each data point by sqrt(weight)
    b_scaled = b * sqrt_weights
    # Multiply each row by sqrt(weight)
    A_scaled = A * sqrt_weights[:, None]

    # Run the weighted least squares
    x, residuals, rank, sing_vals = jnp.linalg.lstsq(A_scaled, b_scaled)
    # TODO: do we need special handling?
    # if rank < A.shape[1]:
    #     # logger.warning("Rank deficient solution")

    return x, residuals.ravel()


@jit
def invert_stack(
    A: ArrayLike, dphi: ArrayLike, weights: ArrayLike | None = None
) -> Array:
    """Solve the SBAS problem for a stack of unwrapped phase differences.

    Parameters
    ----------
    A : ArrayLike
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    dphi : ArrayLike
        The phase differences between the ifg pairs, shape=(n_ifgs, n_rows, n_cols)
    weights : ArrayLike, optional
        The weights for each element of `dphi`.
        Same shape as `dphi`.
        If not provided, all weights are set to 1 (ordinary least squares).

    Returns
    -------
    phi : np.array 3D
        The estimated phase for each SAR acquisition
        Shape is (n_sar_dates, n_rows, n_cols)
    residuals : np.array 2D
        Sums of squared residuals: Squared Euclidean 2-norm for `dphi - A @ x`
        Shape is (n_rows, n_cols)

    Notes
    -----
    To mask out data points of a pixel, the weight can be set to 0.
    When `A` remains full rank, setting the weight to zero is the same as removing
    the entry from the data vector and the corresponding row from `A`.

    """
    n_ifgs, n_rows, n_cols = dphi.shape

    # vectorize the solve function to work on 2D and 3D arrays
    # We are not vectorizing over the A matrix, only the dphi vector
    # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
    # invert_2d = vmap(invert_single, in_axes=(None, 1, 1), out_axes=1)
    invert_2d = vmap(weighted_lstsq_single, in_axes=(None, 1, 1), out_axes=(1, 1))
    # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
    invert_3d = vmap(invert_2d, in_axes=(None, 2, 2), out_axes=(2, 2))

    if weights is None:
        weights = jnp.ones_like(dphi)
    phase, residuals = invert_3d(A, dphi, weights)
    # Add 0 for the reference date to the front
    phase = jnp.concatenate([jnp.zeros((1, n_rows, n_cols)), phase], axis=0)
    # Reshape the residuals to be 2D
    return phase, residuals[0]


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
    # Jax polyfit will grab the first *2* dimensions of y to solve in a batch
    return jnp.polyfit(x, y, deg=1, w=w.reshape(y.shape), rcond=None)[0]


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
    assert unw_stack.shape == weight_stack.shape
    unw_pixels = unw_stack.reshape(n_time, -1)
    weights_pixels = weight_stack.reshape(n_time, 1, -1)

    # coeffs = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)
    velos = vmap(estimate_velocity_pixel, in_axes=(None, -1, -1))(
        x_arr, unw_pixels, weights_pixels
    )
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

    Reads a block of data from each reader, processes it, and returns the result
    as an array-like object.
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
    shape = readers[0].shape[-2:]
    slices = list(io.iter_blocks(shape, block_shape=block_shape))

    pbar = tqdm(total=len(slices))

    # Define the callback to write the result to an output DatasetWrite
    def write_callback(fut: Future):
        rows, cols, data = fut.result()
        writer[..., rows, cols] = data
        pbar.update()

    num_threads = 1
    Executor = ThreadPoolExecutor if num_threads > 1 else DummyProcessPoolExecutor
    with Executor(num_threads) as exc:
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
) -> None:
    """Perform pixel-wise (weighted) linear regression to estimate velocity."""
    if Path(output_file).exists():
        logger.info(f"Output file {output_file} already exists, skipping velocity")
        return

    if date_list is None:
        date_list = [get_dates(f)[1] for f in unw_file_list]
    x_arr = datetime_to_float(date_list)

    def read_and_fit(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        # Only use the cor_reader if it's the same shape as the unw_reader
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
            estimate_velocity(x_arr=x_arr, unw_stack=unw_stack, weight_stack=weights),
        )

    use_cor = cor_file_list is not None and len(cor_file_list) == len(unw_file_list)
    # Note: For some reason, the `RasterStackReader` is much slower than the VRT
    # for files on S3:
    # ~300 files takes >2 min to open, >2 min to read each block
    # VRTStack seems to take ~30 secs to open, 1 min to read
    # Very possible there's a tuning param/rasterio config to fix, but not sure.
    with NamedTemporaryFile(mode="w", suffix=".vrt") as f1, NamedTemporaryFile(
        mode="w", suffix=".vrt"
    ) as f2:
        unw_reader = io.VRTStack(
            file_list=unw_file_list, outfile=f1.name, skip_size_check=True
        )
        if use_cor:
            assert cor_file_list is not None
            cor_reader = io.VRTStack(
                file_list=cor_file_list, outfile=f2.name, skip_size_check=True
            )
            readers = [unw_reader, cor_reader]
        else:
            readers = [unw_reader]

        writer = io.BackgroundRasterWriter(output_file, like_filename=unw_file_list[0])
        process_blocks(
            readers=readers,
            writer=writer,
            func=read_and_fit,
            block_shape=block_shape,
            num_threads=num_threads,
        )

    writer.notify_finished()


class AverageFunc(Protocol):
    """Protocol for temporally averaging a block of data."""

    def __call__(self, ArrayLike, axis: int) -> ArrayLike:
        ...


def create_temporal_average(
    file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
    average_func: Callable[[ArrayLike, int], np.ndarray] = np.nanmean,
) -> None:
    """Average all images in `reader` to create a 2D image in `output_file`.

    Parameters
    ----------
    file_list : Sequence[PathOrStr]
        List of files to average
    output_file : PathOrStr
        The output file to save the average to
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (512, 512)
    num_threads : int, optional
        The parallel blocks to process at once.
        Default is 5.
    average_func : Callable[[ArrayLike, int], np.ndarray], optional
        The function to use to average the images.
        Default is `np.nanmean`, which calls
        `np.nanmean(arr, axis=0)`.
        on each block.

    """

    def read_and_average(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        chunk = readers[0][:, rows, cols]
        return rows, cols, average_func(chunk, 0)

    writer = io.BackgroundRasterWriter(output_file, like_filename=file_list[0])
    with NamedTemporaryFile(mode="w", suffix=".vrt") as f:
        reader = io.VRTStack(file_list=file_list, outfile=f.name, skip_size_check=True)

        process_blocks(
            readers=[reader],
            writer=writer,
            func=read_and_average,
            block_shape=block_shape,
            num_threads=num_threads,
        )

    writer.notify_finished()


def invert_unw_network(
    unw_file_list: Sequence[PathOrStr],
    reference: ReferencePoint,
    output_dir: PathOrStr,
    cor_file_list: Sequence[PathOrStr] | None = None,
    cor_threshold: float = 0.2,
    ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
) -> list[Path]:
    """Perform pixel-wise inversion of unwrapped network to get phase per date.

    Parameters
    ----------
    unw_file_list : Sequence[PathOrStr]
        List of unwrapped phase files.
    reference : ReferencePoint
        The reference point to use for the inversion.
        The data vector from `unw_file_list` at this point will be subtracted
        from all other points when solving.
    output_dir : PathOrStr
        The directory to save the output files
    ifg_date_pairs : Sequence[Sequence[DateOrDatetime]], optional
        List of date pairs to use for the inversion. If not provided, will be
        parsed from filenames in `unw_file_list`.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel
    cor_file_list : Sequence[PathOrStr], optional
        List of correlation files to use for weighting the inversion
    cor_threshold : float, optional
        The correlation threshold to use for weighting the inversion
    num_threads : int, optional
        The parallel blocks to process at once.

    Returns
    -------
    out_paths : list[Path]
        List of the output files created by the inversion.

    """
    if ifg_date_pairs is None:
        ifg_date_pairs = [get_dates(f) for f in unw_file_list]

    try:
        # Ensure it's a list of pairs
        ifg_tuples = [(ref, sec) for (ref, sec) in ifg_date_pairs]  # noqa: C416
    except ValueError as e:
        raise ValueError(
            "Each item in `ifg_date_pairs` must be a sequence of length 2"
        ) from e
    # Make the names of the output files from the SAR dates to solve for
    sar_dates = sorted(set(flatten(ifg_tuples)))
    ref_date = sar_dates[0]
    suffix = ".tif"
    out_paths = [
        Path(output_dir) / (format_dates(ref_date, d) + suffix) for d in sar_dates
    ]
    if all(p.exists() for p in out_paths):
        logger.info("All output files already exist, skipping inversion")
        return out_paths

    A = get_incidence_matrix(ifg_pairs=ifg_tuples, sar_idxs=sar_dates)

    out_vrt_name = Path(output_dir) / "unw_network.vrt"
    unw_reader = io.VRTStack(
        file_list=unw_file_list, outfile=out_vrt_name, skip_size_check=True
    )
    cor_vrt_name = Path(output_dir) / "unw_network.vrt"

    # Get the reference point data
    ref_row, ref_col = reference
    ref_data = unw_reader[:, ref_row, ref_col].reshape(-1, 1, 1)

    def read_and_solve(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        if len(readers) == 2:
            unw_reader, cor_reader = readers
            stack = unw_reader[:, rows, cols]
            weights = cor_reader[:, rows, cols]
            weights[weights < cor_threshold] = 0
        else:
            stack = readers[0][:, rows, cols]
            weights = np.ones_like(stack)

        # subtract the reference
        stack = stack - ref_data

        # TODO: possible second input for weights? from conncomps
        # TODO: do i want to write residuals too? Do i need
        # to have multiple writers then?
        phases = invert_stack(A, stack, weights)[0]
        return rows, cols, np.asarray(phases)

    if cor_file_list is not None:
        cor_reader = io.VRTStack(
            file_list=cor_file_list, outfile=cor_vrt_name, skip_size_check=True
        )
        readers = [unw_reader, cor_reader]
    else:
        readers = [unw_reader]

    writer = io.BackgroundStackWriter(out_paths, like_filename=unw_file_list[0])

    process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_solve,
        block_shape=block_shape,
        num_threads=num_threads,
    )
    writer.notify_finished()
    return out_paths


def select_reference_point(
    ccl_file_list: Sequence[PathOrStr],
    amp_dispersion_file: PathOrStr,
    output_dir: Path,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
) -> ReferencePoint:
    """Automatically select a reference point for a stack of unwrapped interferograms.

    Uses the amplitude dispersion and connected component labels, the point is selected
    which

    1. is within intersection of all nonzero connected component labels (always valid)
    2. has the lowest amplitude dispersion
    """
    conncomp_intersection_file = Path(output_dir) / "conncomp_intersection.tif"

    def intersect_conncomp(arr: ArrayLike, axis: int) -> np.ndarray:
        is_valid_conncomp = arr > 0
        return np.logical_and.reduce(is_valid_conncomp, axis=axis).astype(np.int16)

    if not conncomp_intersection_file.exists():
        logger.info("Creating intersection of connected components")
        create_temporal_average(
            file_list=ccl_file_list,
            output_file=conncomp_intersection_file,
            block_shape=block_shape,
            num_threads=num_threads,
            average_func=intersect_conncomp,
        )

    logger.info("Selecting reference point")
    conncomp_intersection = io.load_gdal(conncomp_intersection_file)
    # ps_mask = io.load_gdal(ps_mask_file, masked=True)
    amp_dispersion = io.load_gdal(amp_dispersion_file, masked=True)

    # Mask out where the conncomps where 0
    amp_dispersion.mask = amp_dispersion.mask | (conncomp_intersection == 0)

    # Pick the point with the lowest amplitude dispersion
    ref_row, ref_col = np.unravel_index(np.argmin(amp_dispersion), amp_dispersion.shape)

    return ReferencePoint(int(ref_row), int(ref_col))
