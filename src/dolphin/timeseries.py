from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional, Protocol, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from numpy.typing import ArrayLike
from opera_utils import get_dates
from scipy import ndimage

from dolphin import DateOrDatetime, io, utils
from dolphin._overviews import ImageType, create_overviews
from dolphin._types import PathOrStr, ReferencePoint
from dolphin.utils import flatten, format_dates, full_suffix, get_nearest_date_idx
from dolphin.workflows import CallFunc

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = ["run"]


class InversionMethod(str, Enum):
    """Method to use for timeseries inversion."""

    L1 = "L1"
    L2 = "L2"


class ReferencePointError(ValueError):
    pass


def run(
    unwrapped_paths: Sequence[PathOrStr],
    conncomp_paths: Sequence[PathOrStr],
    condition_file: PathOrStr,
    condition: CallFunc,
    output_dir: PathOrStr,
    method: InversionMethod = InversionMethod.L2,
    run_velocity: bool = False,
    corr_paths: Sequence[PathOrStr] | None = None,
    weight_velocity_by_corr: bool = False,
    velocity_file: Optional[PathOrStr] = None,
    correlation_threshold: float = 0.2,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    reference_point: tuple[int, int] = (-1, -1),
    wavelength: float | None = None,
    add_overviews: bool = True,
    extra_reference_date: datetime | None = None,
) -> tuple[list[Path], ReferencePoint]:
    """Invert the unwrapped interferograms, estimate timeseries and phase velocity.

    Parameters
    ----------
    unwrapped_paths : Sequence[Path]
        Sequence unwrapped interferograms to invert.
    conncomp_paths : Sequence[Path]
        Sequence connected component files, one per file in `unwrapped_paths`
    condition_file: PathOrStr
        A file with the same size as each raster, like amplitude dispersion or
        temporal coherence
    condition: CallFunc
        The function to apply to the condition file,
        for example numpy.argmin which finds the pixel with lowest value
        the options are [min, max]
    output_dir : Path
        Path to the output directory.
    method : str, choices = "L1", "L2"
        Inversion method to use when solving Ax = b.
        Default is L2, which uses least squares to solve Ax = b (faster).
        "L1" minimizes |Ax - b|_1 at each pixel.
    run_velocity : bool
        Whether to run velocity estimation on the inverted phase series
    corr_paths : Sequence[Path], optional
        Sequence interferometric correlation files, one per file in `unwrapped_paths`.
        If not provided, does no weighting by correlation.
    weight_velocity_by_corr : bool
        Flag to indicate whether the velocity fitting should use correlation as weights.
        Default is False.
    velocity_file : Path, Optional
        The output velocity file
    correlation_threshold : float
        Pixels with correlation below this value will be masked out
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256)
    num_threads : int
        The parallel blocks to process at once.
        Default is 4.
    reference_point : tuple[int, int], optional
        Reference point (row, col) used if performing a time series inversion.
        If not provided, a point will be selected from a consistent connected
        component with low amplitude dispersion or high temporal coherence.
    wavelength : float, optional
        The wavelength of the radar signal, in meters.
        If provided, the output rasters are in meters and meters / year for
        the displacement and velocity rasters.
        If not provided, the outputs are in radians.
        See Notes for line of sight convention.
    add_overviews : bool, optional
        If True, creates overviews of the new velocity raster.
        Default is True.
    extra_reference_date : datetime.datetime, optional
        If provided, makes another set of interferograms referenced to this
        for all dates later than it.

    Returns
    -------
    inverted_phase_paths : list[Path]
        list of Paths to inverted interferograms (single reference phase series).
    reference_point : ReferencePoint
        NamedTuple of reference (row, column) selected.
        If passed as input, simply returned back as output.
        Otherwise, the result is the auto-selection from `select_reference_point`.

    Notes
    -----
    When wavelength is provided, the output rasters are in meters and meters / year,
    where positive values indicate motion *toward* from the radar (i.e. positive values
    in both ascending and descending tracks imply uplift).

    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    condition_func = argmax_index if condition == CallFunc.MAX else argmin_index
    if reference_point == (-1, -1):
        logger.info("Selecting a reference point for unwrapped interferograms")
        ref_point = select_reference_point(
            condition_file=condition_file,
            output_dir=Path(output_dir),
            condition_func=condition_func,
            ccl_file_list=conncomp_paths,
        )
    else:
        ref_point = ReferencePoint(row=reference_point[0], col=reference_point[1])

    ifg_date_pairs = [get_dates(f) for f in unwrapped_paths]
    sar_dates = sorted(set(utils.flatten(ifg_date_pairs)))
    # if we did single-reference interferograms, for `n` sar dates, we will only have
    # `n-1` interferograms. Any more than n-1 ifgs means we need to invert
    is_single_reference = len(unwrapped_paths) == len(sar_dates) - 1
    # TODO: Do we ever want to invert this case: the "trivial" network,
    # which has 1 ifg per date difference, but a moving reference date?
    # The extra condition to check is
    # ... and all(pair[0] == ifg_date_pairs[0][0] for pair in ifg_date_pairs)

    # check if we even need to invert, or if it was single reference
    inverted_phase_paths: list[Path] = []
    if is_single_reference:
        logger.info(
            "Skipping inversion step: only single reference interferograms exist."
        )
        # Copy over the unwrapped paths to `timeseries/`
        inverted_phase_paths = _convert_and_reference(
            unwrapped_paths,
            output_dir=output_dir,
            reference_point=ref_point,
            wavelength=wavelength,
        )
    else:
        logger.info("Inverting network of %s unwrapped ifgs", len(unwrapped_paths))
        inverted_phase_paths = invert_unw_network(
            unw_file_list=unwrapped_paths,
            reference=ref_point,
            output_dir=output_dir,
            block_shape=block_shape,
            num_threads=num_threads,
            wavelength=wavelength,
            method=method,
        )
    if add_overviews:
        logger.info("Creating overviews for timeseries images")
        create_overviews(inverted_phase_paths, image_type=ImageType.UNWRAPPED)

    if run_velocity:
        logger.info("Estimating phase velocity")

        #  We can't pass the correlations after an inversion- the numbers don't match
        # TODO:
        # Is there a better weighting then?
        if not weight_velocity_by_corr or corr_paths is None:
            cor_file_list = None
        else:
            cor_file_list = (
                corr_paths if len(corr_paths) == len(inverted_phase_paths) else None
            )

        if velocity_file is None:
            velocity_file = Path(output_dir) / "velocity.tif"

        create_velocity(
            unw_file_list=inverted_phase_paths,
            output_file=velocity_file,
            reference=ref_point,
            cor_file_list=cor_file_list,
            cor_threshold=correlation_threshold,
            block_shape=block_shape,
            num_threads=num_threads,
        )

    if extra_reference_date is None:
        final_out = inverted_phase_paths
    else:
        final_out = _redo_reference(inverted_phase_paths, extra_reference_date)

    return final_out, ref_point


def _redo_reference(
    inverted_phase_paths: Sequence[Path], extra_reference_date: datetime
):
    """Reset the reference date in `inverted_phase_paths`.

    Affects all files whose secondary is after `extra_reference_date`.

    E.g Given the (day 1, day 2), ..., (day 1, day N) pairs, outputs
    (1, 2), (1, 3), ...(1, r), (r, r+1), ..., (r, N)
    where r is the index of the `extra_reference_date`
    """
    output_path = inverted_phase_paths[0].parent
    inverted_date_pairs: list[tuple[datetime, datetime]] = [
        get_dates(p.stem)[:2] for p in inverted_phase_paths
    ]
    secondary_dates = [pair[1] for pair in inverted_date_pairs]
    extra_ref_idx = get_nearest_date_idx(
        secondary_dates, requested=extra_reference_date
    )
    ref_date = secondary_dates[extra_ref_idx]
    logger.info(f"Re-referencing later timeseries files to {ref_date}")
    extra_ref_img = inverted_phase_paths[extra_ref_idx]
    ref = io.load_gdal(extra_ref_img, masked=True)

    # Use a temp directory while re-referencing
    extra_out_dir = inverted_phase_paths[0].parent / "extra"
    extra_out_dir.mkdir(exist_ok=True)

    for idx in range(extra_ref_idx + 1, len(inverted_date_pairs)):
        # To create the interferogram (r, r+1), we subtract
        # (1, r) from (1, r+1)
        cur_img = inverted_phase_paths[idx]
        new_stem = format_dates(ref_date, secondary_dates[idx])
        cur_output_name = extra_out_dir / f"{new_stem}.tif"
        cur = io.load_gdal(cur_img, masked=True)
        new_out = cur - ref
        io.write_arr(
            arr=new_out, like_filename=extra_ref_img, output_name=cur_output_name
        )

    for idx, p in enumerate(inverted_phase_paths):
        if idx <= extra_ref_idx:
            p.rename(extra_out_dir / p.name)
        else:
            p.unlink()
    # Finally, move them back in to the `timeseries/` folder
    final_out = []
    for p in extra_out_dir.glob("*.tif"):
        final_out.append(p.rename(output_path / p.name))
    return sorted(final_out)


def _convert_and_reference(
    unwrapped_paths: Sequence[PathOrStr],
    *,
    output_dir: PathOrStr,
    reference_point: ReferencePoint,
    wavelength: float | None = None,
) -> list[Path]:
    if wavelength is not None:
        # Positive values are motion towards the radar
        constant = -1 * (wavelength / (4 * np.pi))
        units = "meters"
    else:
        constant = -1
        units = "radians"

    ref_row, ref_col = reference_point
    out_paths: list[Path] = []
    for p in unwrapped_paths:
        # if it ends in `.unw.tif`, change to just `.tif` for consistency
        # with the case where we run an inversion
        cur_name = Path(p).name
        unw_suffix = full_suffix(p)
        target_name = str(cur_name).replace(unw_suffix, ".tif")
        target = Path(output_dir) / target_name
        out_paths.append(target)

        if target.exists():  # Check to prevent overwriting
            continue

        arr_radians = io.load_gdal(p, masked=True)
        nodataval = io.get_raster_nodata(p)
        # Reference to the
        ref_value = arr_radians.filled(np.nan)[ref_row, ref_col]
        if np.isnan(ref_value):
            logger.warning(
                "{ref_point!r} is NaN for {p} . Skipping reference subtraction."
            )
        else:
            arr_radians -= ref_value
        # Make sure we keep the same mask as the original
        out_arr = (arr_radians * constant).filled(nodataval)
        io.write_arr(
            arr=out_arr,
            output_name=target,
            units=units,
            like_filename=p,
            nodata=nodataval,
        )

    return out_paths


def argmin_index(arr: ArrayLike) -> tuple[int, ...]:
    """Get the index tuple of the minimum value of the array.

    If multiple occurrences of the minimum value exist, returns
    the index of the first such occurrence in the flattened array.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    tuple of int
        The index of the minimum value.

    """
    return np.unravel_index(np.argmin(arr), np.shape(arr))


def argmax_index(arr: ArrayLike) -> tuple[int, ...]:
    """Get the index tuple of the maximum value of the array.

    If multiple occurrences of the maximum value exist, returns
    the index of the first such occurrence in the flattened array.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    tuple of int
        The index of the maximum value.

    """
    return np.unravel_index(np.argmax(arr), np.shape(arr))


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
        Shape is (n_sar_dates - 1, n_rows, n_cols)
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

    if weights is None:
        # Can use ordinary least squares with no weights
        # Reshape to be size (M, K) instead of 3D
        b = dphi.reshape(n_ifgs, -1)
        phase_cols, residuals_cols, _, _ = jnp.linalg.lstsq(A, b)
        # Reshape the phase and residuals to be 3D
        phase = phase_cols.reshape(-1, n_rows, n_cols)
        residuals = residuals_cols.reshape(n_rows, n_cols)
    else:
        # vectorize the solve function to work on 2D and 3D arrays
        # We are not vectorizing over the A matrix, only the dphi vector
        # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
        invert_2d = vmap(weighted_lstsq_single, in_axes=(None, 1, 1), out_axes=(1, 1))
        # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
        invert_3d = vmap(invert_2d, in_axes=(None, 2, 2), out_axes=(2, 2))
        phase, residuals = invert_3d(A, dphi, weights)
        # Reshape the residuals to be 2D
        residuals = residuals[0]

    return phase, residuals


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
        The estimated velocity in (unw unit) / year.

    """
    # Jax polyfit will grab the first *2* dimensions of y to solve in a batch
    return jnp.polyfit(x, y, deg=1, w=w.reshape(y.shape), rcond=None)[0]


@jit
def estimate_velocity(
    x_arr: ArrayLike, unw_stack: ArrayLike, weight_stack: ArrayLike | None
) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.
    weight_stack : ArrayLike, optional
        Array of weights for each pixel, shape=`(n_time, n_rows, n_cols)`.
        If not provided, performs one batch unweighted linear fit.

    Returns
    -------
    velocity : np.array 2D
        The estimated velocity in (unw unit) / year calculated as 365.25 * rad/day.
        E.g. if the unwrapped phase is in radians, the velocity is in rad/year.

    """
    # TODO: weighted least squares using correlation?
    n_time, n_rows, n_cols = unw_stack.shape

    unw_pixels = unw_stack.reshape(n_time, -1)
    if weight_stack is None:
        # For jnp.polyfit(...), coeffs[0] is slope, coeffs[1] is the intercept
        velos = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)[0]
    else:
        # We use the same x inputs for all output pixels
        if unw_stack.shape != weight_stack.shape:
            msg = (
                "unw_stack and weight_stack must have the same shape,"
                f" got {unw_stack.shape} and {weight_stack.shape}"
            )
            raise ValueError(msg)

        weights_pixels = weight_stack.reshape(n_time, 1, -1)

        velos = vmap(estimate_velocity_pixel, in_axes=(None, -1, -1))(
            x_arr, unw_pixels, weights_pixels
        )
    # Currently `velos` is in units / day,
    days_per_year = 365.25
    return velos.reshape(n_rows, n_cols) * days_per_year


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


def create_velocity(
    unw_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    reference: ReferencePoint,
    date_list: Sequence[DateOrDatetime] | None = None,
    cor_file_list: Sequence[PathOrStr] | None = None,
    cor_threshold: float = 0.2,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    add_overviews: bool = True,
) -> None:
    """Perform pixel-wise (weighted) linear regression to estimate velocity.

    The units of `output_file` are in (unwrapped units) / year.
    E.g. if the files in `unw_file_list` are in radians, the output velocity
    is in radians / year, which is calculated as 365.25 * radians / day.

    Parameters
    ----------
    unw_file_list : Sequence[PathOrStr]
        List of unwrapped phase files.
    output_file : PathOrStr
        The output file to save the velocity to.
    reference : ReferencePoint
        The (row, col) to use as reference before fitting the velocity.
        This point will be subtracted from all other points before solving.
    date_list : Sequence[DateOrDatetime], optional
        List of dates corresponding to the unwrapped phase files.
        If not provided, will be parsed from filenames in `unw_file_list`.
    cor_file_list : Sequence[PathOrStr], optional
        List of correlation files to use for weighting the velocity estimation.
        If not provided, all weights are set to 1.
    cor_threshold : float, optional
        The correlation threshold to use for weighting the velocity estimation.
        Default is 0.2.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256)
    num_threads : int, optional
        The parallel blocks to process at once.
        Default is 4.
    add_overviews : bool, optional
        If True, creates overviews of the new velocity raster.
        Default is True.

    """
    if Path(output_file).exists():
        logger.info(f"Output file {output_file} already exists, skipping velocity")
        return

    if date_list is None:
        date_list = [get_dates(f)[1] for f in unw_file_list]
    x_arr = datetime_to_float(date_list)

    # Set up the input data readers
    out_dir = Path(output_file).parent
    unw_reader = io.VRTStack(
        file_list=unw_file_list,
        outfile=out_dir / "velocity_inputs.vrt",
        skip_size_check=True,
    )
    if cor_file_list is not None:
        if len(cor_file_list) != len(unw_file_list):
            msg = "Mismatch in number of input files provided:"
            msg += f"{len(cor_file_list) = }, but {len(unw_file_list) = }"
            raise ValueError(msg)

        logger.info("Using correlation to weight velocity fit")
        cor_reader = io.VRTStack(
            file_list=cor_file_list,
            outfile=out_dir / "cor_inputs.vrt",
            skip_size_check=True,
        )
    else:
        logger.info("Using unweighted fit for velocity.")
        cor_reader = None

    # Read in the reference point
    ref_row, ref_col = reference
    logger.info(f"Reading phase reference pixel {reference}")
    ref_data = unw_reader[:, ref_row, ref_col].reshape(-1, 1, 1)

    def read_and_fit(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[np.ndarray, slice, slice]:
        # Only use the cor_reader if it's the same shape as the unw_reader
        if len(readers) == 2:
            unw_reader, cor_reader = readers
            unw_stack = unw_reader[:, rows, cols]
            weights = cor_reader[:, rows, cols]
            weights[weights < cor_threshold] = 0
        else:
            unw_stack = readers[0][:, rows, cols]
            # weights = np.ones_like(unw_stack)
            weights = None
        # Reference the data
        unw_stack = unw_stack - ref_data
        # Fit a line to each pixel with weighted least squares
        return (
            estimate_velocity(x_arr=x_arr, unw_stack=unw_stack, weight_stack=weights),
            rows,
            cols,
        )

    # Note: For some reason, the `RasterStackReader` is much slower than the VRT
    # for files on S3:
    # ~300 files takes >2 min to open, >2 min to read each block
    # VRTStack seems to take ~30 secs to open, 1 min to read
    # Very possible there's a tuning param/rasterio config to fix, but not sure.
    readers = [unw_reader]
    if cor_reader is not None:
        readers.append(cor_reader)
    writer = io.BackgroundRasterWriter(output_file, like_filename=unw_file_list[0])
    io.process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_fit,
        block_shape=block_shape,
        num_threads=num_threads,
    )

    writer.notify_finished()
    if add_overviews:
        logger.info("Creating overviews for velocity image")
        create_overviews([output_file])
    if units := io.get_raster_units(unw_file_list[0]):
        io.set_raster_units(output_file, units=units)
    logger.info("Completed create_velocity")


class AverageFunc(Protocol):
    """Protocol for temporally averaging a block of data."""

    def __call__(self, ArrayLike, axis: int) -> ArrayLike: ...


def create_temporal_average(
    file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    average_func: Callable[[ArrayLike, int], np.ndarray] = np.nanmean,
    read_masked: bool = False,
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
        Default is (256, 256)
    num_threads : int, optional
        The parallel blocks to process at once.
        Default is 4.
    average_func : Callable[[ArrayLike, int], np.ndarray], optional
        The function to use to average the images.
        Default is `np.nanmean`, which calls `np.nanmean(arr, axis=0)` on each block.
    read_masked : bool, optional
        If True, reads the data as a masked array based on the rasters' nodata values.
        Default is False.

    """

    def read_and_average(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        chunk = readers[0][:, rows, cols]
        return average_func(chunk, 0), rows, cols

    writer = io.BackgroundRasterWriter(output_file, like_filename=file_list[0])
    with NamedTemporaryFile(mode="w", suffix=".vrt") as f:
        reader = io.VRTStack(
            file_list=file_list,
            outfile=f.name,
            skip_size_check=True,
            read_masked=read_masked,
        )

        io.process_blocks(
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
    n_cor_looks: int = 1,
    ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
    wavelength: float | None = None,
    method: InversionMethod = InversionMethod.L2,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
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
    cor_file_list : Sequence[PathOrStr], optional
        List of correlation files to use for weighting the inversion
    cor_threshold : float, optional
        The correlation threshold to use for weighting the inversion
        Default is 0.2
    n_cor_looks : int, optional
        The number of looks used to form the input correlation data, used
        to convert correlation to phase variance.
        Default is 1.
    ifg_date_pairs : Sequence[Sequence[DateOrDatetime]], optional
        List of date pairs to use for the inversion. If not provided, will be
        parsed from filenames in `unw_file_list`.
    method : str, choices = "L1", "L2"
        Inversion method to use when solving Ax = b.
        Default is L2, which uses least squares to solve Ax = b (faster).
        "L1" minimizes |Ax - b|_1 at each pixel.
    wavelength : float, optional
        The wavelength of the radar signal, in meters.
        If provided, the output rasters are in meters.
        If not provided, the outputs are in radians.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256).
    num_threads : int
        The parallel blocks to process at once.
        Default is 4.

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
    # Create the `n_sar_dates - 1` output files (skipping the 0 reference raster)
    out_paths = [
        Path(output_dir) / (format_dates(ref_date, d) + suffix) for d in sar_dates[1:]
    ]
    if all(p.exists() for p in out_paths):
        logger.info("All output files already exist, skipping inversion")
        return out_paths
    logger.info(f"Inverting network using {method.upper()}-norm minimization")

    A = get_incidence_matrix(ifg_pairs=ifg_tuples, sar_idxs=sar_dates)

    out_vrt_name = Path(output_dir) / "unw_network.vrt"
    unw_reader = io.VRTStack(
        file_list=unw_file_list, outfile=out_vrt_name, skip_size_check=True
    )
    cor_vrt_name = Path(output_dir) / "cor_network.vrt"

    # Get the reference point data
    ref_row, ref_col = reference
    ref_data = unw_reader[:, ref_row, ref_col].reshape(-1, 1, 1)

    if wavelength is not None:
        # Positive values are motion towards the radar
        constant = -1 * (wavelength / (4 * np.pi))
        units = "meters"
    else:
        constant = -1
        units = "radians"

    def read_and_solve(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        if len(readers) == 2 and method == "L2":
            unw_reader, cor_reader = readers
            stack = unw_reader[:, rows, cols]
            cor = cor_reader[:, rows, cols]
            weights = correlation_to_variance(cor, n_cor_looks)
            weights[cor < cor_threshold] = 0
        else:
            stack = readers[0][:, rows, cols]
            weights = None

        # subtract the reference
        stack = stack - ref_data

        # TODO: possible second input for weights? from conncomps
        # TODO: do i want to write residuals too? Do i need
        # to have multiple writers then?
        phases = invert_stack(A, stack, weights)[0]
        if method.upper() == "L1":
            phases = invert_stack_l1(A, stack)[0]
        else:
            phases = invert_stack(A, stack, weights)[0]
        # Convert to meters, with LOS convention:
        return constant * np.asarray(phases), rows, cols

    if cor_file_list is not None:
        cor_reader = io.VRTStack(
            file_list=cor_file_list, outfile=cor_vrt_name, skip_size_check=True
        )
        readers = [unw_reader, cor_reader]
        logger.info("Using correlation to weight unw inversion")
    else:
        readers = [unw_reader]
        logger.info("Using unweighted unw inversion")

    writer = io.BackgroundStackWriter(
        out_paths, like_filename=unw_file_list[0], units=units
    )

    io.process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_solve,
        block_shape=block_shape,
        num_threads=num_threads,
    )
    writer.notify_finished()

    logger.info("Completed invert_unw_network")
    return out_paths


def correlation_to_variance(correlation: ArrayLike, nlooks: int) -> Array:
    r"""Convert interferometric correlation to phase variance.

    Uses the CRLB formula from Rodriguez, 1992 [1]_ to get the phase variance,
    \sigma_{\phi}^2:

    \[
        \sigma_{\phi}^{2} = \frac{1}{2N_{L}} \frac{1 - \gamma^{2}}{\gamma^{2}}
    \]

    where \gamma is the correlation and N_L is the effective number of looks.

    References
    ----------
    .. [1] Rodriguez, E., and J. M. Martin. "Theory and design of interferometric
    synthetic aperture radars." IEEE Proceedings of Radar and Signal Processing.
    Vol. 139. No. 2. IET Digital Library, 1992.

    """
    return (1 - correlation**2) / (2 * nlooks * correlation**2 + 1e-6)


def select_reference_point(
    *,
    condition_file: PathOrStr,
    output_dir: Path,
    condition_func: Callable[[ArrayLike], tuple[int, ...]] = argmin_index,
    ccl_file_list: Sequence[PathOrStr] | None = None,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> ReferencePoint:
    """Automatically select a reference point for a stack of unwrapped interferograms.

    Uses the condition file and (optionally) connected component labels.
    The point is selected which

    1. has the condition applied to condition file. for example: has the lowest
       amplitude dispersion
    2. (optionally) is within intersection of all nonzero connected component labels

    Parameters
    ----------
    condition_file: PathOrStr
        A file with the same size as each raster, like amplitude dispersion or
        temporal coherence in `ccl_file_list`
    output_dir: Path
        Path to store the computed "conncomp_intersection.tif" raster
    condition_func: Callable[[ArrayLike, ]]
        The function to apply to the condition file,
        for example numpy.argmin which finds the pixel with lowest value
    ccl_file_list : Sequence[PathOrStr]
        List of connected component label phase files.
    block_shape : tuple[int, int]
        Size of blocks to read from while processing `ccl_file_list`
        Default = (256, 256)
    num_threads: int
        Number of parallel blocks to process.
        Default = 5

    Returns
    -------
    ReferencePoint
        The select (row, col) as a namedtuple

    Raises
    ------
    ReferencePointError
        Raised if no valid region is found in the intersection of the connected
        component label files

    """
    output_file = output_dir / "reference_point.txt"
    if output_file.exists():
        ref_point = _read_reference_point(output_file=output_file)
        logger.info(f"Read {ref_point!r} from existing {output_file}")
        return ref_point

    logger.info("Selecting reference point")
    condition_file_values = io.load_gdal(condition_file, masked=True)

    isin_largest_conncomp = np.ones(condition_file_values.shape, dtype=bool)
    if ccl_file_list:
        try:
            isin_largest_conncomp = _get_largest_conncomp_mask(
                ccl_file_list=ccl_file_list,
                output_dir=output_dir,
                block_shape=block_shape,
                num_threads=num_threads,
            )
        except ReferencePointError:
            msg = "Unable to find find a connected component intersection."
            msg += f"Proceeding using only {condition_file = }"
            logger.warning(msg, exc_info=True)

    # Mask out where the conncomps aren't equal to the largest
    condition_file_values.mask = condition_file_values.mask | (~isin_largest_conncomp)

    # Pick the (unmasked) point with the condition applied to condition file
    ref_row, ref_col = condition_func(condition_file_values)

    # Cast to `int` to avoid having `np.int64` types
    ref_point = ReferencePoint(int(ref_row), int(ref_col))
    logger.info(f"Saving {ref_point!r} to {output_file}")
    _write_reference_point(output_file=output_file, ref_point=ref_point)
    return ref_point


def _write_reference_point(output_file: Path, ref_point: ReferencePoint) -> None:
    output_file.write_text(",".join(list(map(str, ref_point))))


def _read_reference_point(output_file: Path):
    return ReferencePoint(*[int(n) for n in output_file.read_text().split(",")])


def _get_largest_conncomp_mask(
    output_dir: Path,
    ccl_file_list: Sequence[PathOrStr] | None = None,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> np.ndarray:
    def intersect_conncomp(arr: np.ma.MaskedArray, axis: int) -> np.ndarray:
        # Track where input is nodata
        any_masked = np.any(arr.mask, axis=axis)
        # Get the logical AND of all nonzero conncomp labels
        fillval = arr.fill_value
        is_valid_conncomp = arr.filled(0) > 0
        all_are_valid = np.all(is_valid_conncomp, axis=axis).astype(arr.dtype)
        # Reset nodata
        all_are_valid[any_masked] = fillval
        return all_are_valid

    conncomp_intersection_file = Path(output_dir) / "conncomp_intersection.tif"
    if ccl_file_list and not conncomp_intersection_file.exists():
        logger.info("Creating intersection of connected components")
        create_temporal_average(
            file_list=ccl_file_list,
            output_file=conncomp_intersection_file,
            block_shape=block_shape,
            num_threads=num_threads,
            average_func=intersect_conncomp,
            read_masked=True,
        )

    conncomp_intersection = io.load_gdal(conncomp_intersection_file, masked=True)

    # Find the largest conncomp region in the intersection
    label, nlabels = ndimage.label(
        conncomp_intersection.filled(0), structure=np.ones((3, 3))
    )
    if nlabels == 0:
        raise ReferencePointError(
            "Connected components intersection left no valid regions"
        )
    logger.info("Found %d connected components in intersection", nlabels)

    # Make a mask of the largest conncomp:
    # Find the label with the most pixels using bincount
    label_counts = np.bincount(label.ravel())
    # (ignore the 0 label)
    largest_idx = np.argmax(label_counts[1:]) + 1
    # Create a mask of pixels with this label
    isin_largest_conncomp = label == largest_idx
    return isin_largest_conncomp


@jax.jit
def _shrinkage(a: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """Apply the shrinkage operator element-wise."""
    return jnp.maximum(0, a - kappa) - jnp.maximum(0, -a - kappa)


@jax.jit
def least_absolute_deviations(
    A: jnp.ndarray,
    b: jnp.ndarray,
    R: jnp.ndarray,
    rho: float = 0.4,
    alpha: float = 1.0,
    max_iter: int = 20,
) -> jnp.ndarray:
    """Solve Least Absolute Deviations (LAD) via ADMM.

    Solves the following problem via ADMM:

        minimize ||Ax - b||_1

    See [@Boyd2010DistributedOptimizationStatistical] for more on ADMM.

    Parameters
    ----------
    A : jnp.ndarray
        The matrix A in the equation Ax = b.
        Shape is (M, N)
    b : jnp.ndarray
        The vector b in the equation Ax = b.
        Shape is (M,)
    R : jnp.ndarray
        Precomputed lower-triangular Cholesky factor of A^T A.
        Shape is (N, N)
    rho : float, optional
        The augmented Lagrangian parameter
        By default 0.4.
    alpha : float, optional
        The over-relaxation parameter (typical values are between 1.0 and 1.8)
        By default 1.0.
    max_iter : int, optional
        The maximum number of iterations, by default 15.

    Returns
    -------
    x_solution : jnp.ndarray
        The solution vector x of shape (N, )
    residual : jnp.ndarray, scalar
        The objective residual `sum(abs(b - A @ x_solution))`


    Notes
    -----
    The implementation is based on the [MATLAB implementation of LAD here](https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html).
    One caveat is that there are a fixed number of iterations used for this
    problem here. Inverting interferogram network have a very similar structure
    each time, and the results converge quickly relative to other large-scale
    LAD problems which ADMM can solve.

    """
    m, n = A.shape
    x0 = jnp.zeros(n)
    z0 = jnp.zeros(m)
    u0 = jnp.zeros(m)

    def body_fun(_i, state):
        x, z, z_old, u = state
        # x-update
        q = A.T @ (b + z - u)
        x = jax.scipy.linalg.cho_solve((R, True), q)

        # z-update with relaxation
        Ax_hat = alpha * (A @ x) + (1 - alpha) * (z_old + b)
        z_new = _shrinkage(Ax_hat - b + u, 1 / rho)

        # u-update
        u = u + Ax_hat - z_new - b

        return x, z_new, z, u

    x_final, _, _, _ = lax.fori_loop(0, max_iter, body_fun, (x0, z0, z0, u0))
    residual = jnp.sum(jnp.abs(b - A @ x_final))
    return x_final, residual


@jit
def invert_stack_l1(A: ArrayLike, dphi: ArrayLike) -> Array:
    R = jax.scipy.linalg.cholesky(A.T @ A, lower=True)

    # vectorize the solve function to work on 2D and 3D arrays
    # We are not vectorizing over the A matrix, only the dphi vector
    # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
    invert_2d = vmap(
        least_absolute_deviations, in_axes=(None, 1, None), out_axes=(1, 0)
    )
    # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
    invert_3d = vmap(invert_2d, in_axes=(None, 2, None), out_axes=(2, 1))
    phase, residuals = invert_3d(A, dphi, R)
    # Reshape the residuals to be 2D
    # residuals = jnp.sum(residual_vecs, axis=0)

    return phase, residuals
