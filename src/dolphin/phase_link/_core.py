from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import get_array_module, take_looks

from ._cpl import run_cpl

logger = logging.getLogger(__name__)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""


class MleOutput(NamedTuple):
    """Output of the MLE solver."""

    mle_est: Array
    """Estimated linked phase."""

    temp_coh: Array
    """Temporal coherence."""

    avg_coh: Array | None = None
    """Average coherence across dates for each SLC."""


DEFAULT_STRIDES = Strides(1, 1)


def run_phase_linking(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    use_evd: bool = False,
    beta: float = 0.01,
    reference_idx: int = 0,
    nodata_mask: np.ndarray = None,
    ps_mask: Optional[np.ndarray] = None,
    neighbor_arrays: Optional[np.ndarray] = None,
    avg_mag: Optional[np.ndarray] = None,
    use_slc_amp: bool = True,
) -> MleOutput:
    """Estimate the linked phase for a stack using the MLE estimator.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0.01.
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    nodata_mask : np.ndarray, optional
        A mask of bad/nodata pixels to ignore when estimating the covariance.
        Pixels with `True` (or 1) are ignored, by default None
        If None, all pixels are used, by default None.
    ps_mask : np.ndarray, optional
        A mask of pixels marking persistent scatterers (PS) to
        skip when multilooking.
        Pixels with `True` (or 1) are PS and will be ignored
        (combined with `nodata_mask`).
        The phase from these pixels will be inserted back
        into the final estimate directly from `slc_stack`.
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
        If None, the average magnitude will be computed from `slc_stack`.
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_images, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    If `calc_average_coh` is True, `avg_coh` will also be returned.
    """
    _, rows, cols = slc_stack.shape
    # Common pre-processing for both CPU and GPU versions:

    # Mask nodata pixels if given
    if nodata_mask is None:
        nodata_mask = np.zeros((rows, cols), dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool)

    # Track the PS pixels, if given, and remove them from the stack
    # This will prevent the large amplitude PS pixels from dominating
    # the covariance estimation.
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    _check_all_nans(slc_stack)

    # Make sure we also are ignoring pixels which are nans for all SLCs
    if nodata_mask.shape != (rows, cols) or ps_mask.shape != (rows, cols):
        msg = (
            f"nodata_mask.shape={nodata_mask.shape}, ps_mask.shape={ps_mask.shape},"
            f" but != SLC (rows, cols) {rows, cols}"
        )
        raise ValueError(msg)
    # for any area that has nans in the SLC stack, mark it as nodata
    nodata_mask |= np.any(np.isnan(slc_stack), axis=0)
    # Make sure the PS mask didn't have extra burst borders that are nodata here
    ps_mask[nodata_mask] = False

    # TODO: Any other masks we need?
    ignore_mask = np.logical_or.reduce((nodata_mask, ps_mask))

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, ignore_mask] = np.nan

    # mle_est, temp_coh, avg_coh = run_cpl(
    mle_out = run_cpl(
        slc_stack=slc_stack_masked,
        half_window=half_window,
        strides=strides,
        use_evd=use_evd,
        beta=beta,
        reference_idx=reference_idx,
        neighbor_arrays=neighbor_arrays,
        use_slc_amp=use_slc_amp,
    )

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(nodata_mask, *strides, func_type="all")

    # Set no data pixels to np.nan
    mle_out.temp_coh[mask_looked] = np.nan

    # Fill in the PS pixels from the original SLC stack, if it was given
    if np.any(ps_mask):
        _fill_ps_pixels(
            mle_out.mle_est,
            mle_out.temp_coh,
            slc_stack,
            ps_mask,
            strides,
            avg_mag,
            reference_idx,
        )

    return mle_out


def mle_stack(
    C_arrays,
    use_evd: bool = False,
    beta: float = 0.01,
    reference_idx: int = 0,
):
    """Estimate the linked phase for a stack of covariance matrices.

    This function is used for both the CPU and GPU versions after
    covariance estimation.
    Will use cupy if available, (and if the input is a GPU array).
    Otherwise, uses numpy (for CPU version).


    Parameters
    ----------
    C_arrays : ndarray, shape = (rows, cols, nslc, nslc)
        The sample covariance matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance][])
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm of [@Ansari2018EfficientPhaseEstimation].
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
        Default is 0.01.
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        All outputs are multiplied by the conjugate of the data at this index.

    Returns
    -------
    ndarray, shape = (nslc, rows, cols)
        The estimated linked phase, same shape as the input slcs (possibly multilooked)
    """
    xp = get_array_module(C_arrays)
    # estimate the wrapped phase based on the EMI paper
    # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix
    Gamma = xp.abs(C_arrays)

    if use_evd:
        V = _get_eigvecs(C_arrays, use_evd=True)
        column_idx = -1
    else:
        if beta > 0:
            # Perform regularization
            Id = xp.eye(Gamma.shape[-1], dtype=Gamma.dtype)
            # repeat the identity matrix for each pixel
            Id = xp.tile(Id, (Gamma.shape[0], Gamma.shape[1], 1, 1))
            Gamma = (1 - beta) * Gamma + beta * Id

        Gamma_inv = xp.linalg.inv(Gamma)
        V = _get_eigvecs(Gamma_inv * C_arrays, use_evd=False)
        column_idx = 0

    # The shape of V is (rows, cols, nslc, nslc)
    # at pixel (r, c), the columns of V[r, c] are the eigenvectors.
    # They're ordered by increasing eigenvalue, so the first column is the
    # eigenvector corresponding to the smallest eigenvalue (phase solution for EMI),
    # and the last column is for the largest eigenvalue (used by EVD)
    evd_estimate = V[:, :, :, column_idx]

    # The phase estimate on the reference day will be size (rows, cols)
    ref = evd_estimate[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = evd_estimate * xp.conjugate(ref[:, :, None])

    # Return the phase (still as a GPU array)
    phase_stack = xp.angle(evd_estimate)
    # Move the SLC dimension to the front (to match the SLC stack shape)
    return xp.moveaxis(phase_stack, -1, 0)


@partial(jit, static_argnames=("use_evd",))
def _get_eigvecs(C_arrays: ArrayLike, use_evd: bool = False) -> Array:
    # Subset index for scipy.eigh: larges eig for EVD. Smallest for EMI.
    subset_idx = C_arrays.shape[-1] - 1 if use_evd else 0

    def get_top_eigvecs(C: Array):
        # The eigenvalues in ascending order, each repeated according
        # The column ``eigenvectors[:, i]`` is the normalized eigenvector
        # corresponding to the eigenvalue ``eigenvalues[i]``.
        return jnp.linalg.eigh(C)[1][:, [subset_idx]]

    # vmap over the first 2 dimensions (rows, cols)
    get_eigvecs_block = vmap(vmap(get_top_eigvecs))
    return get_eigvecs_block(C_arrays)


def _check_all_nans(slc_stack: np.ndarray):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        msg = f"slc_stack[{bad_slc_idxs}] out of {len(slc_stack)} are all NaNs."
        raise PhaseLinkRuntimeError(msg)


def _fill_ps_pixels(
    mle_est: np.ndarray,
    temp_coh: np.ndarray,
    slc_stack: np.ndarray,
    ps_mask: np.ndarray,
    strides: Strides,
    avg_mag: np.ndarray,
    reference_idx: int = 0,
    use_max_ps: bool = False,
):
    """Fill in the PS locations in the MLE estimate with the original SLC data.

    Overwrites `mle_est` and `temp_coh` in place.

    Parameters
    ----------
    mle_est : ndarray, shape = (nslc, rows, cols)
        The complex valued-MLE estimate of the phase.
    temp_coh : ndarray, shape = (rows, cols)
        The temporal coherence of the estimate.
    slc_stack : np.ndarray
        The original SLC stack, with shape (n_images, n_rows, n_cols)
    ps_mask : ndarray, shape = (rows, cols)
        Boolean mask of pixels marking persistent scatterers (PS).
    strides : Strides
        The decimation (y, x) factors
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
    reference_idx : int, default = 0
        SLC to use as reference for PS pixels. All pixel values are multiplied
        by the conjugate of this index
    use_max_ps : bool, optional
        If True, use the brightest PS pixel in each look window to fill in the
        MLE estimate. If False, use the average of all PS pixels in each look window.

    Returns
    -------
    ps_masked_looked : ndarray
        boolean array of PS, multilooked (using "any") to same size as `mle_est`
    """
    if avg_mag is None:
        # Get the average magnitude of the SLC stack
        # nanmean will ignore single NaNs, but not all NaNs, per pixel
        with warnings.catch_warnings():
            # ignore the warning about nansum/nanmean of empty slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_mag = np.nanmean(np.abs(slc_stack), axis=0)
    mag = avg_mag.copy()

    # null out all the non-PS pixels when finding the brightest PS pixels
    mag[~ps_mask] = np.nan
    # For ps_mask, we set to True if any pixels within the window were PS
    ps_mask_looked = take_looks(ps_mask, *strides, func_type="any", edge_strategy="pad")
    # make sure it's the same size as the MLE result/temp_coh after padding
    ps_mask_looked = ps_mask_looked[: mle_est.shape[1], : mle_est.shape[2]]

    if use_max_ps:
        print("Using max PS pixel to fill in MLE estimate")
        # Get the indices of the brightest pixels within each look window
        slc_r_idxs, slc_c_idxs = _get_max_idxs(mag, *strides)
        # we're only filling where there are PS pixels
        ref = np.exp(-1j * np.angle(slc_stack[reference_idx][slc_r_idxs, slc_c_idxs]))
        for i in range(len(slc_stack)):
            mle_est[i][ps_mask_looked] = slc_stack[i][slc_r_idxs, slc_c_idxs] * ref
    else:
        # Get the average of all PS pixels within each look window
        # The referencing to SLC 0 is done in _get_avg_ps
        avg_ps = _get_avg_ps(slc_stack, ps_mask, strides)[
            :, : mle_est.shape[1], : mle_est.shape[2]
        ]
        mle_est[:, ps_mask_looked] = avg_ps[:, ps_mask_looked]

    # Force PS pixels to have high temporal coherence
    temp_coh[ps_mask_looked] = 1


def _get_avg_ps(
    slc_stack: np.ndarray, ps_mask: np.ndarray, strides: Strides
) -> np.ndarray:
    # First, set all non-PS pixels to NaN
    slc_stack_nanned = slc_stack.copy()
    slc_stack_nanned[:, ~ps_mask] = np.nan
    # Reference all ps pixels in the SLC stack to the first SLC
    slc_stack_nanned[:, ps_mask] *= np.exp(
        -1j * np.angle(slc_stack_nanned[0, ps_mask])
    )[None]
    # Then, take the average of all PS pixels within each look window
    return take_looks(
        slc_stack_nanned,
        *strides,
        func_type="nanmean",
        edge_strategy="pad",
    )


def _get_max_idxs(arr, row_looks, col_looks):
    """Get the indices of the maximum value in each look window."""
    if row_looks == 1 and col_looks == 1:
        # No need to pad if we're not looking
        return np.where(arr == arr)
    # Adjusted from this answer to not take every moving window
    # https://stackoverflow.com/a/72742009/4174466
    windows = np.lib.stride_tricks.sliding_window_view(arr, (row_looks, col_looks))[
        ::row_looks, ::col_looks
    ]
    maxvals = np.nanmax(windows, axis=(2, 3))
    indx = np.array((windows == np.expand_dims(maxvals, axis=(2, 3))).nonzero())

    # In [82]: (windows == np.expand_dims(maxvals, axis = (2, 3))).nonzero()
    # This gives 4 arrays:
    # First two are the window indices
    # (array([0, 0, 0, 1, 1, 1]),
    # array([0, 1, 2, 0, 1, 2]),
    # last two are the relative indices (within each window)
    # array([0, 0, 1, 1, 1, 1]),
    # array([1, 1, 1, 1, 1, 0]))
    window_positions, relative_positions = indx.reshape((2, 2, -1))
    # Multiply the first two by the window size to get the absolute indices
    # of the top lefts of the windows
    window_offsets = np.array([row_looks, col_looks]).reshape((2, 1))
    # Then add the last two to get the relative indices
    rows, cols = relative_positions + window_positions * window_offsets
    return rows, cols
