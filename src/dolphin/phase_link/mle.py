import logging
from typing import Dict, Optional, Tuple

import numpy as np

from dolphin._types import Filename
from dolphin.utils import (
    check_gpu_available,
    get_array_module,
    take_looks,
    upsample_nearest,
)

logger = logging.getLogger(__name__)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""

    pass


def run_mle(
    slc_stack: np.ndarray,
    half_window: Dict[str, int],
    strides: Dict[str, int] = {"x": 1, "y": 1},
    beta: float = 0.0,
    reference_idx: int = 0,
    mask: np.ndarray = None,
    ps_mask: Optional[np.ndarray] = None,
    avg_mag: Optional[np.ndarray] = None,
    output_cov_file: Optional[Filename] = None,
    n_workers: int = 1,
    gpu_enabled: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the linked phase for a stack using the MLE estimator.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : Dict[str, int]
        The half window size as {"x": half_win_x, "y": half_win_y}
        The full window size is 2 * half_window + 1 for x, y.
    strides : Dict[str, int], optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    beta : float, optional
        The regularization parameter, by default 0.0.
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    mask : np.ndarray, optional
        A mask of bad pixels to ignore when estimating the covariance.
        Pixels with `True` (or 1) are ignored, by default None
        If None, all pixels are used, by default None.
    ps_mask : np.ndarray, optional
        A mask of pixels marking persistent scatterers (PS) to
        also ignore when multilooking.
        Pixels with `True` (or 1) are PS and will be ignored (combined
        with `mask`).
        The phase from these pixels will be inserted back
        into the final estimate directly from `slc_stack`.
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to estimate the
        to find the brights PS pixels to fill within each look window.
        If None, the average magnitude is estimated from the SLC stack.
        By default None.
    output_cov_file : str, optional
        HDF5 filename to save the estimated covariance at each pixel.
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.
    gpu_enabled : bool, optional
        If False, do not use the GPU, even if it is available.

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_images, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    from ._mle_cpu import run_cpu as _run_cpu
    from ._mle_gpu import run_gpu as _run_gpu

    num_slc, rows, cols = slc_stack.shape
    # Common pre-processing for both CPU and GPU versions:

    # Mask nodata pixels if given
    if mask is None:
        mask = np.zeros((rows, cols), dtype=bool)
    else:
        mask = mask.astype(bool)
    # Make sure we also are ignoring pixels which are nans for all SLCs
    mask |= np.all(np.isnan(slc_stack), axis=0)

    # Track the PS pixels, if given, and remove them from the stack
    # This will prevent the large amplitude PS pixels from dominating
    # the covariance estimation.
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    _check_all_nans(slc_stack)

    # TODO: Any other masks we need?
    ignore_mask = np.logical_or.reduce((mask, ps_mask))

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_copy = slc_stack.copy()
    slc_stack_copy[:, ignore_mask] = np.nan

    #######################################
    gpu_is_available = check_gpu_available()
    if not gpu_enabled or not gpu_is_available:
        mle_est, temp_coh = _run_cpu(
            slc_stack,
            half_window,
            strides,
            beta,
            reference_idx,
            output_cov_file,
            n_workers=n_workers,
        )
    else:
        mle_est, temp_coh = _run_gpu(
            slc_stack,
            half_window,
            strides,
            beta,
            reference_idx,
            output_cov_file,
            # is it worth passing the blocks-per-grid?
        )

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(mask, strides["y"], strides["x"], func_type="all")
    # Set no data pixels to np.nan
    temp_coh[mask_looked] = np.nan

    if avg_mag is None:
        # Get the average magnitude of the SLC stack
        avg_mag = np.abs(slc_stack).mean(axis=0)
    # null out all the non-PS pixels
    avg_mag[~ps_mask] = np.nan

    # Get the indices of the maxes within each look window
    slc_r_idxs, slc_c_idxs = _get_maxes(avg_mag, strides["y"], strides["x"])

    # For ps_mask, we set to True if any pixels within the window were PS
    ps_mask_looked = take_looks(ps_mask, strides["y"], strides["x"], func_type="any")

    # Fill in the PS pixels from the original SLC stack, if it was given
    if np.any(ps_mask_looked):
        # ref = np.conj(slc_stack[0][ps_mask])
        ref = np.conj(slc_stack[0][slc_r_idxs, slc_c_idxs])
        for i in range(num_slc):
            mle_est[i][ps_mask_looked] = slc_stack[i][slc_r_idxs, slc_c_idxs] * ref
        # Force PS pixels to have high temporal coherence
        temp_coh[ps_mask_looked] = 1

    return mle_est, temp_coh


def _get_maxes(arr, row_looks, col_looks):
    # Get the max value in each look window
    # We have to pad so that the shapes are a multiple of the look sizes
    # Otherwise, the maxes sizes will be off
    max_nums = take_looks(
        arr, row_looks, col_looks, func_type="nanmax", edge_strategy="pad"
    )
    # Repeat the max values to back to the original size
    maxes_filled = upsample_nearest(
        max_nums, arr.shape[-2:], looks=(row_looks, col_looks)
    )
    # Find the indices of the max values in the original image
    return np.where(maxes_filled == arr)


def mle_stack(C_arrays, beta: float = 0.0, reference_idx: float = 0):
    """Estimate the linked phase for a stack of covariance matrices.

    This function is used for both the CPU and GPU versions after
    covariance estimation.
    Will use cupy if available, (and if the input is a GPU array).
    Otherwise, uses numpy (for CPU version).

    Parameters
    ----------
    C_arrays : ndarray, shape = (rows, cols, nslc, nslc)
        The sample covariance matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance_cpu][])
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        If the SLC stack from which `C_arrays` was computed contained
        compressed SLCs at the stack, then this should be the index
        of the first non-compressed SLC.

    Returns
    -------
    ndarray, shape = (nslc, rows, cols)
        The estimated linked phase, same shape as the input slcs (possibly multilooked)

    References
    ----------
        [1] Ansari, H., De Zan, F., & Bamler, R. (2018). Efficient phase
        estimation for interferogram stacks. IEEE Transactions on
        Geoscience and Remote Sensing, 56(7), 4109-4125.

    """
    xp = get_array_module(C_arrays)
    # estimate the wrapped phase based on the EMI paper
    # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix
    Gamma = xp.abs(C_arrays)

    if beta > 0:
        # Perform regularization
        Id = xp.eye(Gamma.shape[-1], dtype=Gamma.dtype)
        # repeat the identity matrix for each pixel
        Id = xp.tile(Id, (Gamma.shape[0], Gamma.shape[1], 1, 1))
        Gamma = (1 - beta) * Gamma + beta * Id

    Gamma_inv = xp.linalg.inv(Gamma)
    _, V = xp.linalg.eigh(Gamma_inv * C_arrays)

    # The shape of V is (rows, cols, nslc, nslc)
    # at pixel (r, c), the columns of V[r, c] are the eigenvectors.
    # They're ordered by increasing eigenvalue, so the first column is the
    # eigenvector corresponding to the smallest eigenvalue (our phase solution).
    evd_estimate = V[:, :, :, 0]
    # The phase estimate on the reference day will be size (rows, cols)
    ref = evd_estimate[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = evd_estimate * xp.conjugate(ref[:, :, None])

    # Return the phase (still as a GPU array)
    phase_stack = xp.angle(evd_estimate)
    # Move the SLC dimension to the front (to match the SLC stack shape)
    return np.moveaxis(phase_stack, -1, 0)


def _check_all_nans(slc_stack):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        raise PhaseLinkRuntimeError(f"SLC stack[{bad_slc_idxs}] has are all NaNs.")
