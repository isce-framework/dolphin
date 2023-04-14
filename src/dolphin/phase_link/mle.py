import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pymp
from scipy.linalg import eigh

from dolphin.utils import get_array_module, gpu_is_available, take_looks

logger = logging.getLogger(__name__)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""

    pass


def run_mle(
    slc_stack: np.ndarray,
    half_window: Dict[str, int],
    strides: Dict[str, int] = {"x": 1, "y": 1},
    beta: float = 0.01,
    reference_idx: int = 0,
    nodata_mask: np.ndarray = None,
    ps_mask: Optional[np.ndarray] = None,
    avg_mag: Optional[np.ndarray] = None,
    use_slc_amp: bool = True,
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
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
        If None, the average magnitude is estimated from the SLC stack.
        By default None.
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.
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
        raise ValueError(
            f"nodata_mask.shape={nodata_mask.shape}, ps_mask.shape={ps_mask.shape},"
            f" but != SLC (rows, cols) {rows, cols}"
        )
    nodata_mask |= np.all(np.isnan(slc_stack), axis=0)
    # TODO: Any other masks we need?
    ignore_mask = np.logical_or.reduce((nodata_mask, ps_mask))

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, ignore_mask] = np.nan

    #######################################
    if not gpu_enabled or not gpu_is_available():
        mle_est, temp_coh = _run_cpu(
            slc_stack_masked,
            half_window,
            strides,
            beta,
            reference_idx,
            use_slc_amp,
            n_workers=n_workers,
        )
    else:
        mle_est, temp_coh = _run_gpu(
            slc_stack_masked,
            half_window,
            strides,
            beta,
            reference_idx,
            use_slc_amp,
            # is it worth passing the blocks-per-grid?
        )

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(nodata_mask, strides["y"], strides["x"], func_type="all")
    # Set no data pixels to np.nan
    temp_coh[mask_looked] = np.nan

    # Fill in the PS pixels from the original SLC stack, if it was given
    if np.any(ps_mask):
        _fill_ps_pixels(mle_est, temp_coh, slc_stack, ps_mask, strides, avg_mag)

    return mle_est, temp_coh


def mle_stack(
    C_arrays, beta: float = 0.01, reference_idx: float = 0, n_workers: int = 1
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
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance_cpu][])
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
        Default is 0.01.
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        If the SLC stack from which `C_arrays` was computed contained
        compressed SLCs at the stack, then this should be the index
        of the first non-compressed SLC.
    n_workers : int, optional
        The number of workers to use (CPU version) for the eigenvector problem.
        If 1 (default), no multiprocessing is used.

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
    V = _get_eigvecs(Gamma_inv * C_arrays, n_workers=n_workers)

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
    return xp.moveaxis(phase_stack, -1, 0)


def _get_eigvecs(C, n_workers: int = 1):
    xp = get_array_module(C)
    if xp == np:
        # The block splitting isn't needed for numpy.
        # return np.linalg.eigh(C)[1]
        return _get_eigvecs_scipy(C, n_workers=n_workers)

    # Make sure we don't overflow: cupy https://github.com/cupy/cupy/issues/7261
    # The work_size must be less than 2**30, so
    # Keep (rows*cols) approximately less than 2**21? make it 2**20 to be safer
    # see https://github.com/cupy/cupy/issues/7261#issuecomment-1362991323 for that math
    rows, cols, _, _ = C.shape
    max_batch_size = 2**20
    num_blocks = 1 + (rows * cols) // max_batch_size
    if num_blocks > 1:
        # V_out wil lbe the eigenvectors, shape (rows, cols, nslc, nslc)
        V_out = xp.empty_like(C)
        # Split the computation into blocks
        # This is to avoid overflow errors in cupy.linalg.eigh
        for i in range(num_blocks):
            # get chunks of rows at a time
            start = i * (rows // num_blocks)
            end = (i + 1) * (rows // num_blocks)
            if i == num_blocks - 1:
                end = rows
            V_out[start:end] = xp.linalg.eigh(C[start:end])[1]
    else:
        _, V_out = xp.linalg.eigh(C)
    return V_out


def _get_eigvecs_scipy(C, n_workers=1):
    C_shared = pymp.shared.array(C.shape, dtype="complex64")
    C_shared[:] = C[:]
    rows, cols, nslc, _ = C.shape
    out = pymp.shared.array((rows, cols, nslc), dtype="complex64")
    with pymp.Parallel(n_workers) as p:
        # Looping over linear index for pixels (less nesting of pymp context managers)
        for idx in p.range(rows * cols):
            # Iterating over every output pixels, convert to a row/col index
            r, c = np.unravel_index(idx, (rows, cols))
            out[r, c, :] = eigh(C_shared[r, c], subset_by_index=[0, 0])[1].ravel()

    del C_shared
    # Add the last dimension back to match the shape of the cupy output
    return out[:, :, :, None]


def _check_all_nans(slc_stack):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        raise PhaseLinkRuntimeError(
            f"slc_stack[{bad_slc_idxs}] out of {len(slc_stack)} are all NaNs."
        )


def _fill_ps_pixels(
    mle_est, temp_coh, slc_stack, ps_mask, strides, avg_mag, use_max_ps: bool = False
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
    strides : dict
        The look window strides
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
    use_max_ps : bool, optional
        If True, use the brightest PS pixel in each look window to fill in the
        MLE estimate. If False, use the average of all PS pixels in each look window.
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
    ps_mask_looked = take_looks(
        ps_mask, strides["y"], strides["x"], func_type="any", edge_strategy="pad"
    )
    # make sure it's the same size as the MLE result/temp_coh after padding
    ps_mask_looked = ps_mask_looked[: mle_est.shape[1], : mle_est.shape[2]]

    if use_max_ps:
        print("Using max PS pixel to fill in MLE estimate")
        # Get the indices of the brightest pixels within each look window
        slc_r_idxs, slc_c_idxs = _get_max_idxs(mag, strides["y"], strides["x"])
        # we're only filling where there are PS pixels
        ref = np.conj(slc_stack[0][slc_r_idxs, slc_c_idxs])
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
    slc_stack: np.ndarray, ps_mask: np.ndarray, strides: dict
) -> np.ndarray:
    # First, set all non-PS pixels to NaN
    slc_stack_nanned = slc_stack.copy()
    slc_stack_nanned[:, ~ps_mask] = np.nan
    # Reference all ps pixels in the SLC stack to the first SLC
    slc_stack_nanned[:, ps_mask] *= np.conj(slc_stack_nanned[0, ps_mask])[None]
    # Then, take the average of all PS pixels within each look window
    return take_looks(
        slc_stack_nanned,
        strides["y"],
        strides["x"],
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
