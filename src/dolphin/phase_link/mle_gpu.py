from cmath import isnan
from cmath import sqrt as csqrt
from math import ceil
from typing import Optional, Tuple

import cupy as cp
import numpy as np
from numba import cuda

from dolphin.utils import Pathlike, half_window_to_full

from .mle import estimate_temp_coh, full_cov, mle_stack

# TODO: make a version which has the same API as the CPU


class MLERuntimeError(Exception):
    """Exception raised for running the MLE GPU code."""

    pass


def run_mle_gpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    beta: float = 0.0,
    reference_idx: int = 0,
    mask: np.ndarray = None,
    ps_mask: np.ndarray = None,
    output_cov_file: Optional[Pathlike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the linked phase for a stack using the MLE estimator.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : Tuple[int, int]
        The half window size as [half_x, half_y] in pixels.
        The full window size is 2 * half_window + 1 for x, y.
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
    output_cov_file : str, optional
        HDF5 filename to save the estimated covariance at each pixel.

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_images, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    num_slc, rows, cols = slc_stack.shape

    if mask is None:
        mask = np.zeros((rows, cols), dtype=bool)
    else:
        mask = mask.astype(bool)
    # Make sure we also are ignoring pixels which are nans for all SLCs
    mask |= np.all(np.isnan(slc_stack), axis=0)

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

    # Copy the read-only data to the device
    d_slc_stack = cuda.to_device(slc_stack_copy)

    # Make a buffer for each pixel's coherence matrix
    # d_ means "device_", i.e. on the GPU
    d_C_arrays = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)

    # Divide up the stack using a 2D grid
    threads_per_block = (16, 16)
    blocks_x = ceil(rows / threads_per_block[0])
    blocks_y = ceil(cols / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    estimate_c_gpu[blocks, threads_per_block](d_slc_stack, half_window, d_C_arrays)

    if output_cov_file:
        # Copy back to the host
        # _save_covariance(d_C_arrays, output_cov_file)
        pass

    d_output_phase = mle_stack(d_C_arrays, beta=beta, reference_idx=reference_idx)
    d_cpx_phase = cp.exp(1j * d_output_phase)

    # Get the temporal coherence
    temp_coh = estimate_temp_coh(d_cpx_phase, d_C_arrays).get()
    # Set no data pixels to np.nan
    temp_coh[mask] = np.nan

    # copy back to host to set the PS pixels (if a PS mask is passed)
    output_phase = d_output_phase.get()
    if np.any(ps_mask):
        ps_ref = slc_stack[0][ps_mask]
        for i in range(num_slc):
            output_phase[i][ps_mask] = np.angle(slc_stack[i][ps_mask] * np.conj(ps_ref))
        # Force PS pixels to have high temporal coherence
        temp_coh[ps_mask] = 1

    # # https://docs.cupy.dev/en/stable/user_guide/memory.html
    # may just be cached a lot of the huge memory available on aurora
    # But if we need to free GPU memory:
    # cp.get_default_memory_pool().free_all_blocks()

    # use the amplitude from the original SLCs
    mle_est = np.abs(slc_stack) * np.exp(1j * output_phase)
    return mle_est, temp_coh


@cuda.jit
def estimate_c_gpu(slc_stack, half_window, C_buf):
    """GPU kernel for estimating the linked phase at one pixel."""
    # Get the global position within the 2D GPU grid
    i, j = cuda.grid(2)
    N, rows, cols = slc_stack.shape
    # Check if we are within the bounds of the array
    if i >= rows or j >= cols:
        return
    # Get the half window size
    half_x, half_y = half_window
    # if i - half_x < 0 or i + half_x >= rows

    # Get the window
    # Clamp min indexes to 0
    rstart = max(i - half_y, 0)
    cstart = max(j - half_x, 0)
    # Clamp max indexes to the array size
    rend = min(i + half_y + 1, rows)
    cent = min(j + half_x + 1, cols)

    # Clamp the window to the image bounds
    samples_stack = slc_stack[:, rstart:rend, cstart:cent]

    # estimate the coherence matrix, store in current pixel's buffer
    C = C_buf[i, j, :, :]
    coh_mat(samples_stack, C)


@cuda.jit(device=True)
def coh_mat(samples_stack, cov_mat):
    """Given a (n_slc, n_samps) samples, estimate the coherence matrix."""
    nslc = cov_mat.shape[0]
    # Iterate over the upper triangle of the output matrix
    for i_slc in range(nslc):
        for j_slc in range(i_slc + 1, nslc):
            numer = 0.0
            a1 = 0.0
            a2 = 0.0
            # At each C_ij, iterate over the samples in the window
            for rs in range(samples_stack.shape[1]):
                for cs in range(samples_stack.shape[2]):
                    s1 = samples_stack[i_slc, rs, cs]
                    s2 = samples_stack[j_slc, rs, cs]
                    if isnan(s1) or isnan(s2):
                        continue
                    numer += s1 * s2.conjugate()
                    a1 += s1 * s1.conjugate()
                    a2 += s2 * s2.conjugate()

            aprod = a1 * a2
            # If one window is all NaNs, skip
            if isnan(aprod) or abs(aprod) < 1e-6:
                # TODO: advantage of using nan here?
                # Seems like using 0 will ignore it in the estimation
                # cov_mat[i_slc, j_slc] = cov_mat[j_slc, i_slc] = np.nan
                cov_mat[i_slc, j_slc] = cov_mat[j_slc, i_slc] = 0
            else:
                c = numer / csqrt(aprod)
                cov_mat[i_slc, j_slc] = c
                cov_mat[j_slc, i_slc] = c.conjugate()
        cov_mat[i_slc, i_slc] = 1.0


def run_mle_multilooked_gpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    beta: float = 0.0,
    output_cov_file: Optional[Pathlike] = None,
    reference_idx: int = 0,
):
    """Estimate a down-sampled version of the linked phase using the MLE estimator."""
    d_slc_stack = cp.asarray(slc_stack)
    window = half_window_to_full(half_window)
    # window is (xsize, ysize), want as (row looks, col looks)
    looks = (window[1], window[0])

    # Get the covariance at each pixel on the GPU
    d_C_arrays = full_cov(d_slc_stack, looks)

    if output_cov_file:
        # TODO: save the covariance matrix as uint8 for space
        # Copy back to the host
        # _save_covariance(d_C_arrays, output_cov_file)
        pass

    # Estimate the phase
    phase = mle_stack(d_C_arrays, beta=beta, reference_idx=reference_idx)
    d_cpx_phase = cp.exp(1j * phase)
    # Get the temporal coherence
    d_temp_coh = estimate_temp_coh(d_cpx_phase, d_C_arrays)
    return d_cpx_phase.get(), d_temp_coh.get()


def _check_all_nans(slc_stack):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        raise MLERuntimeError(f"SLC stack[{bad_slc_idxs}] has are all NaNs.")


# def run(
#     slc_stack_file: np.ndarray,
#     half_window: Tuple[int, int],
#     beta: float = 0.0,
#     mask: np.ndarray = None,
#     output_cov_file: str = None,
# ):
#     pass

# def _save_covariance(d_C_arrays, output_cov_file):
#     # TODO: convert to UInt8 to compress
#     C_buf = d_C_arrays.get()
#     print(f"Saving covariance matrix at each pixel to {output_cov_file}")
#     with h5py.File(output_cov_file, "w") as f:
#         f.create_dataset("C", data=C_buf)
