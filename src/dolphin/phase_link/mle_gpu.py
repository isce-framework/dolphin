from cmath import isnan
from cmath import sqrt as csqrt
from math import ceil
from typing import Tuple

import cupy as cp
import h5py
import numpy as np
from numba import cuda

from dolphin.utils import half_window_to_full

from .mle import full_cov_multilooked, mle_stack

# TODO: make a version which has the same API as the CPU


def run_mle_gpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    beta: float = 0.0,
    reference_idx: int = 0,
    mask: np.ndarray = None,
    ps_mask: np.ndarray = None,
    output_cov_file: str = None,
) -> np.ndarray:
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
    np.ndarray
        The estimated linked phase
    """
    num_slc, rows, cols = slc_stack.shape

    if mask is None:
        mask = np.zeros((rows, cols), dtype=bool)
    else:
        mask = mask.astype(bool)
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    ignore_mask = np.logical_or(mask, ps_mask)
    # Make a copy, and set the masked pixels to np.nan
    slc_stack_copy = slc_stack.copy()
    slc_stack_copy[:, ignore_mask] = np.nan

    # Copy the read-only data to the device
    d_slc_stack = cuda.to_device(slc_stack_copy)

    # Make a buffer for each pixel's coherence matrix
    # d_ means "device_", i.e. on the GPU
    d_C_buf = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)

    # Divide up the stack using a 2D grid
    threads_per_block = (16, 16)
    blocks_x = ceil(rows / threads_per_block[0])
    blocks_y = ceil(cols / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    estimate_c_gpu[blocks, threads_per_block](d_slc_stack, half_window, d_C_buf)

    if output_cov_file:
        # Copy back to the host
        C_buf = d_C_buf.get()
        print(f"Saving covariance matrix at each pixel to {output_cov_file}")
        with h5py.File(output_cov_file, "w") as f:
            f.create_dataset("C", data=C_buf)

    d_output_phase = mle_stack(d_C_buf, beta=beta, reference_idx=reference_idx)
    # copy back to host
    output_phase = d_output_phase.get()
    # and set the PS pixel phases if using a PS mask
    if np.any(ps_mask):
        ps_ref = slc_stack[0][ps_mask]
        for i in range(num_slc):
            # print(np.all(ref_ps_phase - np.angle(slc_stack[i])[ps_mask] < 1e-6))
            output_phase[i][ps_mask] = np.angle(slc_stack[i][ps_mask] * np.conj(ps_ref))
            # output_phase[i][ps_mask] = np.angle(slc_stack[i])[ps_mask]
            # output_phase[i] = np.where(
            #     ps_mask, np.angle(slc_stack[i]), output_phase[i]
            # )

    # Finally, use the amplitude from the original SLCs
    return np.abs(slc_stack) * np.exp(1j * output_phase)


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

            c = numer / csqrt(a1 * a2)
            cov_mat[i_slc, j_slc] = c
            cov_mat[j_slc, i_slc] = c.conjugate()
        cov_mat[i_slc, i_slc] = 1.0


def run_mle_multilooked_gpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    beta: float = 0.0,
    output_cov_file: str = None,
    reference_idx: int = 0,
):
    """Estimate a down-sampled version of the linked phase using the MLE estimator."""
    d_slc_stack = cp.asarray(slc_stack)
    window = half_window_to_full(half_window)
    # window is (xsize, ysize), want as (row looks, col looks)
    looks = (window[1], window[0])

    # Get the covariance at each pixel on the GPU
    d_C_buf = full_cov_multilooked(d_slc_stack, looks)

    if output_cov_file:
        # Copy back to the host
        C_buf = d_C_buf.get()
        print(f"Saving covariance matrix at each pixel to {output_cov_file}")
        with h5py.File(output_cov_file, "w") as f:
            f.create_dataset("C", data=C_buf)

    # Estimate the phase
    phase = mle_stack(d_C_buf, beta=beta, reference_idx=reference_idx)
    return np.exp(1j * phase.get())


# def run(
#     slc_stack_file: np.ndarray,
#     half_window: Tuple[int, int],
#     beta: float = 0.0,
#     mask: np.ndarray = None,
#     output_cov_file: str = None,
# ):
#     pass
