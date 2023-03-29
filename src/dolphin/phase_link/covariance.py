"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""
from cmath import isnan
from cmath import sqrt as csqrt
from typing import Dict, Tuple

import numpy as np
import pymp
from numba import cuda, njit

from dolphin.io import compute_out_shape

# CPU version of the covariance matrix computation


def estimate_stack_covariance_cpu(
    slc_stack: np.ndarray,
    half_window: Dict[str, int],
    strides: Dict[str, int] = {"x": 1, "y": 1},
    n_workers=1,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the CPU.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols).
    half_window : Dict[str, int]
        The half window size as {"x": half_win_x, "y": half_win_y}
        The full window size is 2 * half_window + 1 for x, y.
    strides : Dict[str, int], optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.

    Returns
    -------
    C_arrays : np.ndarray
        The covariance matrix at each pixel, with shape
        (n_rows, n_cols, n_slc, n_slc).

    Raises
    ------
    ValueError
        If `slc_stack` is not complex data.
    """
    if not np.iscomplexobj(slc_stack):
        raise ValueError("The SLC stack must be complex.")
    # Get the dimensions
    nslc, rows, cols = slc_stack.shape
    dtype = slc_stack.dtype
    slcs_shared = pymp.shared.array(slc_stack.shape, dtype=dtype)
    slcs_shared[:] = slc_stack[:]

    out_rows, out_cols = compute_out_shape((rows, cols), strides)
    C_arrays = pymp.shared.array((out_rows, out_cols, nslc, nslc), dtype=dtype)

    row_strides, col_strides = strides["y"], strides["x"]
    half_col, half_row = half_window["x"], half_window["y"]
    with pymp.Parallel(n_workers) as p:
        # Looping over linear index for pixels (less nesting of pymp context managers)
        for idx in p.range(out_rows * out_cols):
            # Iterating over every output pixels, convert to a row/col index
            out_r, out_c = np.unravel_index(idx, (out_rows, out_cols))

            # the input indexes computed from the output idx and strides
            # Note: weirdly, moving these out of the loop causes r_start
            # to be 0 in some cases...
            r_start = row_strides // 2
            c_start = col_strides // 2
            in_r = r_start + out_r * row_strides
            in_c = c_start + out_c * col_strides

            (r_start, r_end), (c_start, c_end) = _get_slices_cpu(
                half_row, half_col, in_r, in_c, rows, cols
            )
            # Read the 3D current chunk
            samples_stack = slcs_shared[:, r_start:r_end, c_start:c_end]
            # Compute one covariance matrix for the output pixel
            coh_mat_single(
                samples_stack.reshape(nslc, -1), C_arrays[out_r, out_c, :, :]
            )

    del slcs_shared
    return C_arrays


@njit
def coh_mat_single(neighbor_stack, cov_mat=None):
    """Given a (n_slc, n_samps) samples, estimate the coherence matrix."""
    nslc = neighbor_stack.shape[0]
    if cov_mat is None:
        cov_mat = np.zeros((nslc, nslc), dtype=neighbor_stack.dtype)

    for ti in range(nslc):
        # Start with the diagonal equal to 1
        cov_mat[ti, ti] = 1.0
        for tj in range(ti + 1, nslc):
            c1, c2 = neighbor_stack[ti, :], neighbor_stack[tj, :]
            a1 = np.nansum(np.abs(c1) ** 2)
            a2 = np.nansum(np.abs(c2) ** 2)

            # check if either had 0 good pixels
            a_prod = a1 * a2
            if abs(a_prod) < 1e-6:
                cov = 0.0 + 0.0j
            else:
                cov = np.nansum(c1 * np.conjugate(c2)) / np.sqrt(a_prod)

            cov_mat[ti, tj] = cov
            cov_mat[tj, ti] = np.conjugate(cov)

    return cov_mat


# GPU version of the covariance matrix computation
@cuda.jit
def estimate_stack_covariance_gpu(
    slc_stack, half_rowcol: Tuple[int, int], strides_rowcol: Tuple[int, int], C_out
):
    """Estimate the linked phase at all pixels of `slc_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    out_x, out_y = cuda.grid(2)
    out_rows, out_cols = C_out.shape[:2]
    # Check if we are within the bounds of the array
    if out_y >= out_rows or out_x >= out_cols:
        return

    row_strides, col_strides = strides_rowcol
    r_start = row_strides // 2
    c_start = col_strides // 2
    in_r = r_start + out_y * row_strides
    in_c = c_start + out_x * col_strides

    half_row, half_col = half_rowcol

    N, rows, cols = slc_stack.shape
    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices_gpu(
        half_row, half_col, in_r, in_c, rows, cols
    )
    samples_stack = slc_stack[:, r_start:r_end, c_start:c_end]

    # estimate the coherence matrix, store in current pixel's buffer
    C = C_out[out_y, out_x, :, :]
    _coh_mat_gpu(samples_stack, C)


@cuda.jit(device=True)
def _coh_mat_gpu(samples_stack, cov_mat):
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

            a_prod = a1 * a2
            # If one window is all NaNs, skip
            if isnan(a_prod) or abs(a_prod) < 1e-6:
                # TODO: advantage of using nan here?
                # Seems like using 0 will ignore it in the estimation
                # cov_mat[i_slc, j_slc] = cov_mat[j_slc, i_slc] = np.nan
                cov_mat[i_slc, j_slc] = cov_mat[j_slc, i_slc] = 0
            else:
                c = numer / csqrt(a_prod)
                cov_mat[i_slc, j_slc] = c
                cov_mat[j_slc, i_slc] = c.conjugate()
        cov_mat[i_slc, i_slc] = 1.0


def _get_slices(half_r: int, half_c: int, r: int, c: int, rows: int, cols: int):
    """Get the slices for the given pixel and half window size."""
    # Clamp min indexes to 0
    r_start = max(r - half_r, 0)
    c_start = max(c - half_c, 0)
    # Clamp max indexes to the array size
    r_end = min(r + half_r + 1, rows)
    c_end = min(c + half_c + 1, cols)
    return (r_start, r_end), (c_start, c_end)


# Make cpu and gpu compiled versions of the helper function
_get_slices_cpu = njit(_get_slices)
_get_slices_gpu = cuda.jit(device=True)(_get_slices)
