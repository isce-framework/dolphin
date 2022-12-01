"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""
from cmath import isnan
from cmath import sqrt as csqrt
from typing import Tuple

import numpy as np
import pymp
from numba import cuda, njit

# CPU version of the covariance matrix computation


def estimate_stack_covariance_cpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    n_workers=1,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the CPU.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols).
    half_window : Tuple[int, int]
        The half window size as [half_x, half_y] in pixels.
        The full window size is 2 * half_window + 1 for x, y.
    strides : Tuple[int, int], optional
        The (row, col) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.

    Returns
    -------
    C_arrays : np.ndarray
        The covariance matrix at each pixel, with shape
        (n_rows, n_cols, n_slc, n_slc).
    """
    # Get the dimensions
    nslc, rows, cols = slc_stack.shape
    dtype = slc_stack.dtype
    slcs_shared = pymp.shared.array(slc_stack.shape, dtype=dtype)
    slcs_shared[:] = slc_stack[:]

    out_rows, out_cols = compute_out_shape((rows, cols), strides)
    C_arrays = pymp.shared.array((out_rows, out_cols, nslc, nslc), dtype=dtype)

    row_strides, col_strides = strides
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
                half_window, in_r, in_c, rows, cols
            )
            # Read the 3D current chunk
            samples_stack = slcs_shared[:, r_start:r_end, c_start:c_end]
            # Compute one covariance matrix for the output pixel
            coh_mat_single(
                samples_stack.reshape(nslc, -1), C_arrays[out_r, out_c, :, :]
            )

    del slcs_shared
    return C_arrays


@njit(cache=True)
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
    slc_stack, half_window: Tuple[int, int], strides: Tuple[int, int], C_out
):
    """Estimate the linked phase at all pixels of `slc_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    out_x, out_y = cuda.grid(2)
    out_rows, out_cols = C_out.shape[:2]
    # Check if we are within the bounds of the array
    if out_y >= out_rows or out_x >= out_cols:
        return

    row_strides, col_strides = strides
    r_start = row_strides // 2
    c_start = col_strides // 2
    in_r = r_start + out_y * row_strides
    in_c = c_start + out_x * col_strides

    N, rows, cols = slc_stack.shape
    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices_gpu(
        half_window, in_r, in_c, rows, cols
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


# (r_start, r_end), (c_start, c_end) = _get_slices_cpu(
#                 half_window, in_r, in_c, rows, cols
#             )
def _get_slices(half_window, r, c, rows, cols):
    """Get the slices for the given pixel and half window size."""
    # Get the half window size
    half_x, half_y = half_window

    # Clamp min indexes to 0
    r_start = max(r - half_y, 0)
    c_start = max(c - half_x, 0)
    # Clamp max indexes to the array size
    r_end = min(r + half_y + 1, rows)
    c_end = min(c + half_x + 1, cols)
    return (r_start, r_end), (c_start, c_end)


# Make cpu and gpu compiled versions of the helper function
_get_slices_cpu = njit(_get_slices)
_get_slices_gpu = cuda.jit(device=True)(_get_slices)


def compute_out_shape(
    shape: Tuple[int, int], strides: Tuple[int, int]
) -> Tuple[int, int]:
    """Calculate the output size for an input `shape` and row/col `strides`.

    For instance, in a 6 x 6 array with `strides=(3, 3)`,
    we could expect the pixels to be centered on indexes
    `[7, 10, 25, 28]`.

        [[ 0  1  2   3  4  5]
        [ 6  7  8    9 10 11]
        [12 13 14   15 16 17]

        [18 19 20   21 22 23]
        [24 25 26   27 28 29]
        [30 31 32   33 34 35]]


    So the output size would be `(2, 2)`.

    Parameters
    ----------
    shape : Tuple[int, int]
        Input size: (rows, cols)
    strides : Tuple[int, int]
        (row strides, col_strides)

    Returns
    -------
    out_shape : Tuple[int, int]
        Size of output after striding
    """
    rows, cols = shape
    rs, cs = strides
    # initial starting pixel
    r_off, c_off = (rs // 2, cs // 2)
    remaining_rows = rows - r_off - 1
    remaining_cols = cols - c_off - 1
    out_rows = remaining_rows // rs + 1
    out_cols = remaining_cols // cs + 1

    return out_rows, out_cols


def _save_covariance(output_cov_file, C_arrays):
    import h5py

    # TODO: accept slices for where to save in existing file
    # TODO: convert to UInt8 to compress ussing fill/compress
    # encoding_abs = dict(
    #     scale_factor=1/255,
    #     _FillValue=0,
    #     dtype="uint8",
    #     compression="gzip",
    #     shuffle=True
    # )
    coh_mag = np.abs(C_arrays)
    coh_mag_uint = (np.nan_to_num(coh_mag) * 255).astype(np.uint8)

    print(f"Saving covariance matrix at each pixel to {output_cov_file}")
    with h5py.File(output_cov_file, "w") as f:
        f.create_dataset(
            "correlation",
            data=coh_mag_uint,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )
