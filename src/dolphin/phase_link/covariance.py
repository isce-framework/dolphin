"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""
from __future__ import annotations

from cmath import isnan
from cmath import sqrt as csqrt
from typing import Optional, Union

import numba
import numpy as np
import pymp
from numba import cuda, njit

from dolphin._types import Filename
from dolphin.io import compute_out_shape
from dolphin.utils import _get_slices

# CPU version of the covariance matrix computation


def estimate_stack_covariance_cpu(
    slc_stack: np.ndarray,
    half_window: dict[str, int],
    strides: dict[str, int] = {"x": 1, "y": 1},
    neighbor_arrays: Optional[np.ndarray] = None,
    n_workers=1,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the CPU.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols).
    half_window : dict[str, int]
        The half window size as {"x": half_win_x, "y": half_win_y}
        The full window size is 2 * half_window + 1 for x, y.
    strides : dict[str, int], optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
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

    cur_neighbors = np.empty((0, 0, 0, 0), dtype=bool)
    if neighbor_arrays is not None and neighbor_arrays.size > 0:
        do_shp = True
        neighbor_arrays_shared = pymp.shared.array(neighbor_arrays.shape, dtype=bool)
        neighbor_arrays_shared[:] = neighbor_arrays[:]
    else:
        do_shp = False

    with pymp.Parallel(n_workers) as p:
        # Looping over linear index for pixels (less nesting of pymp context managers)
        for idx in p.range(out_rows * out_cols):
            # Iterating over every output pixels, convert to a row/col index
            out_r, out_c = np.unravel_index(idx, (out_rows, out_cols))

            # the input indexes computed from the output idx and strides
            # Note: weirdly, moving these out of the loop causes r_start
            # to be 0 in some cases...
            in_r_start = row_strides // 2
            in_c_start = col_strides // 2
            in_r = in_r_start + out_r * row_strides
            in_c = in_c_start + out_c * col_strides

            # Check if the window is completely in bounds
            if in_r + half_row >= rows or in_r - half_row < 0:
                continue
            if in_c + half_col >= cols or in_c - half_col < 0:
                continue

            (r0, r1), (c0, c1) = _get_slices(half_row, half_col, in_r, in_c, rows, cols)
            # Read the 3D current chunk
            samples_stack = slcs_shared[:, r0:r1, c0:c1]
            # Read the current neighbor mask
            if do_shp:
                # TODO: this will be different shape than samples_stack at edges
                # does this matter? prob just clipping by the overlapping half window
                cur_neighbors = neighbor_arrays_shared[out_r, out_c, :, :]
            # Compute one covariance matrix for the output pixel
            coh_mat_single(
                samples_stack.reshape(nslc, -1),
                C_arrays[out_r, out_c, :, :],
                cur_neighbors.ravel(),
                do_shp,
            )

    del slcs_shared
    return C_arrays


@njit(nogil=True)
def coh_mat_single(slc_samples, cov_mat=None, neighbor_mask=None, do_shp: bool = True):
    """Given a (n_slc, n_samps) samples, estimate the coherence matrix."""
    nslc, nsamps = slc_samples.shape
    if cov_mat is None:
        cov_mat = np.zeros((nslc, nslc), dtype=slc_samples.dtype)
    if neighbor_mask is None:
        do_shp = False
        neighbor_mask = np.zeros((0,), dtype=np.bool_)
    if neighbor_mask.size <= 1:
        do_shp = False

    for ti in range(nslc):
        # Start with the diagonal equal to 1
        cov_mat[ti, ti] = 1.0
        for tj in range(ti + 1, nslc):
            c1, c2 = slc_samples[ti, :], slc_samples[tj, :]
            # a1 = np.nansum(np.abs(c1) ** 2)
            # a2 = np.nansum(np.abs(c2) ** 2)
            # Manually sum to skip based on the neighbor mask
            numer = a1 = a2 = 0.0
            for sidx in range(nsamps):
                if do_shp and not neighbor_mask[sidx]:
                    continue
                s1 = c1[sidx]
                s2 = c2[sidx]
                if np.isnan(s1) or np.isnan(s2):
                    continue
                numer += s1 * s2.conjugate()
                a1 += s1 * s1.conjugate()
                a2 += s2 * s2.conjugate()

            # check if either had 0 good pixels
            a_prod = a1 * a2
            if abs(a_prod) < 1e-6:
                cov = 0.0 + 0.0j
            else:
                # cov = np.nansum(c1 * np.conjugate(c2)) / np.sqrt(a_prod)
                cov = numer / csqrt(a_prod)

            cov_mat[ti, tj] = cov
            cov_mat[tj, ti] = np.conjugate(cov)

    return cov_mat


# GPU version of the covariance matrix computation
# Note that we can't do keyword args with numba.cuda.jit
@cuda.jit
def estimate_stack_covariance_gpu(
    slc_stack,
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    neighbor_arrays,
    C_out,
    do_shp,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    out_x, out_y = cuda.grid(2)
    out_rows, out_cols = C_out.shape[:2]

    # Convert the output locations to higher-res input locations
    row_strides, col_strides = strides_rowcol
    r1 = row_strides // 2
    c1 = col_strides // 2
    in_r = r1 + out_y * row_strides
    in_c = c1 + out_x * col_strides
    N, rows, cols = slc_stack.shape

    half_row, half_col = halfwin_rowcol
    # # Check if we are within the bounds of the array
    # if out_y >= out_rows or out_x >= out_cols:
    #     return
    # Check if the window is completely in bounds
    if in_r + half_row >= rows or in_r - half_row < 0:
        return
    if in_c + half_col >= cols or in_c - half_col < 0:
        return

    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices(
        half_row, half_col, in_r, in_c, rows, cols
    )
    samples_stack = slc_stack[:, r_start:r_end, c_start:c_end]
    if do_shp:
        neighbors_stack = neighbor_arrays[out_y, out_x, :, :]
    else:
        neighbors_stack = cuda.local.array((2, 2), dtype=numba.bool_)

    # estimate the coherence matrix, store in current pixel's buffer
    C = C_out[out_y, out_x, :, :]
    _coh_mat_gpu(samples_stack, neighbors_stack, do_shp, C)


@cuda.jit(device=True)
def _coh_mat_gpu(samples_stack, neighbors_stack, do_shp, cov_mat):
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
                    # Skip if it's not a valid neighbor
                    if do_shp and not neighbors_stack[rs, cs]:
                        continue

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


def _save_coherence_matrices(
    filename: Filename,
    C: np.ndarray,
    chunks: Union[None, tuple[int, ...], str, bool] = None,
    **compression_opts,
):
    import h5py

    if chunks is None:
        nslc = C.shape[-1]
        chunks = (10, 10, nslc, nslc)

    if not compression_opts:
        compression_opts = dict(
            compression="lzf",
            shuffle=True,
        )
    compression_opts["chunks"] = chunks

    with h5py.File(filename, "w") as f:
        # Create datasets for dimensions
        y_dim = f.create_dataset("y", data=np.arange(C.shape[0]))
        x_dim = f.create_dataset("x", data=np.arange(C.shape[1]))
        slc1_dim = f.create_dataset("slc1", data=np.arange(C.shape[2]))
        slc2_dim = f.create_dataset("slc2", data=np.arange(C.shape[3]))

        # Create the main dataset and set dimensions as attributes
        # f.require_dataset(name="data", shape=shape)
        data_dset = f.create_dataset(
            "data",
            # Save only the upper triangle
            # Quantize as a uint8 so that coherence = DN / 255
            data=(255 * np.triu(C)).astype("uint8"),
            **compression_opts,
        )
        # Set dimension scales for each dimension in the main dataset
        dims = [y_dim, x_dim, slc1_dim, slc2_dim]
        labels = ["y", "x", "slc1", "slc2"]
        for i, (dim, label) in enumerate(zip(dims, labels)):
            data_dset.dims[i].attach_scale(dim)
            data_dset.dims[i].label = label
