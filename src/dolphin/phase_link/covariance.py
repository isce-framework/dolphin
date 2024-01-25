"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""

from __future__ import annotations

from cmath import sqrt as csqrt
from functools import partial
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
import pymp
from jax import jit, lax, vmap
from numba import njit

from dolphin._types import Filename
from dolphin.utils import _get_slices, compute_out_shape

# CPU version of the covariance matrix computation
_get_slices = njit(_get_slices)


def estimate_stack_covariance_cpu(
    slc_stack: np.ndarray,
    half_window: dict[str, int],
    strides: Optional[dict[str, int]] = None,
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
    if strides is None:
        strides = {"x": 1, "y": 1}
    if not np.iscomplexobj(slc_stack):
        msg = "The SLC stack must be complex."
        raise ValueError(msg)
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


@partial(jit, static_argnames=["half_window_rowcol", "strides_rowcol"])
def estimate_stack_covariance_jax(
    slc_stack: np.ndarray,
    # half_window: dict[str, int],
    # strides: Optional[dict[str, int]] = None,
    half_window_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int] = (1, 1),
    neighbor_arrays: Optional[np.ndarray] = None,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the CPU.

    The output sizes can be computed using `compute_out_shape`:
        out_rows, out_cols = compute_out_shape((rows, cols), strides)

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
        The neighbor arrays to use for SHP,
        shape = (n_out_rows, n_out_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.

    Returns
    -------
    C_arrays : np.ndarray
        The covariance matrix at each pixel, with shape
        (n_out_rows, n_out_cols, n_slc, n_slc).


    Raises
    ------
    ValueError
        If `slc_stack` is not complex data.
    """
    if not np.iscomplexobj(slc_stack):
        msg = "The SLC stack must be complex."
        raise ValueError(msg)
    # Get the dimensions
    nslc, rows, cols = slc_stack.shape

    row_strides, col_strides = strides_rowcol
    half_col, half_row = half_window_rowcol

    out_rows, out_cols = compute_out_shape(
        (rows, cols), {"x": col_strides, "y": row_strides}
    )

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2

    # padded_slc_stack = _pad_stack_border(slc_stack, half_row, half_col)
    padded_slc_stack = slc_stack
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    def _process_row_col(out_r, out_c):
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides
        slc_window = _get_stack_window(padded_slc_stack, in_r, in_c, half_row, half_col)
        slc_samples = slc_window.reshape(nslc, -1)
        cur_neighbors = neighbor_arrays[out_r, out_c, :, :]
        neighbor_mask = cur_neighbors.ravel()

        # cur_neighbors = _get_neighbor_window(
        #     neighbor_arrays, out_r, out_c, half_row, half_col
        # )
        return coh_mat_vectorized(slc_samples, neighbor_mask=neighbor_mask)

        # # Read the current neighbor mask
        # if do_shp:
        #     # TODO: this will be different shape than samples_stack at edges
        #     # does this matter? prob just clipping by the overlapping half window
        #     cur_neighbors = neighbor_arrays_shared[out_r, out_c, :, :]
        # # Compute one covariance matrix for the output pixel
        # coh_mat_single(
        #     samples_stack.reshape(nslc, -1),
        #     C_arrays[out_r, out_c, :, :],
        #     cur_neighbors.ravel(),
        #     do_shp,
        # )

    _process_2d = vmap(_process_row_col)
    _process_3d = vmap(_process_2d)
    return _process_3d(out_r_indices, out_c_indices)


def _pad_stack_border(stack, half_row, half_col):
    """Pad the stack with zeros to avoid edge effects."""
    # (No padding along depth/date dimension, (row padding), (col padding))
    return jnp.pad(stack, ((0, 0), (half_row, half_row), (half_col, half_col)))


def _get_stack_window(padded_stack, r, c, half_row, half_col):
    """Dynamically slice the stack at (r, c) with size (2*half_row+1, 2*half_col+1)."""
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#non-array-inputs-numpy-vs-jax
    # Note: out of bounds indexing for JAX clamps to the nearest value
    # This is fine as we want to trim the borders anyway, so we don't need the
    # extra checks in utils._get_slices

    # Center the slice on (r, c), so we need the starts to move up/left
    start_indices = (0, r - half_row, c - half_col)
    # start_indices = (0, r, c)
    slice_sizes = (padded_stack.shape[0], 2 * half_row + 1, 2 * half_col + 1)
    return lax.dynamic_slice(padded_stack, start_indices, slice_sizes)


# def _get_neighbor_window(neighbor_arrays, out_r, out_c, half_row, half_col):
#     slice_size = (2 * half_row + 1, 2 * half_col + 1)
#     start_indices = (out_r, out_c)
#     return lax.dynamic_slice(neighbor_arrays, start_indices, slice_size)


def single_pixel_covariance(padded_stack, r, c, half_row, half_col, n_images):
    neighborhood = _get_stack_window(padded_stack, r, c, half_row, half_col).reshape(
        n_images, -1
    )
    return coh_mat_vectorized(neighborhood)


def compute_covariances(stack, half_row, half_col):
    n_images, nrows, ncols = stack.shape
    padded_stack = _pad_stack_border(stack, half_row, half_col)

    def cov_fn(r, c):
        return single_pixel_covariance(padded_stack, r, c, half_row, half_col, n_images)

    r_indices, c_indices = jnp.meshgrid(
        jnp.arange(nrows), jnp.arange(ncols), indexing="ij"
    )

    cov_matrices = vmap(vmap(cov_fn))(r_indices, c_indices)

    return cov_matrices


def coh_mat_vectorized(slc_samples, neighbor_mask=None):
    nslc, nsamps = slc_samples.shape

    if neighbor_mask is None:
        neighbor_mask = jnp.ones(nsamps, dtype=jnp.bool_)
    valid_samples_mask = ~jnp.isnan(slc_samples)
    combined_mask = valid_samples_mask & neighbor_mask[None, :]

    # Mask the slc samples
    masked_slc = jnp.where(combined_mask, slc_samples, 0)

    # Compute cross-correlation
    numer = jnp.dot(masked_slc, jnp.conj(masked_slc.T))

    # Compute auto-correlations
    a1 = jnp.sum(jnp.abs(masked_slc) ** 2, axis=-1)
    a2 = a1[:, None]

    # Compute covariance matrix
    cov_mat = numer / jnp.sqrt(a1 * a2)

    return cov_mat


def coh_mat_vectorized2(slc_samples, neighbor_mask=None):
    nslc, nsamps = slc_samples.shape

    # We assume the covariance matrix is initialized to zeros
    jnp.zeros((nslc, nslc), dtype=slc_samples.dtype)

    # Setup mask
    if neighbor_mask is None:
        neighbor_mask = jnp.ones(nsamps, dtype=jnp.bool_)

    valid_samples_mask = ~jnp.isnan(slc_samples)
    combined_mask = valid_samples_mask & neighbor_mask[None, :]

    # Inner computations
    c1_expanded = slc_samples[:, :, None]
    c2_expanded = slc_samples[:, None, :]
    s1s2_conj = c1_expanded * jnp.conjugate(c2_expanded)

    numer = jnp.where(
        combined_mask[:, :, None] & combined_mask[:, None, :], s1s2_conj, 0
    ).sum(axis=-1)

    a1 = (
        jnp.where(combined_mask, slc_samples * jnp.conjugate(slc_samples), 0)
        .sum(axis=-1)
        .reshape(1, -1)
    )
    a2 = (
        jnp.where(combined_mask, slc_samples * jnp.conjugate(slc_samples), 0)
        .sum(axis=-1)
        .reshape(-1, 1)
    )

    a_prod = a1 * a2
    coherence = jnp.where(jnp.abs(a_prod) < 1e-6, 0.0, numer / jnp.sqrt(a_prod))
    # coherence = jnp.where(
    #     jnp.abs(a1 * a2) < 1e-6, 0.0, numer / jnp.sqrt(a1[:, None] * a2[None, :])
    # )

    # Fill diagonal with 1.0
    diag_indices = jnp.arange(nslc)
    coherence = jnp.where(diag_indices[:, None] == diag_indices, 1.0, coherence)

    # # Ensure the matrix is symmetric
    # upper_tri_indices = jnp.triu_indices(nslc, k=1)
    # coherence = coherence.at[upper_tri_indices[::-1]].set(jnp.conjugate(coherence[upper_tri_indices]))

    return coherence


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
            if abs(a_prod) < 1e-6:  # noqa: SIM108
                cov = 0.0 + 0.0j
            else:
                # cov = np.nansum(c1 * np.conjugate(c2)) / np.sqrt(a_prod)
                cov = numer / csqrt(a_prod)

            cov_mat[ti, tj] = cov
            cov_mat[tj, ti] = np.conjugate(cov)

    return cov_mat


def test(slc_samples):
    """Given a (n_slc, n_samps) samples, estimate the coherence matrix."""
    nslc, nsamps = slc_samples.shape
    numermat = np.zeros((nslc, nslc), dtype=slc_samples.dtype)
    a1mat = np.zeros((nslc, nslc), dtype=slc_samples.dtype)
    a2mat = np.zeros((nslc, nslc), dtype=slc_samples.dtype)

    for ti in range(nslc):
        for tj in range(ti + 1, nslc):
            c1, c2 = slc_samples[ti, :], slc_samples[tj, :]
            # a1 = np.nansum(np.abs(c1) ** 2)
            # a2 = np.nansum(np.abs(c2) ** 2)
            # Manually sum to skip based on the neighbor mask
            numer = a1 = a2 = 0.0
            for sidx in range(nsamps):
                s1 = c1[sidx]
                s2 = c2[sidx]
                if np.isnan(s1) or np.isnan(s2):
                    continue
                numer += s1 * s2.conjugate()
                a1 += s1 * s1.conjugate()
                a2 += s2 * s2.conjugate()

            numermat[ti, tj] = numer
            a1mat[ti, tj] = a1
            a2mat[ti, tj] = a2

    return numermat, a1mat, a2mat


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
        compression_opts = {
            "compression": "lzf",
            "shuffle": True,
        }
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
