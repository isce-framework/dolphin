"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from jax.typing import ArrayLike

from dolphin._types import Filename, HalfWindow, Strides
from dolphin.utils import compute_out_shape

DEFAULT_STRIDES = Strides(1, 1)


@partial(jit, static_argnames=["half_window", "strides"])
def estimate_stack_covariance(
    slc_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    neighbor_arrays: Optional[np.ndarray] = None,
) -> Array:
    """Estimate the linked phase at all pixels of `slc_stack`.

    Parameters
    ----------
    slc_stack : ArrayLike
        The SLC stack, with shape (n_slc, n_rows, n_cols).
    half_window : tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.

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
        msg = "The SLC stack must be complex."
        raise ValueError(msg)
    # Get the dimensions
    nslc, rows, cols = slc_stack.shape

    row_strides, col_strides = strides
    half_col, half_row = half_window

    out_rows, out_cols = compute_out_shape(
        (rows, cols), {"x": col_strides, "y": row_strides}
    )

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2

    padded_slc_stack = slc_stack
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    if neighbor_arrays is None:
        neighbor_arrays = jnp.ones(
            (out_rows, out_cols, 2 * half_window[0] + 1, 2 * half_window[1] + 1),
            dtype=bool,
        )

    def _process_row_col(out_r, out_c):
        """Get slices for and process one pixel's window."""
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides
        slc_window = _get_stack_window(padded_slc_stack, in_r, in_c, half_row, half_col)
        slc_samples = slc_window.reshape(nslc, -1)
        cur_neighbors = neighbor_arrays[out_r, out_c, :, :]
        neighbor_mask = cur_neighbors.ravel()

        return coh_mat_single(slc_samples, neighbor_mask=neighbor_mask)

    _process_2d = vmap(_process_row_col)
    _process_3d = vmap(_process_2d)
    return _process_3d(out_r_indices, out_c_indices)


def _get_stack_window(
    padded_stack: ArrayLike, r: int, c: int, half_row: int, half_col: int
) -> Array:
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


def coh_mat_single(
    slc_samples: ArrayLike, neighbor_mask: Optional[ArrayLike] = None
) -> Array:
    """Given (n_slc, n_samps) SLC samples, estimate the coherence matrix."""
    _, nsamps = slc_samples.shape

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
