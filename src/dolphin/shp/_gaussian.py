"""Gaussian weighting for multilooking."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit

from dolphin.utils import compute_out_shape


@partial(
    jit,
    static_argnames=["halfwin_rowcol", "strides", "input_shape"],
)
def estimate_neighbors(
    halfwin_rowcol: tuple[int, int],
    input_shape: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
) -> Array:
    """Generate Gaussian weights for multilooking.

    Unlike GLRT or KS methods, Gaussian multilooking uses a fixed weight
    pattern (same for all pixels) based on a 2D Gaussian window.

    Parameters
    ----------
    halfwin_rowcol : tuple[int, int]
        Half the size of the window in (row, col) dimensions.
    input_shape : tuple[int, int]
        Shape of the input image (rows, cols).
    strides : tuple[int, int]
        The (row, col) strides to use for the sliding window.
        By default (1, 1), meaning output size equals input size.

    Returns
    -------
    weights : Array, 4D
        Float array of Gaussian weights for each pixel in the window.
        Shape is (out_rows, out_cols, window_rows, window_cols).
        Weights are normalized so they sum to 1 (excluding the center pixel).

    """
    rows, cols = input_shape
    half_row, half_col = halfwin_rowcol

    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    window_rsize = 2 * half_row + 1
    window_csize = 2 * half_col + 1

    # Create a 2D Gaussian window
    # Use sigma = half_window / 2 so that the window covers ~2 sigma
    sigma_row = half_row / 2.0 if half_row > 0 else 0.5
    sigma_col = half_col / 2.0 if half_col > 0 else 0.5

    # Create coordinate grids centered at 0
    row_coords = jnp.arange(window_rsize) - half_row
    col_coords = jnp.arange(window_csize) - half_col

    # Compute 2D Gaussian: exp(-(r^2/(2*sr^2) + c^2/(2*sc^2)))
    row_gauss = jnp.exp(-0.5 * (row_coords / sigma_row) ** 2)
    col_gauss = jnp.exp(-0.5 * (col_coords / sigma_col) ** 2)
    gaussian_window = jnp.outer(row_gauss, col_gauss)

    # Set center pixel to 0 (don't include self in weighting, matching GLRT behavior)
    gaussian_window = gaussian_window.at[half_row, half_col].set(0.0)

    # Normalize so weights sum to 1
    gaussian_window = gaussian_window / jnp.sum(gaussian_window)

    # Broadcast to all output pixels (same weights for each pixel)
    # Shape: (out_rows, out_cols, window_rows, window_cols)
    weights = jnp.broadcast_to(
        gaussian_window[None, None, :, :],
        (out_rows, out_cols, window_rsize, window_csize),
    )

    return weights
