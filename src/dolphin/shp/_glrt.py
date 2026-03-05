from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from numpy.typing import ArrayLike
from scipy import stats

from dolphin.utils import compute_out_shape


@partial(
    jit,
    static_argnames=["halfwin_rowcol", "strides", "nslc", "alpha"],
)
def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    nslc: int = 1,
    alpha: float = 0.001,
):
    """Estimate the number of neighbors based on the GLRT.

    Based on the method described in [@Parizzi2011AdaptiveInSARStack].
    Assumes Rayleigh distributed amplitudes ([@Siddiqui1962ProblemsConnectedRayleigh])

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel.
    var: ArrayLike, 2D
        Variance of each pixel's amplitude.
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    strides: tuple[int, int]
        The (x, y) strides (in pixels) to use for the sliding window.
        By default (1, 1), or no strides (output size = input size).
    nslc : int
        Number of images in the stack used to compute `mean` and `var`.
        Used to compute the degrees of freedom for the statistical test to
        determine the threshold value.
    alpha : float
        Significance level at which to reject the null hypothesis.
        Rejecting means declaring a neighbor is not a SHP.
        Default is 0.001.

    Notes
    -----
    When `strides` is not (1, 1), the output first two dimensions
    are smaller than `mean` and `var` by a factor of `strides`. This
    will match the downstream shape of the strided phase linking results.

    Returns
    -------
    is_shp : np.ndarray, 4D
        Boolean array marking which neighbors are SHPs for each pixel in the block.
        Shape is (out_rows, out_cols, window_rows, window_cols), where
            `out_rows` and `out_cols` are computed by
            `[dolphin.io.compute_out_shape][]`
            `window_rows = 2 * halfwin_rowcol[0] + 1`
            `window_cols = 2 * halfwin_rowcol[1] + 1`

    """
    rows, cols = jnp.asarray(mean).shape
    half_row, half_col = halfwin_rowcol
    row_strides, col_strides = strides

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2
    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    # Convert mean/var to the Rayleigh scale parameter
    scale_squared = (jnp.asarray(var) + jnp.asarray(mean) ** 2) / 2
    # 1 Degree of freedom, regardless of N
    threshold = stats.chi2.ppf(1 - alpha, df=1)

    window_rsize = 2 * half_row + 1
    window_csize = 2 * half_col + 1
    slice_sizes = (window_rsize, window_csize)

    # Create indices for window rows and columns
    window_row_indices = jnp.arange(window_rsize)
    window_col_indices = jnp.arange(window_csize)

    def _get_window(arr, r: int, c: int, half_row: int, half_col: int) -> Array:
        r0 = r - half_row
        c0 = c - half_col
        start_indices = (r0, c0)
        return lax.dynamic_slice(arr, start_indices, slice_sizes)

    def _process_row_col(out_r, out_c):
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides

        scale_1 = scale_squared[in_r, in_c]  # One pixel
        # and one window for scale 2, which broadcasts over scale_1
        scale_2 = _get_window(scale_squared, in_r, in_c, half_row, half_col)

        # Compute the starting indices for the window in the full image
        r0 = in_r - half_row
        c0 = in_c - half_col

        # Determine valid indices within the bounds of the original image
        valid_rows = (window_row_indices >= -r0) & (window_row_indices < rows - r0)
        valid_cols = (window_col_indices >= -c0) & (window_col_indices < cols - c0)
        valid_mask = jnp.outer(valid_rows, valid_cols)

        # Compute the GLRT test statistic
        scale_pooled = (scale_1 + scale_2) / 2
        test_stat = nslc * (
            2 * jnp.log(scale_pooled) - jnp.log(scale_1) - jnp.log(scale_2)
        )
        is_shp = threshold > test_stat

        # Zero out edge pixels where window is not fully in bounds
        is_shp = jnp.where(valid_mask, is_shp, False)
        # Ensure current pixel is not counted as its own neighbor
        return is_shp.at[half_row, half_col].set(False)

    # Now make a 2D grid of indices to access all output pixels
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    # Create the vectorized function to run on all rows/columns
    _process_3d = vmap(vmap(_process_row_col))
    return _process_3d(out_r_indices, out_c_indices)
