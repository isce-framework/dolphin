from __future__ import annotations

import csv
from functools import lru_cache, partial
from pathlib import Path

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from numpy.typing import ArrayLike

from dolphin.utils import compute_out_shape


@lru_cache
def _read_cutoff_csv():
    filename = Path(__file__).parent / "glrt_cutoffs.csv"

    result = {}
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            n = int(row["N"])
            alpha = float(row["alpha"])
            cutoff = float(row["cutoff"])
            result[(n, alpha)] = cutoff

    return result


@partial(jit, static_argnames=["halfwin_rowcol", "strides", "nslc", "alpha"])
def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: tuple[int, int] = (1, 1),
    alpha: float = 0.001,
):
    """Estimate the number of neighbors based on the GLRT."""
    # Convert mean/var to the Rayleigh scale parameter
    rows, cols = mean.shape
    half_row, half_col = halfwin_rowcol
    row_strides, col_strides = strides
    # window_size = rsize * csize

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2
    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    scale_squared = (var + mean**2) / 2
    threshold = get_cutoff_jax(alpha=alpha, N=nslc)

    def _get_window(arr, r: int, c: int, half_row: int, half_col: int) -> Array:
        r0 = r - half_row
        c0 = c - half_col
        start_indices = (r0, c0)

        rsize = 2 * half_row + 1
        csize = 2 * half_col + 1
        slice_sizes = (rsize, csize)

        return lax.dynamic_slice(arr, start_indices, slice_sizes)

    def _process_row_col(out_r, out_c):
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides

        scale_1 = scale_squared[in_r, in_c]  # One pixel
        # and one window for scale 2, will broadcast
        scale_2 = _get_window(scale_squared, in_r, in_c, half_row, half_col)
        # Compute the GLRT test statistic.
        scale_pooled = (scale_1 + scale_2) / 2
        test_stat = 2 * jnp.log(scale_pooled) - jnp.log(scale_1) - jnp.log(scale_2)

        return threshold > test_stat

    # Now make a 2D grid of indices to access all output pixels
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    # Create the vectorized function in 2d
    _process_2d = vmap(_process_row_col)
    # Then in 3d
    _process_3d = vmap(_process_2d)
    return _process_3d(out_r_indices, out_c_indices)


def get_cutoff(alpha: float, N: int) -> float:
    r"""Compute the upper cutoff for the GLRT test statistic.

    Statistic is

    \[
    2\log(\sigma_{pooled}) - \log(\sigma_{p}) -\log(\sigma_{q})
    \]

    Parameters
    ----------
    alpha: float
        Significance level (0 < alpha < 1).
    N: int
        Number of samples.

    Returns
    -------
    float
        Cutoff value for the GLRT test statistic.

    """
    n_alpha_to_cutoff = _read_cutoff_csv()
    return n_alpha_to_cutoff[(max(N, 50), alpha)]


get_cutoff_jax = jit(get_cutoff, static_argnames=["alpha", "N"])
