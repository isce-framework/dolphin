from __future__ import annotations

from math import log
from typing import Optional

import numba
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from dolphin._types import Strides
from dolphin.utils import _get_slices, compute_out_shape

from ._common import remove_unconnected

_get_slices = numba.njit(_get_slices)


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: Optional[dict] = None,
    alpha: float = 0.05,
    prune_disconnected: bool = False,
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
    nslc : int
        Number of images in the stack used to compute `mean` and `var`.
        Used to compute the degrees of freedom for the t- and F-tests to
        determine the critical values.
    strides: dict, optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    alpha : float, default=0.05
        Significance level at which to reject the null hypothesis.
        Rejecting means declaring a neighbor is not a SHP.
    prune_disconnected : bool, default=False
        If True, keeps only SHPs that are 8-connected to the current pixel.
        Otherwise, any pixel within the window may be considered an SHP, even
        if it is not directly connected.


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
    if strides is None:
        strides = {"x": 1, "y": 1}
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    # 1 Degree of freedom, regardless of N
    threshold = stats.chi2.ppf(1 - alpha, df=1)

    strides_rowcol = (strides["y"], strides["x"])
    out_rows, out_cols = compute_out_shape((rows, cols), Strides(*strides_rowcol))
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=np.bool_
    )
    return _loop_over_pixels(
        mean,
        var,
        nslc,
        halfwin_rowcol,
        strides_rowcol,
        threshold,
        prune_disconnected,
        is_shp,
    )


@numba.njit(nogil=True)
def _compute_glrt_test_stat(scale_sq_1, scale_sq_2, N):
    """Compute the GLRT test statistic."""
    scale_pooled = (scale_sq_1 + scale_sq_2) / 2
    return N * (2 * log(scale_pooled) - log(scale_sq_1) - log(scale_sq_2))


@numba.njit(nogil=True, parallel=True)
def _loop_over_pixels(
    mean: ArrayLike,
    var: ArrayLike,
    N: int,
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    threshold: float,
    prune_disconnected: bool,
    is_shp: np.ndarray,
) -> np.ndarray:
    """Loop common to SHP tests using only mean and variance."""
    half_row, half_col = halfwin_rowcol
    row_strides, col_strides = strides_rowcol
    # location to start counting from in the larger input
    r0, c0 = row_strides // 2, col_strides // 2
    in_rows, in_cols = mean.shape
    out_rows, out_cols = is_shp.shape[:2]

    # Convert mean/var to the Rayleigh scale parameter
    scale_squared = (var + mean**2) / 2

    for out_r in numba.prange(out_rows):
        for out_c in range(out_cols):
            in_r = r0 + out_r * row_strides
            in_c = c0 + out_c * col_strides

            scale_1 = scale_squared[in_r, in_c]
            # Clamp the window to the image bounds
            (r_start, r_end), (c_start, c_end) = _get_slices(
                half_row, half_col, in_r, in_c, in_rows, in_cols
            )
            if mean[in_r, in_c] == 0:
                # Skip nodata pixels
                continue

            for in_r2 in range(r_start, r_end):
                for in_c2 in range(c_start, c_end):
                    # window offsets for dims 3,4 of `is_shp`
                    r_off = in_r2 - r_start
                    c_off = in_c2 - c_start

                    # Don't count itself as a neighbor
                    if in_r2 == in_r and in_c2 == in_c:
                        is_shp[out_r, out_c, r_off, c_off] = False
                        continue
                    scale_2 = scale_squared[in_r2, in_c2]

                    T = _compute_glrt_test_stat(scale_1, scale_2, N)

                    is_shp[out_r, out_c, r_off, c_off] = threshold > T
            if prune_disconnected:
                # For this pixel, prune the groups not connected to the center
                remove_unconnected(is_shp[out_r, out_c], inplace=True)

    return is_shp
