from __future__ import annotations

from math import log

import numba
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2

from dolphin.io import compute_out_shape
from dolphin.utils import _get_slices


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: dict = {"x": 1, "y": 1},
    alpha: float = 0.05,
):
    """Estimate the number of neighbors based on the GLRT.

    Assumes Rayleigh distributed amplitudes, based on the method described [1]_.

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

    References
    ----------
        [1] Parizzi and Brcic, 2011, "Adaptive InSAR Stack Multilooking Exploiting
        Amplitude Statistics"
    """
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    threshold = get_glrt_cutoff(alpha=alpha, N=nslc)

    out_rows, out_cols = compute_out_shape((rows, cols), strides)
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=np.bool_
    )
    strides_rowcol = (strides["y"], strides["x"])
    return _loop_over_pixels(
        mean, var, half_row, half_col, nslc, threshold, strides_rowcol, is_shp
    )


@numba.njit(nogil=True, parallel=True)
def _loop_over_pixels(
    mean: np.ndarray,
    var: np.ndarray,
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    N: int,
    threshold: float,
    is_shp: np.ndarray,
) -> np.ndarray:
    """Compare the GLRT test statistic for each pixel to the pre-computed threshold."""
    half_row, half_col = halfwin_rowcol
    row_strides, col_strides = strides_rowcol
    # location to start counting from in the larger input
    r0, c0 = row_strides // 2, col_strides // 2
    in_rows, in_cols = mean.shape
    out_rows, out_cols = is_shp.shape[:2]

    sigma_hat_squared = var + mean**2

    for out_r in numba.prange(out_rows):
        for out_c in range(out_cols):
            in_r = r0 + out_r * row_strides
            in_c = c0 + out_c * col_strides

            sigma_hat_1 = sigma_hat_squared[in_r, in_c]
            # Clamp the window to the image bounds
            (r_start, r_end), (c_start, c_end) = _get_slices(
                half_row, half_col, in_r, in_c, in_rows, in_cols
            )
            # for i in range(-half_row, half_row + 1):
            # for j in range(-half_col, half_col + 1):
            for i in range(r_start, r_end):
                for j in range(c_start, c_end):
                    if i == in_r and j == in_c:
                        is_shp[out_r, out_c, i, j] = True
                        continue
                    sigma_hat_2 = sigma_hat_squared[i, j]
                    sigma_hat_pooled = (sigma_hat_1 + sigma_hat_2) / 2
                    T = (
                        2 * N * log(sigma_hat_pooled)
                        - N * log(sigma_hat_1)
                        - N * log(sigma_hat_2)
                    )
                    is_shp[out_r, out_c, i + half_row, j + half_col] = T < threshold
    return is_shp


def get_glrt_cutoff(alpha: float, N: int) -> float:
    """Compute the cutoff for the GLRT test statistic.

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
    # Degrees of freedom for the chi-squared distribution
    df = 1

    # Inverse of the chi-squared cumulative distribution function (CDF) at alpha
    cutoff = chi2.ppf(1 - alpha, df)
    # cutoff = chi2.ppf(1 - alpha, N)

    return cutoff
