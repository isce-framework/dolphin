from __future__ import annotations

# import cupy as cp
import numba
import numpy as np
from numba import prange
from numpy.typing import ArrayLike
from scipy.stats import f, t

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
    """Estimate the number of neighbors based on a combined t- and F-test.

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
    """
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    # we're doing two checks with the same alpha, so we need to adjust
    # the significance level for each test so P_FA(t-test) & P_FA(f-test) = alpha
    # e.g. for alpha = 0.05, a = 0.0253
    a = 1 - (1 - alpha) ** (1 / 2)
    cv_t = get_t_critical_values(a, nslc)
    cv_f = get_f_critical_values(a, nslc)

    out_rows, out_cols = compute_out_shape((rows, cols), strides)
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=mean.dtype
    )
    strides_rowcol = (strides["y"], strides["x"])
    _loop_over_pixels(
        mean, var, half_row, half_col, nslc, *cv_t, *cv_f, strides_rowcol, is_shp
    )
    return is_shp


@numba.njit(nogil=True, parallel=True, fastmath=True)
def _loop_over_pixels(
    mean: ArrayLike,
    variance: ArrayLike,
    half_row: int,
    half_col: int,
    nslc: int,
    cv_t_low: float,
    cv_t_high: float,
    cv_f_low: float,
    cv_f_high: float,
    strides_rowcol: tuple[int, int],
    is_shp: ArrayLike,
) -> None:
    in_rows, in_cols = mean.shape
    out_rows, out_cols = is_shp.shape[:2]
    row_strides, col_strides = strides_rowcol
    # location to start counting from in the larger input
    r0, c0 = row_strides // 2, col_strides // 2

    for out_r in prange(out_rows):
        for out_c in range(out_cols):
            in_r = r0 + out_r * row_strides
            in_c = c0 + out_c * col_strides
            mu1 = mean[in_r, in_c]
            var1 = variance[in_r, in_c]
            # Clamp the window to the image bounds
            (r_start, r_end), (c_start, c_end) = _get_slices(
                half_row, half_col, in_r, in_c, in_rows, in_cols
            )
            for i in range(r_start, r_end):
                for j in range(c_start, c_end):
                    # itself is always a neighbor
                    if i == in_r and j == in_c:
                        is_shp[out_r, out_c, i, j] = True
                        continue

                    # t-test: test for difference of means
                    mu2 = mean[i, j]
                    var2 = variance[i, j]
                    # welch_t_statistic(mu1, mu2, var1, var2, nslc)
                    t_stat = (mu1 - mu2) / np.sqrt((var1 + var2) / nslc)

                    # F-test: test for difference of variances
                    f_stat = var1 / var2

                    # critical values: use 2-sided tests for t- and f-test
                    passes_t = cv_t_low < t_stat < cv_t_high
                    passes_f = cv_f_low < f_stat < cv_f_high
                    # Needs to pass both tests to be a SHP
                    is_shp[out_r, out_c, i, j] = passes_t and passes_f


def get_t_critical_values(alpha: float, n: int) -> tuple[float, float]:
    """Get the critical values for the two-tailed t-distribution.

    Parameters
    ----------
    alpha : float
        The significance level.
    n : int
        The number of samples in each group.

    Returns
    -------
    float, float
        The lower and upper critical values.
    """
    dof = 2 * (n - 1)
    crit_value_t_lower = t.ppf(alpha / 2, dof)
    crit_value_t_upper = t.ppf(1 - alpha / 2, dof)
    return crit_value_t_lower, crit_value_t_upper


def get_f_critical_values(alpha: float, n: int) -> tuple[float, float]:
    """Get the critical values for the two-tailed F-distribution.

    Parameters
    ----------
    alpha : float
        The significance level.
    n : int
        The number of samples in each group.

    Returns
    -------
    float, float
        The lower and upper critical values.
    """
    dfn = dfd = n - 1  # degrees of freedom, same for numerator and denominator
    crit_value_f_lower = f.ppf(alpha / 2, dfn, dfd)
    crit_value_f_upper = f.ppf(1 - alpha / 2, dfn, dfd)
    return crit_value_f_lower, crit_value_f_upper
