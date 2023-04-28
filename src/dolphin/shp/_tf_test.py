from __future__ import annotations

# import cupy as cp
import numba
import numpy as np
from numba import prange
from numpy.typing import ArrayLike
from scipy.stats import f, t


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    n: int,
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
    n : int
        Number of images in the stack used to compute `mean` and `var`.
        Used to compute the degrees of freedom for the t- and F-tests to
        determine the critical values.
    alpha : float, default=0.05
        Significance level at which to reject the null hypothesis.
        Rejecting means declaring a neighbor is not a SHP.

    Returns
    -------
    is_shp : np.ndarray, 4D
        Boolean array marking which neighbors are SHPs for each pixel in the block.
        Shape is (rows, cols, window_rows, window_cols), where
            window_rows = 2 * halfwin_rowcol[0] + 1
            window_cols = 2 * halfwin_rowcol[1] + 1
    """
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    # we're doing two checks with the same alpha, so we need to adjust
    # the significance level for each test so P_FA(t-test) & P_FA(f-test) = alpha
    a = 1 - (1 - alpha) ** (1 / 2)
    # for a = 0.05, this is 0.0253

    cv_t = get_t_critical_values(a, n)
    cv_f = get_f_critical_values(a, n)
    is_shp = np.zeros(
        (rows, cols, 2 * half_row + 1, 2 * half_col + 1), dtype=mean.dtype
    )
    _loop_over_pixels(
        mean, var, half_row, half_col, n, cv_t[0], cv_t[1], cv_f[0], cv_f[1], is_shp
    )
    return is_shp


@numba.njit(nogil=True, parallel=True, fastmath=True)
def _loop_over_pixels(
    mean,
    variance,
    half_row,
    half_col,
    n,
    cv_t_low,
    cv_t_high,
    cv_f_low,
    cv_f_high,
    is_shp,
) -> None:
    rows, cols = mean.shape
    for r in prange(half_row, rows - half_row):
        for c in range(half_col, cols - half_col):
            mu1 = mean[r, c]
            var1 = variance[r, c]
            for i in range(-half_row, half_row + 1):
                for j in range(-half_col, half_col + 1):
                    mu2 = mean[r + i, c + j]
                    var2 = variance[r + i, c + j]
                    # welch_t_statistic(mu1, mu2, var1, var2, n)
                    t_stat = (mu1 - mu2) / np.sqrt((var1 + var2) / n)
                    # F-test: ratio of variances
                    f_stat = var1 / var2

                    # 2-sided tests for t- and f-test
                    passes_t = cv_t_low < t_stat < cv_t_high
                    passes_f = cv_f_low < f_stat < cv_f_high
                    is_shp[r, c, i + half_row, j + half_col] = passes_t and passes_f


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
