from __future__ import annotations

from typing import Optional

import numba
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2


@numba.njit
def pooled_var(mu1: ArrayLike, mu2: ArrayLike, var1: ArrayLike, var2: ArrayLike):
    r"""Compute the pooled variance for sample group given their means and variances.

    Parameters
    ----------
    mu1 : ArrayLike
        Mean of the first group
    mu2 : ArrayLike
        Mean of the second group
    var1 : ArrayLike
        Variance of the first group
    var2 : ArrayLike
        Variance of the second group

    Returns
    -------
    ArrayLike
        Pooled variance

    """
    return (var1 + var2 - mu1 * mu2) / 2 + (mu1**2 + mu2**2) / 4


@numba.njit
def glrt_statistic(
    mu1: ArrayLike, mu2: ArrayLike, var1: ArrayLike, var2: ArrayLike
) -> ArrayLike:
    r"""Compute the GLRT statistic for two groups (assuming same size).

    Parameters
    ----------
    mu1 : ArrayLike
        Mean of the first group
    mu2 : ArrayLike
        Mean of the second group
    var1 : ArrayLike
        Variance of the first group
    var2 : ArrayLike
        Variance of the second group

    Returns
    -------
    ArrayLike
        GLRT statistic

    Notes
    -----
    The GLRT statistic is defined [1]_ as

    \[
    GLRT = \frac{(\mu_1 - \mu_2)^2}{\sigma^2}
    \]

    where :math:`\sigma^2` is the pooled variance of the two groups.

    References
    ----------
    .. [1] Cover, Thomas M., and Joy A. Thomas. Elements of information theory.
    """
    pooled_var_ = pooled_var(mu1, mu2, var1, var2)
    return (mu1 - mu2) ** 2 / pooled_var_


@numba.njit(fastmath=True, nogil=True)
def _norm_pdf(x, mu, sigma):
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma**2)
    return coeff * np.exp(exponent)


@numba.njit
def get_likelihood_ratio(
    x1: ArrayLike,
    x2: ArrayLike,
    mu1: Optional[ArrayLike] = None,
    mu2: Optional[ArrayLike] = None,
    var1: Optional[ArrayLike] = None,
    var2: Optional[ArrayLike] = None,
    dist: str = "norm",
):
    """Compute the likelihoods for the two groups.

    Parameters
    ----------
    x1 : ArrayLike
        First group
    x2 : ArrayLike
        Second group
    mu1 : Optional[ArrayLike]
        Mean of the first group. If None, it is computed from the data.
    mu2 : Optional[ArrayLike]
        Mean of the second group. If None, it is computed from the data.
    var1 : Optional[ArrayLike]
        Variance of the first group. If None, it is computed from the data.
    var2 : Optional[ArrayLike]
        Variance of the second group. If None, it is computed from the data.
    dist : str
        Distribution to use. Can be "norm" or "rice".

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Likelihoods for the first and second groups

    """
    if mu1 is None:
        mu1 = np.mean(x1)
    if mu2 is None:
        mu2 = np.mean(x2)
    if var1 is None:
        var1 = np.var(x1)
    if var2 is None:
        var2 = np.var(x2)

    mu_x = (mu1 + mu2) / 2
    var_x = pooled_var(mu1, mu2, var1, var2)
    L0, L1 = _get_likelihoods_gaussian(x1, x2, mu1, mu2, var1, var2, mu_x, var_x)
    return L1 / L0
    # if dist == "norm":
    #     L1, L0 = _get_likelihoods_gaussian(x1, x2, mu1, mu2, var1, var2, mu_x, var_x)
    # elif dist == "rice":
    #     L1, L0 = _get_likelihoods_rice(x1, x2, mu1, mu2, var1, var2, mu_x, var_x)
    # return L1 / L0


@numba.njit
def _get_likelihoods_gaussian(x1, x2, mu1, mu2, var1, var2, mu_x, var_x):
    # Likelihoods under the combined distribution (Gaussian)
    likelihoods_x1_H0 = _norm_pdf(x1, mu_x, np.sqrt(var_x))
    likelihoods_x2_H0 = _norm_pdf(x2, mu_x, np.sqrt(var_x))
    # Total likelihood under H0
    L0 = np.prod(likelihoods_x1_H0) * np.prod(likelihoods_x2_H0)
    # Likelihoods under their own distributions (Gaussian)
    likelihoods_x1_H1 = _norm_pdf(x1, mu1, np.sqrt(var1))
    likelihoods_x2_H1 = _norm_pdf(x2, mu2, np.sqrt(var2))

    # Total likelihood under H1
    L1 = np.prod(likelihoods_x1_H1) * np.prod(likelihoods_x2_H1)
    return L0, L1


@numba.njit
def _get_likelihoods_rice(x1, x2, mu1, mu2, var1, var2, mu_x, var_x):
    # Estimate the shape parameter (nu) for the Rice distribution for each sample:
    # Assuming non-negative data (x1 and x2)
    nu1 = np.sqrt(mu1**2 / (2 * var1))
    nu2 = np.sqrt(mu2**2 / (2 * var2))

    # Compute the likelihoods under the null hypothesis (H0):
    # Adjust the shape parameter (nu) accordingly
    nu_x = np.sqrt(mu_x**2 / (2 * var_x))
    # Likelihoods under the combined distribution (Rice)
    # TODO: do i use scale=1? or scale=np.sqrt(var_x)?
    # likelihoods_x1_H0_rice = rice.pdf(x1, nu_x)
    # likelihoods_x2_H0_rice = rice.pdf(x2, nu_x)
    likelihoods_x1_H0_rice = _rice_pdf(x1, nu_x, np.sqrt(var_x))
    likelihoods_x2_H0_rice = _rice_pdf(x2, nu_x, np.sqrt(var_x))

    # Total likelihood under H0
    L0_rice = np.prod(likelihoods_x1_H0_rice) * np.prod(likelihoods_x2_H0_rice)

    # Compute the likelihoods under the alternative hypothesis (H1):
    # Likelihoods under their own distributions (Rice)
    # likelihoods_x1_H1_rice = rice.pdf(x1, nu1)
    # likelihoods_x2_H1_rice = rice.pdf(x2, nu2)
    likelihoods_x1_H1_rice = _rice_pdf(x1, nu1, sigma=np.sqrt(var1))
    likelihoods_x2_H1_rice = _rice_pdf(x2, nu2, sigma=np.sqrt(var2))

    # Total likelihood under H1
    L1_rice = np.prod(likelihoods_x1_H1_rice) * np.prod(likelihoods_x2_H1_rice)
    return L0_rice, L1_rice


def get_critical_value(alpha: float, df: int) -> float:
    """Get the critical value for the GLRT statistic.

    Parameters
    ----------
    alpha : float
        Significance level
    df : int
        Degrees of freedom

    Returns
    -------
    float
        Critical value
    """
    return chi2.ppf(1 - alpha, df)


def estimate_neighbors_glrt(
    amp_stack: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    mean: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    alpha: float = 0.05,
    dist: str = "norm",
):
    """Estimate the number of neighbors for the GLRT statistic.

    Parameters
    ----------
    amp_stack : ArrayLike, 3D
        Input amplitude data
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    mean : ArrayLike, 2D
        Mean amplitude of each pixel. If None, it is computed from the data.
    var: ArrayLike, 2D
        Variance of each pixel's amplitude. If None, it is computed from the data.
    alpha : float
        Significance level. Default is 0.05.
    dist : str
        Distribution to use. Can be "norm" or "rice".

    Returns
    -------
    int
        Number of neighbors

    """
    n, rows, cols = amp_stack.shape
    # GLRT statistic is a chi-squared with n(?) degree of freedom
    cv = get_critical_value(alpha, n)
    cv = 3.841458820694124
    # TODO: Make a numba stencil for the GLRT statistic
    half_row, half_col = halfwin_rowcol

    if mean is None:
        mean = np.mean(amp_stack, axis=0)
    if var is None:
        var = np.var(amp_stack, axis=0)

    is_shp = np.zeros(
        (rows, cols, 2 * half_row + 1, 2 * half_col + 1), dtype=amp_stack.dtype
    )
    return _loop_over_pixels(
        amp_stack, half_row, half_col, rows, cols, mean, var, cv, dist, is_shp
    )


@numba.njit
def _loop_over_pixels(
    amp_stack,
    half_row,
    half_col,
    rows,
    cols,
    mean,
    variance,
    cv,
    dist,
    is_shp,
):
    for r in range(half_row, rows - half_row):
        for c in range(half_col, cols - half_col):
            x1 = amp_stack[:, r, c]
            mu1 = mean[r, c]
            var1 = variance[r, c]
            for i in range(-half_row, half_row + 1):
                for j in range(-half_col, half_col + 1):
                    x2 = amp_stack[:, r + i, c + j]
                    mu2 = mean[r + i, c + j]
                    var2 = variance[r + i, c + j]
                    lr = get_likelihood_ratio(x1, x2, mu1, mu2, var1, var2, dist=dist)
                    # is_shp[r, c, i + half_row, j + half_col] = lr > cv
                    is_shp[r, c, i + half_row, j + half_col] = lr
    return is_shp


@numba.njit
def _bessel_i0(x, num_terms=30):
    result = 0.0
    factor = 1.0
    term = 1.0

    for k in range(num_terms):
        result += term
        factor *= x / 2 / (k + 1)
        term = factor**2

    return result


@numba.njit
def _rice_pdf(x, nu, sigma: float = 1.0):
    out = np.zeros_like(x)
    coeff = x / (sigma**2)
    exponent = -((x**2 + nu**2) / (2 * sigma**2))

    # bessel_term = _bessel_i0((x * nu) / (sigma**2))
    for i in range(len(x)):
        bessel_term = _bessel_i0((x[i] * nu) / (sigma**2))
        out[i] = coeff[i] * np.exp(exponent[i]) * bessel_term
    return out
