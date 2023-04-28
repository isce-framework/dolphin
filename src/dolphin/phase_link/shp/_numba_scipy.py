from __future__ import annotations

from math import lgamma

import numpy as np
from numba import njit, prange
from scipy.stats import f, t


@njit
def _trans(x, loc, scale):
    inv_scale = type(scale)(1) / scale
    return (x - loc) * inv_scale


@njit
def _logpdf_t(x, df, loc=0, scale=1):
    T = type(df)
    z = _trans(x, loc, scale)
    k = T(0.5) * (df + T(1))
    c = lgamma(k) - lgamma(T(0.5) * df)
    c -= T(0.5) * np.log(df * T(np.pi))
    c -= np.log(scale)
    for i in prange(len(z)):
        z[i] = -k * np.log(T(1) + (z[i] * z[i]) / df) + c
    return z


@njit
def _log_beta(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)


@njit
def _logpdf_f(x, dfn, dfd, loc=0, scale=1):
    n = 1.0 * dfn
    m = 1.0 * dfd
    z = _trans(x, loc, scale)

    log_numer = m / 2 * np.log(m) + n / 2 * np.log(n) + (n / 2 - 1) * np.log(z)
    log_denom = ((n + m) / 2) * np.log(m + n * z) + _log_beta(n / 2, m / 2)

    lPx = log_numer - log_denom
    return lPx


def get_f_critical_values(alpha: float, dfn: int, dfd: int) -> tuple[float, float]:
    """Get the critical values for the two-tailed F-distribution.

    Parameters
    ----------
    alpha : float
        The significance level.
    dfn : int
        The numerator degrees of freedom.
    dfd : int
        The denominator degrees of freedom.

    Returns
    -------
    float, float
        The lower and upper critical values.
    """
    crit_value_f_lower = f.ppf(alpha / 2, dfn, dfd)
    crit_value_f_upper = f.ppf(1 - alpha / 2, dfn, dfd)
    return crit_value_f_lower, crit_value_f_upper


def get_t_critical_values(alpha: float, df: int) -> tuple[float, float]:
    """Get the critical values for the two-tailed t-distribution.

    Parameters
    ----------
    alpha : float
        The significance level.
    df : int
        The degrees of freedom.

    Returns
    -------
    float, float
        The lower and upper critical values.
    """
    crit_value_t_lower = t.ppf(alpha / 2, df)
    crit_value_t_upper = t.ppf(1 - alpha / 2, df)
    return crit_value_t_lower, crit_value_t_upper
