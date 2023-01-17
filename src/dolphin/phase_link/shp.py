from math import gcd

import numba
import numpy as np


@numba.njit
def ks_2samp(data1, data2):
    """Compute the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes must be equal.

    Returns
    -------
    statistic : float
        KS statistic.
    pvalue : float
        Two-tailed p-value.

    Notes
    -----
    This is a simplified version of the scipy.stats.ks_2samp function.
    https://github.com/scipy/scipy/blob/v1.10.0/scipy/stats/_stats_py.py#L7948-L8179
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if n1 != n2:
        raise ValueError("Data passed to ks_2samp must be of the same size")
    if min(n1, n2) == 0:
        raise ValueError("Data passed to ks_2samp must not be empty")

    data_all = np.concatenate((data1, data2))
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    cddiffs = cdf1 - cdf2
    # Ensure sign of minS is not negative.
    # np.clip not yet implemented in earlier numba, at least up to 0.53
    minS = np.maximum(0.0, np.minimum(1.0, -np.min(cddiffs)))

    maxS = np.max(cddiffs)
    d = max(minS, maxS)
    g = gcd(n1, n2)
    prob = -np.inf

    # n1, n2 are the sample sizes
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    # d is the computed max difference in ECDFs
    d = h * 1.0 / lcm
    if h == 0:
        prob = 1.0
    prob = _compute_prob_outside_square(n1, h)

    prob = np.maximum(0, np.minimum(1, prob))
    return (d, prob)


@numba.njit
def _compute_prob_outside_square(n, h):
    """Compute the proportion of paths that pass outside the two diagonal lines.

    Taken from https://github.com/scipy/scipy/blob/v1.10.0/scipy/stats/_stats_py.py#L7788

    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n

    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.
    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2*( binom(2n, n-h) - binom(2n, n-2a)+binom(2n, n-3a) - ... ) / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


@numba.njit
def ks_test_2(x1, x2, alpha=0.05):
    """Kolmogorov-Smirnov test for two samples.

    Adapted from https://github.com/isce-framework/fringe/blob/main/tests/KS2/ks2test.py

    Parameters
    ----------
    x1 : array_like
        First sample.
    x2 : array_like
        Second sample.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    H : int
        H = 1 if the null hypothesis is rejected,
        H = 0 if the null hypothesis is not rejected.
    pValue : float
        p-value of the test.
    ks_statistic : float
        KS statistic.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    .. [2] https://www.mathworks.com/help/stats/kstest2.html

    """
    binEdges = np.hstack(
        (np.array([0.0]), np.sort(np.concatenate((x1, x2))), np.array([100000000.0]))
    )

    binCounts1 = np.histogram(x1, binEdges)[0]
    binCounts2 = np.histogram(x2, binEdges)[0]

    sampleCDF1 = np.cumsum(binCounts1) / np.sum(binCounts1)
    sampleCDF2 = np.cumsum(binCounts2) / np.sum(binCounts2)

    deltaCDF = np.abs(sampleCDF1 - sampleCDF2)

    ks_statistic = deltaCDF.max()

    n1 = len(x1)
    n2 = len(x2)

    n = float(n1 * n2) / (n1 + n2)
    lambd = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks_statistic, 0.0)

    j = np.linspace(1, 101, 101)
    pValue = 2 * sum(
        (np.power(-1, j - 1) * np.exp(-2 * lambd * lambd * np.power(j, 2)))
    )  # multiple by 2 in MATLAB
    pValue = max(0.0, min(1.0, pValue))

    # assign the H values after calculation is done
    if alpha >= pValue:
        H = 1
    else:
        H = 0

    return H, pValue, ks_statistic
