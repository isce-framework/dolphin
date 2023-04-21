from __future__ import annotations

from math import exp, gcd, log, sqrt
from typing import Optional

# import cupy as cp
import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike
from scipy.stats import chi2

from ._utils import _get_slices_gpu


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
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if n1 != n2:
        raise ValueError("Data passed to ks_2samp must be of the same size")
    if min(n1, n2) == 0:
        raise ValueError("Data passed to ks_2samp must not be empty")
    if np.iscomplexobj(data1) or np.iscomplexobj(data2):
        raise ValueError("ks_2samp only accepts real input data")

    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate((data1, data2))
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    cdf_diffs = cdf1 - cdf2
    # Ensure sign of minS is not negative.
    # np.clip not yet implemented in earlier numba, at least up to 0.53
    minS = np.maximum(0.0, np.minimum(1.0, -np.min(cdf_diffs)))

    maxS = np.max(cdf_diffs)
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
    else:
        prob = _compute_prob_outside_square(n1, h)

    prob = np.maximum(0, np.minimum(1, prob))
    # return (d, prob)
    return prob


def _ks_2samp_block(data1, data_block):
    """Compute the Kolmogorov-Smirnov statistic on a block."""
    if data_block.ndim == 1:
        return ks_2samp(data1, data_block)
    elif data_block.ndim == 3:
        # For 3D data, reshape to 2D where each col is a pixel
        data_cols = data_block.reshape(data_block.shape[0], -1)
    else:
        data_cols = data_block
    return np.apply_along_axis(lambda d: ks_2samp(d, data1), axis=0, arr=data_cols)


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


# GPU version of the SHP finding algorithm using KS test
@cuda.jit
def estimate_neighbors_ks(
    sorted_amp_stack,
    half_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    alpha: float,
    neighbor_arrays: ArrayLike,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    out_x, out_y = cuda.grid(2)
    out_rows, out_cols = neighbor_arrays.shape[:2]
    # Check if we are within the bounds of the array
    if out_y >= out_rows or out_x >= out_cols:
        return

    num_slc, rows, cols = sorted_amp_stack.shape
    ecdf_dist_cutoff = _get_ecdf_critical_distance_gpu(num_slc, alpha)
    half_row, half_col = half_rowcol

    row_strides, col_strides = strides_rowcol
    r_start = row_strides // 2
    c_start = col_strides // 2
    in_r = r_start + out_y * row_strides
    in_c = c_start + out_x * col_strides

    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices_gpu(
        half_row, half_col, in_r, in_c, rows, cols
    )

    amp_block = sorted_amp_stack[:, r_start:r_end, c_start:c_end]
    # TODO: if this is a bottleneck, we can do something smarter
    # by computing only the bottom right corner, then mirroring
    # also, we can use strides to only compute the output
    # pixels that are actually needed
    neighbors_pixel = neighbor_arrays[out_y, out_x, :, :]
    _get_neighbors(amp_block, half_rowcol, ecdf_dist_cutoff, neighbors_pixel)


@cuda.jit(device=True)
def _get_neighbors(amp_block, half_rowcol, ecdf_dist_cutoff, neighbors):
    _, rows, cols = amp_block.shape

    r_c, c_c = rows // 2, cols // 2
    if rows < 2 * half_rowcol[0] + 1 or cols < 2 * half_rowcol[1] + 1:
        neighbors[:] = True
        return

    x1 = amp_block[:, r_c, c_c]
    for i in range(rows):
        for j in range(cols):
            if i == r_c and j == c_c:
                neighbors[i, j] = True
                continue
            x2 = amp_block[:, i, j]

            ecdf_max_dist = _get_max_cdf_dist_gpu(x1, x2)

            neighbors[i, j] = ecdf_max_dist < ecdf_dist_cutoff


def _get_max_cdf_dist(x1, x2):
    """Get the maximum CDF distance between two arrays.

    Parameters
    ----------
    x1 : np.ndarray
        First array, size n, sorted
    x2 : np.ndarray
        Second array, size n, sorted

    Returns
    -------
    float
        Maximum empirical CDF distance between the two arrays

    Examples
    --------
    >>> x1 = np.array([1, 2, 3, 4, 5])
    >>> x2 = np.array([1, 2, 3, 4, 5])
    >>> _get_max_cdf_dist(x1, x2)
    0
    >>> x2 = np.array([2, 3, 4, 5, 6])
    >>> round(_get_max_cdf_dist(x1, x2), 2)
    0.2
    >>> round(_get_max_cdf_dist(x2, x1), 2)
    0.2
    >>> x2 = np.array([6, 7, 8, 9, 10])
    >>> _get_max_cdf_dist(x1, x2)
    1.0
    """
    n = x1.shape[0]
    i1 = i2 = i_out = 0
    cdf1 = cdf2 = 0
    max_dist = 0
    while i_out < 2 * n:
        if i1 == n:
            cdf2 += 1 / n
            i2 += 1
        elif i2 == n:
            cdf1 += 1 / n
            i1 += 1

        elif x1[i1] < x2[i2]:
            cdf1 += 1 / n
            i1 += 1
        elif x1[i1] > x2[i2]:
            cdf2 += 1 / n
            i2 += 1
        else:  # a tie
            cdf1 += 1 / n
            cdf2 += 1 / n
            i1 += 1
            i2 += 1
            i_out += 1  # jumping 2 ahead for tie
        i_out += 1
        max_dist = max(max_dist, abs(cdf1 - cdf2))
    return max_dist


_get_max_cdf_dist_gpu = cuda.jit(device=True)(_get_max_cdf_dist)
_get_max_cdf_dist_cpu = numba.njit(_get_max_cdf_dist, nogil=True)


def _get_ecdf_critical_distance(nslc, alpha):
    N = nslc / 2.0
    cur_dist = 0.01
    critical_distance = 0.1
    sqrt_N = sqrt(N)
    while cur_dist <= 1.0:
        value = cur_dist * (sqrt_N + 0.12 + 0.11 / sqrt_N)
        pvalue = 0
        for t in range(1, 101):
            pvalue += ((-1) ** (t - 1)) * exp(-2 * (value**2) * (t**2))
        pvalue = max(0.0, min(1.0, 2 * pvalue))

        if pvalue <= alpha:
            critical_distance = cur_dist
            break

        cur_dist += 0.001
    return critical_distance


_get_ecdf_critical_distance_gpu = cuda.jit(device=True)(_get_ecdf_critical_distance)
_get_ecdf_critical_distance_cpu = numba.njit(_get_ecdf_critical_distance)


def kl_dist(mu1, mu2, v1, v2):
    r"""Compute the Kullback-Leibler distance between two Gaussians.

    Parameters
    ----------
    mu1 : float
        Mean of the first Gaussian
    mu2 : float
        Mean of the second Gaussian
    v1 : float
        Variance of the first Gaussian
    v2 : float
        Variance of the second Gaussian

    Returns
    -------
    float
        KL distance between the two Gaussians

    Examples
    --------
    >>> kl_dist(0, 0, 1, 1)
    0.0
    >>> kl_dist(0, 0, 1, 2)
    0.0965735902799727
    >>> kl_dist(0, 0, 2, 1)
    0.1534264097200273
    >>> kl_dist(1, 3, 1, 1)
    2.0

    Notes
    -----
    The KL distance is defined [1]_ as


    \[
    KL(p || q) = \int p(x) \log \frac{p(x)}{q(x)} dx
    \]

    where :math:`p` and :math:`q` are two probability distributions.
    In this case, :math:`p` is a Gaussian with mean :math:`\mu_1` and variance
    :math:`v_1` and :math:`q` is a Gaussian with mean :math:`\mu_2` and
    variance :math:`v_2`. This special case simplifies the integral to

    \[
    KL(p || q) = \frac{1}{2} \log \frac{v_2}{v_1} +
    \frac{v_1 + (\mu_1 - \mu_2)^2}{2 v_2} - \frac{1}{2}
    \]


    References
    ----------
    .. [1] Cover, Thomas M., and Joy A. Thomas. Elements of information theory.
    """
    return (log(v2 / v1) + ((v1 + (mu1 - mu2) ** 2) / v2) - 1) / 2


_kl_dist_gpu = cuda.jit(device=True)(kl_dist)
_kl_dist_cpu = numba.njit(kl_dist, fastmath=True, nogil=True)


@numba.njit
def estimate_neighbors_kl_cpu(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Get the SHPs using the KL distance for each pixel in a block.

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel
    var : ArrayLike, 2D
        Variance of each pixel's amplitude
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    threshold : float, optional
        Threshold for the KL distance, by default 0.5

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

    is_shp = np.zeros(
        (rows, cols, 2 * half_row + 1, 2 * half_col + 1), dtype=numba.bool_
    )
    for r in range(half_row, rows - half_row):
        for c in range(half_col, cols - half_col):
            for i in range(-half_row, half_row + 1):
                for j in range(-half_col, half_col + 1):
                    kld = _kl_dist_cpu(
                        mean[r, c],
                        mean[r + i, c + j],
                        var[r, c],
                        var[r + i, c + j],
                    )
                    is_shp[r, c, i + half_row, j + half_col] = kld <= threshold
    return is_shp


@cuda.jit
def estimate_neighbors_kl_gpu(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    out_neighbor_arrays: ArrayLike,
    threshold: float = 0.5,
):
    """Estimate the number of neighbors for each pixel using KL distance.

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel
    var : ArrayLike, 2D
        Variance of each pixel's amplitude
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    out_neighbor_arrays : ArrayLike, 4D
        Boolean array indicating whether a pixel is a neighbor.
        Shape is (rows, cols, window_rows, window_cols)
    threshold : float
        Threshold for the KL distance to label a pixel as a neighbor
    """
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    c, r = cuda.grid(2)
    if r >= rows or c >= cols:
        return

    # Get the slices to use for the current pixel
    (r_start, r_end), (c_start, c_end) = _get_slices_gpu(
        half_row, half_col, r, c, rows, cols
    )

    # Compute the KL distance for each pixel in the block
    for i in range(r_start, r_end):
        for j in range(c_start, c_end):
            kld = _kl_dist_gpu(mean[r, c], mean[i, j], var[r, c], var[i, j])
            out_neighbor_arrays[r, c, i - r, j - c] = kld <= threshold


# GLRT version
# ############################################


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
