from math import exp, gcd, sqrt
from typing import Tuple

# import cupy as cp
import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike, NDArray

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


# GPU version of the SHP finding algorithm
@cuda.jit
def estimate_neighbors(
    sorted_amp_stack,
    half_rowcol: Tuple[int, int],
    alpha: float,
    neighbor_arrays,
):
    """Estimate the linked phase at all pixels of `slc_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    c, r = cuda.grid(2)
    num_slc, rows, cols = sorted_amp_stack.shape
    # Check if we are within the bounds of the array
    if r >= rows or c >= cols:
        return

    ecdf_dist_cutoff = _get_ecdf_critical_distance_gpu(num_slc, alpha)
    half_row, half_col = half_rowcol

    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices_gpu(
        half_row, half_col, r, c, rows, cols
    )
    amp_block = sorted_amp_stack[:, r_start:r_end, c_start:c_end]
    # TODO: if this is a bottleneck, we can do something smarter
    # by computing only the bottom right corner, then mirroring
    # also, we can use strides to only compute the output
    # pixels that are actually needed
    neighbors_pixel = neighbor_arrays[r, c, :, :]
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
    0.0
    >>> x2 = np.array([2, 3, 4, 5, 6])
    >>> _get_max_cdf_dist(x1, x2)
    0.2
    >>> _get_max_cdf_dist(x2, x1)
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
        else:
            cdf2 += 1 / n
            i2 += 1
        i_out += 1
        max_dist = max(max_dist, abs(cdf1 - cdf2))
    return max_dist


_get_max_cdf_dist_gpu = cuda.jit(device=True)(_get_max_cdf_dist)
_get_max_cdf_dist_cpu = numba.njit(_get_max_cdf_dist)


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
    return (np.log(v2 / v1) + ((v1 + (mu1 - mu2) ** 2) / v2) - 1) / 2


_kl_dist_gpu = cuda.jit(device=True)(kl_dist)
_kl_dist_cpu = numba.njit(kl_dist)


@numba.njit
def kl_block(
    mean: ArrayLike, var: ArrayLike, halfwin_rowcol: Tuple[int, int]
) -> NDArray:
    """Compute the KL distance for each pixel in a block.

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel
    var : ArrayLike, 2D
        Variance of each pixel's amplitude
    halfwin_rowcol : Tuple[int, int]
        Half the size of the block in (row, col) dimensions

    Returns
    -------
    Nd, 4D
        KL distance for each pixel in the block
        Shape is (rows, cols, window_rows, window_cols)
        where window_rows = 2 * halfwin_rowcol[0] + 1
              window_cols = 2 * halfwin_rowcol[1] + 1


    """
    half_r, half_c = halfwin_rowcol
    rows, cols = mean.shape

    out = np.zeros((rows, cols, 2 * half_r + 1, 2 * half_c + 1))
    for r in range(half_r, rows - half_r):
        for c in range(half_c, cols - half_c):
            for i in range(-half_r, half_r + 1):
                for j in range(-half_c, half_c + 1):
                    kld = kl_dist(
                        mean[r, c],
                        mean[r + i, c + j],
                        var[r, c],
                        var[r + i, c + j],
                    )
                    out[r, c, i + half_r, j + half_c] = kld
    return out


@numba.njit
def kl_dist_2samp(p: ArrayLike, q: ArrayLike) -> float:
    """Compute the Kullback-Leibler distance between two samples.

    Assumes the samples are independent and normally distributed.
    """
    return kl_dist(np.mean(p), np.mean(q), np.var(p), np.var(q))
