from __future__ import annotations

from math import exp, gcd, sqrt

import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike

from ._utils import _get_slices


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
    (r_start, r_end), (c_start, c_end) = _get_slices(
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
