from __future__ import annotations

from math import log

import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike

from ._utils import _get_slices


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
def estimate_neighbors_cpu(
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
    (r_start, r_end), (c_start, c_end) = _get_slices(
        half_row, half_col, r, c, rows, cols
    )

    # Compute the KL distance for each pixel in the block
    for i in range(r_start, r_end):
        for j in range(c_start, c_end):
            kld = _kl_dist_gpu(mean[r, c], mean[i, j], var[r, c], var[i, j])
            out_neighbor_arrays[r, c, i - r, j - c] = kld <= threshold
