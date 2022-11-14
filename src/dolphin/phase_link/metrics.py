import numpy as np
from numba import njit

from dolphin.log import get_log

logger = get_log()


@njit(cache=True)
def estimate_temp_coh(est, cov_matrix):
    """Estimate the temporal coherence of the neighborhood.

    Parameters
    ----------
    est : np.array
        The estimated/solved phase timeseries.
    cov_matrix : np.array
        The covariance matrix of the neighborhood.

    Returns
    -------
    float
        The temporal coherence of the time series compared to cov_matrix.
    """
    gamma = 0
    N = len(est)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            theta = np.angle(cov_matrix[i, j])
            phi = np.angle(est[i] * np.conj(est[j]))

            gamma += np.exp(1j * theta) * np.exp(-1j * phi)
            count += 1
    # assert count == (N * (N - 1)) / 2
    return np.abs(gamma) / count


# @njit(cache=True)
# def estimate_temp_coh_block(est, cov_matrix):
#     """Estimate the temporal coherence of the neighborhood.

#     Parameters
#     ----------
#     est : np.array
#         The estimated/solved phase timeseries.
#     cov_matrix : np.array
#         The covariance matrix of the neighborhood.

#     Returns
#     -------
#     float
#         The temporal coherence of the time series compared to cov_matrix.
#     """
#     try:
#         import cupy as cp

#         xp = cp.get_array_module(cov_matrix)
#     except ImportError:
#         logger.debug("cupy not installed, falling back to numpy")
#         xp = np

#     gamma = 0
#     N = len(est)
#     count = 0
#     for i in range(N):
#         for j in range(i + 1, N):
#             theta = np.angle(cov_matrix[i, j])
#             phi = np.angle(est[i] * np.conj(est[j]))

#             gamma += np.exp(1j * theta) * np.exp(-1j * phi)
#             count += 1
#     # assert count == (N * (N - 1)) / 2
#     return np.abs(gamma) / count
