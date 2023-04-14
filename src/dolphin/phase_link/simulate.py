"""Module for simulating stacks of SLCs to test phase linking algorithms.

Contains simple versions of MLE and EVD estimator to compare against the
full CPU/GPU stack implementations.
"""
import numpy as np
import numpy.linalg as la
from numba import njit


@njit(cache=True)
def _ccg_noise(N: int) -> np.array:
    return (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)


@njit(cache=True)
def _seed(a):
    """Seed the random number generator for numba.

    https://numba.readthedocs.io/en/stable/reference/numpysupported.html#initialization
    """
    np.random.seed(a)


@njit(cache=True)
def simulate_sample(corr_matrix: np.array) -> np.array:
    """Simulate one sample from a given correlation matrix.

    Parameters
    ----------
    corr_matrix : np.array
        The correlation matrix

    Returns
    -------
    np.array
    """
    w, v = la.eigh(corr_matrix)
    w[w < 1e-3] = 0.0
    w = w.astype(v.dtype)

    v_star = np.conj(v.T)  # Hermitian
    C = v @ np.diag(np.sqrt(w)) @ v_star
    z = _ccg_noise(corr_matrix.shape[0])
    z = z.astype(C.dtype)
    return C @ z


@njit(cache=True)
def simulate_neighborhood_stack(
    corr_matrix: np.array, neighbor_samples: int = 200
) -> np.array:
    """Simulate a matrix of neighborhood samples (num_slc, num_samples).

    Parameters
    ----------
    corr_matrix : np.array
        The correlation matrix to use for the simulation.
        Size is (num_slc, num_slc)
    neighbor_samples : int, optional
        Number of samples to simulate, by default 200

    Returns
    -------
    np.array
        A stack of neighborhood samples
        size (corr_matrix.shape[0], neighbor_samples)
    """
    nslc = corr_matrix.shape[0]
    # A 2D matrix for a neighborhood over time.
    # Each column is the neighborhood complex data for each acquisition date
    neighbor_stack = np.zeros((nslc, neighbor_samples), dtype=np.complex64)
    for ii in range(neighbor_samples):
        slcs = simulate_sample(corr_matrix)
        # To ensure that the neighborhood is homogeneous,
        # we set the amplitude of all SLCs to one
        neighbor_stack[:, ii] = np.exp(1j * np.angle(slcs))

    return neighbor_stack


@njit(cache=True)
def simulate_C(
    num_acq=50,
    gamma_inf=0.1,
    gamma0=0.999,
    Tau0=72,
    acq_interval=12,
    add_signal=False,
    signal_std=0.1,
):
    """Simulate a correlation matrix for a pixel."""
    time_series_length = num_acq * acq_interval
    t = np.arange(0, time_series_length, acq_interval)
    if add_signal:
        k, signal_rate = 1, 2
        signal_phase, truth = _sim_signal(
            t,
            signal_rate=signal_rate,
            std_random=signal_std,
            k=k,
        )
    else:
        signal_phase = truth = np.zeros(len(t), dtype=np.float64)

    C = _sim_coherence_mat(t, gamma0, gamma_inf, Tau0, signal_phase)
    return C, truth
    # return C


@njit(cache=True)
def _sim_coherence_mat(t, gamma0, gamma_inf, Tau0, signal):
    length = t.shape[0]
    C = np.ones((length, length), dtype=np.complex64)
    for ii in range(length):
        for jj in range(ii + 1, length):
            gamma = (gamma0 - gamma_inf) * np.exp((t[ii] - t[jj]) / Tau0) + gamma_inf
            C[ii, jj] = gamma * np.exp(1j * (signal[ii] - signal[jj]))
            C[jj, ii] = np.conj(C[ii, jj])

    return C


@njit(cache=True)
def _sim_signal(
    t,
    signal_rate: float = 1.0,
    std_random: float = 0,
    k: int = 1,
):
    # time_series_length: length of time-series in days
    # acquisition_interval: time-difference between subsequent acquisitions (days)
    # signal_rate: linear rate of the signal (rad/year)
    # k: seasonal parameter, 1 for annual and  2 for semi-annual
    truth = signal_rate * (t - t[0]) / 365.0
    if k > 0:
        seasonal = np.sin(2 * np.pi * k * t / 365.0) + np.cos(2 * np.pi * k * t / 365.0)
        truth += seasonal

    # adding random temporal signal (which simulates atmosphere + DEM error + ...)
    signal_phase = truth + std_random / 2 * np.random.randn(len(t))
    # we divided std by 2 since we're subtracting the first value
    signal_phase = signal_phase - signal_phase[0]

    # wrap the phase to -pi to p
    signal_phase = np.angle(np.exp(1j * signal_phase))
    truth = np.angle(np.exp(1j * (truth - truth[0])))

    return signal_phase, truth


def rmse(x, y):
    """Calculate the root mean squared error between two arrays."""
    return np.sqrt(np.mean((x - y) ** 2))


@njit(cache=True)
def mle(cov_mat, beta=0.01):
    """Estimate the linked phase using the MLE estimator.

    Parameters
    ----------
    cov_mat : np.array
        The sample covariance matrix
    beta : float, optional
        The regularization parameter, by default 0.0

    Returns
    -------
    np.array
        The estimated linked phase
    """
    dtype = cov_mat.dtype
    cov_mat = cov_mat.astype(dtype)
    # estimate the wrapped phase based on the EMI paper
    # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix
    Gamma = np.abs(cov_mat).astype(dtype)
    if beta:
        Gamma = (1 - beta) * Gamma + beta * np.eye(Gamma.shape[0], dtype=Gamma.dtype)
        Gamma_inv = la.inv(Gamma).astype(dtype)
    else:
        Gamma_inv = la.inv(Gamma).astype(dtype)
    _, v = la.eigh(Gamma_inv * cov_mat)

    # smallest eigenvalue is idx 0
    # reference to the first acquisition
    evd_estimate = v[:, 0] * np.conjugate(v[0, 0])
    return evd_estimate.astype(dtype)


@njit(cache=True)
def evd(cov_mat):
    """Estimate the linked phase the largest eigenvector of `cov_mat`."""
    # estimate the wrapped phase based on the eigenvalue decomp of the cov. matrix
    # n = len(cov_mat)
    # lambda_, v = la.eigh(cov_mat, subset_by_index=[n - 1, n - 1])  # only scipy.linalg
    # v = v.flatten()
    lambda_, v = la.eigh(cov_mat)

    # Biggest eigenvalue is the last one
    # reference to the first acquisition
    evd_estimate = v[:, -1] * np.conjugate(v[0, -1])

    return evd_estimate.astype(cov_mat.dtype)
