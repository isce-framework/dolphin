import numpy as np
import numpy.linalg as la
from numba import njit

from .mle import coh_mat, evd, mle


@njit(cache=True)
def seed(a):
    """Seed the random number generator for numba.

    https://numba.readthedocs.io/en/stable/reference/numpysupported.html#initialization
    """
    np.random.seed(a)


@njit(cache=True)
def _ccg_noise(N: int) -> np.array:
    return (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)


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
    signal_phase = truth + std_random * np.random.randn(len(t))
    signal_phase = signal_phase - signal_phase[0]

    # wrap the phase to -pi to p
    signal_phase = np.angle(np.exp(1j * signal_phase))
    truth = np.angle(np.exp(1j * (truth - truth[0])))

    return signal_phase, truth


def rmse(x, y):
    """Calculate the root mean squared error between two arrays."""
    return np.sqrt(np.mean((x - y) ** 2))


def plot_compare_mle_evd(ns=200, unwrap=False, seed=None):
    """Compare the results of the MLE and EVD methods."""
    import matplotlib.pyplot as plt

    np.random.seed(seed)
    C, signal = simulate_C(num_acq=30, Tau0=12, gamma_inf=0, add_signal=True)
    # C = simulate_C(num_acq=30, Tau0=12, gamma_inf=0, add_signal=True)
    samps = simulate_neighborhood_stack(C, ns)
    C_hat = coh_mat(samps)

    truth = signal
    est_evd = np.angle(evd(C_hat))
    est_mle = np.angle(mle(C_hat))

    idxs = np.arange(0, len(est_evd))

    fig, ax = plt.subplots()
    if unwrap:

        def u(x):
            return np.unwrap(x)

    else:

        def u(x):
            return x

    ax.plot(idxs, u(truth), lw=4, label="truth")
    ax.plot(
        idxs, u(est_evd), lw=3, label="EVD: RMSE={:.2f}".format(rmse(truth, est_evd))
    )
    ax.plot(
        idxs, u(est_mle), lw=2, label="MLE: RMSE={:.2f}".format(rmse(truth, est_mle))
    )
    ax.legend()


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
