"""Module for simulating stacks of SLCs to test phase linking algorithms.

Contains simple versions of MLE and EVD estimator to compare against the
full CPU/GPU stack implementations.
"""

import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndi
from numba import njit
from numpy.typing import ArrayLike


@njit(cache=True)
def _ccg_noise(N: int) -> np.array:
    return (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)


rng = np.random.default_rng()


@njit(cache=True)
def _seed(a):
    """Seed the random number generator for numba.

    https://numba.readthedocs.io/en/stable/reference/numpysupported.html#initialization
    """
    np.random.seed(a)


def ccg_noise(N: int) -> np.array:
    """Create N samples of standard complex circular Gaussian noise."""
    return (
        rng.normal(scale=1 / np.sqrt(2), size=2 * N)
        .astype(np.float32)
        .view(np.complex64)
    )


def simulate_neighborhood_stack(
    corr_matrix: ArrayLike, neighbor_samples: int = 200
) -> np.array:
    """Simulate a stack of samples from a given correlation matrix.

    Parameters
    ----------
    corr_matrix : ArrayLike
        The correlation matrix to use for generating the samples.
    neighbor_samples : int
        The number of samples to generate for each pixel in the neighborhood.

    Returns
    -------
    np.array
        A 2D array of shape (N, neighbor_samples) containing the simulated samples,
        where N is the number of pixels in the neighborhood.

    """
    N = corr_matrix.shape[0]
    noise = ccg_noise(neighbor_samples * N)
    noise_as_columns = noise.reshape(N, neighbor_samples)

    L = np.linalg.cholesky(corr_matrix)
    samps = L @ noise_as_columns
    return samps


@njit(cache=True)
def simulate_coh(
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

    # wrap the signal_phase to -pi to pi
    signal_phase = np.mod(signal_phase + np.pi, 2 * np.pi) - np.pi
    truth = np.mod((truth - truth[0]) + np.pi, 2 * np.pi) - np.pi

    return signal_phase.astype(np.float64), truth.astype(np.float64)


def rmse(x, y, axis=None):
    """Calculate the root mean squared error between two arrays.

    If x and y are complex, the RMSE is calculated using the angle between them.
    """
    if np.iscomplexobj(x) and np.iscomplexobj(y):
        return np.sqrt(np.mean((np.angle(x * y.conj()) ** 2), axis=axis))

    return np.sqrt(np.mean((x - y) ** 2, axis=axis))


@njit(cache=True)
def mle(cov_mat, beta=0.00):
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


def make_defo_stack(
    shape: tuple[int, int, int], sigma: float, max_amplitude: float = 1
) -> np.ndarray:
    """Create the time series of deformation to add to each SAR date.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the deformation stack (num_time_steps, rows, cols).
    sigma : float
        Standard deviation of the Gaussian deformation.
    max_amplitude : float, optional
        Maximum amplitude of the final deformation. Defaults to 1.

    Returns
    -------
    np.ndarray
        Deformation stack with time series, shape (num_time_steps, rows, cols).

    """
    num_time_steps, rows, cols = shape
    shape2d = (rows, cols)
    # Get shape of deformation in final form (normalized to 1 max)
    final_defo = make_gaussian(shape=shape2d, sigma=sigma).reshape((1, *shape2d))
    final_defo *= max_amplitude / np.max(final_defo)
    # Broadcast this shape with linear evolution
    time_evolution = np.linspace(0, 1, num=num_time_steps)[:, None, None]
    return (final_defo * time_evolution).astype(np.float32)


def create_noisy_deformation(C: np.ndarray, defo_stack: np.ndarray) -> np.ndarray:
    """Create noisy deformation samples given a covariance matrix and deformation stack.

    Parameters
    ----------
    C : np.ndarray
        Covariance matrix of shape (num_time, num_time).
    defo_stack : np.ndarray
        Deformation stack of shape (num_time, rows, cols).

    Returns
    -------
    np.ndarray
        Noisy deformation samples of shape (num_time, rows, cols).

    """

    def _get_diffs(stack: np.ndarray) -> np.ndarray:
        """Create all differences between the deformation stack.

        Parameters
        ----------
        stack : np.ndarray
            Signal stack of shape (num_time, rows, cols).

        Returns
        -------
        np.ndarray, complex64
            Covariance phases of shape (rows, cols, num_time, num_time).

        """
        # Create all possible differences using broadcasting
        # shape: (num_time, 1, rows, cols)
        stack_i = np.exp(1j * stack[:, np.newaxis, :, :])
        # shape: (1, num_time, rows, cols)
        stack_j = np.exp(1j * stack[np.newaxis, :, :, :])
        diff_stack = stack_i * stack_j.conj()

        # Reshape to (rows, cols, num_time, num_time)
        return diff_stack.transpose(2, 3, 0, 1)

    num_time, rows, cols = defo_stack.shape
    shape2d = (rows, cols)
    num_pixels = np.prod(shape2d)

    assert C.shape == (num_time, num_time)

    C_tiled = np.tile(C, (*shape2d, 1, 1))
    signal_cov = _get_diffs(defo_stack)
    C_tiled_with_signal = C_tiled * signal_cov
    C_unstacked = C_tiled_with_signal.reshape(num_pixels, num_time, num_time)

    noise = ccg_noise(num_time * num_pixels)
    noise_unstacked = noise.reshape(num_pixels, num_time, 1)

    L_unstacked = np.linalg.cholesky(C_unstacked)
    samps = L_unstacked @ noise_unstacked

    samps3d = samps.reshape(*shape2d, num_time)
    return np.moveaxis(samps3d, -1, 0)


def make_gaussian(
    shape: tuple[int, int],
    sigma: float,
    row: int | None = None,
    col: int | None = None,
    normalize: bool = False,
    amp: float | None = None,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Create a Gaussian blob of given shape and width.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols)
    sigma : float
        Standard deviation of the Gaussian.
    row : int, optional
        Center row of the blob. Defaults to None.
    col : int, optional
        Center column of the blob. Defaults to None.
    normalize : bool, optional
        Normalize the amplitude peak to 1. Defaults to False.
    amp : float, optional
        Peak height of the Gaussian. Defaults to None.
    noise_sigma : float, optional
        Standard deviation of random Gaussian noise added to the image. Defaults to 0.0.

    Returns
    -------
    ndarray
        Gaussian blob.

    """
    delta = np.zeros(shape)
    rows, cols = shape
    if col is None:
        col = cols // 2
    if row is None:
        row = rows // 2
    delta[row, col] = 1

    out = ndi.gaussian_filter(delta, sigma, mode="constant") * sigma**2
    if normalize or amp is not None:
        out /= out.max()
    if amp is not None:
        out *= amp
    if noise_sigma > 0:
        out += noise_sigma * np.random.standard_normal(shape)
    return out
