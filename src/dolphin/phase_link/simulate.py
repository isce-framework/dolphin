"""Module for simulating stacks of SLCs to test phase linking algorithms.

Contains simple versions of MLE and EVD estimator to compare against the
full CPU/GPU stack implementations.
"""

from __future__ import annotations

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
    """Simulate one set of samples from a given correlation matrix.

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
    num_time = corr_matrix.shape[0]
    noise = ccg_noise(neighbor_samples * num_time)
    noise_as_columns = noise.reshape(num_time, neighbor_samples)

    L = np.linalg.cholesky(corr_matrix)
    samps = L @ noise_as_columns
    return samps


def simulate_coh(
    num_acq: int = 50,
    gamma_inf: float = 0.1,
    gamma0: float = 0.999,
    Tau0: float = 72,
    acq_interval: int = 12,
    add_signal: bool = False,
    signal_std: float = 0.1,
) -> np.array:
    """Simulate a correlation matrix for a pixel.

    Parameters
    ----------
    num_acq : int
        The number of acquisitions.
    gamma_inf : float
        The asymptotic coherence value.
    gamma0 : float
        The initial coherence value.
    Tau0 : float
        The coherence decay time constant.
    acq_interval : int
        The acquisition interval in days.
    add_signal : bool
        Whether to add a simulated signal.
    signal_std : float
        The standard deviation of the simulated signal.

    Returns
    -------
    np.array
        The simulated correlation matrix.
    np.array
        The simulated truth signal.

    """
    time_series_length = num_acq * acq_interval
    time = np.arange(0, time_series_length, acq_interval)
    if add_signal:
        k, signal_rate = 1, 2
        noisy_truth, truth = _sim_signal(
            time,
            signal_rate=signal_rate,
            std_random=signal_std,
            k=k,
        )
    else:
        noisy_truth = truth = np.zeros(len(time), dtype=np.float64)

    C = simulate_coh_stack(time, gamma0, gamma_inf, Tau0, noisy_truth).squeeze()
    return C, truth


def simulate_coh_stack(
    time: np.ndarray,
    gamma0: np.ndarray,
    gamma_inf: np.ndarray,
    Tau0: np.ndarray,
    signal: np.ndarray,
    amplitudes: ArrayLike | None = None,
) -> np.ndarray:
    """Create a coherence matrix at each pixel.

    Parameters
    ----------
    time : np.ndarray
        Array of time values, arbitrary units.
        shape (N,) where N is the number of acquisitions.
    gamma0 : np.ndarray
        Initial coherence value for each pixel.
    gamma_inf : np.ndarray
        Asymptotic coherence value for each pixel.
    Tau0 : np.ndarray
        Coherence decay time constant for each pixel.
    signal : np.ndarray
        Simulated signal phase for each pixel.
    amplitudes: ArrayLike, optional
        If provided, set the amplitudes of the output pixels.
        Default is to use all ones.

    Returns
    -------
    np.ndarray
        The simulated coherence matrix for each pixel.

    """
    num_time = time.shape[0]
    time_diff = time[:, None] - time[None, :]
    time_diff = time_diff[None, None, :, :]
    gamma0 = np.atleast_2d(gamma0)[:, :, None, None]
    gamma_inf = np.atleast_2d(gamma_inf)[:, :, None, None]
    Tau0 = np.atleast_2d(Tau0)[:, :, None, None]
    if amplitudes is None:
        amplitudes = np.ones((num_time, 1, 1), dtype=np.float32)

    gamma = (gamma0 - gamma_inf) * np.exp(time_diff / Tau0) + gamma_inf
    phase_diff = signal[:, None] - signal[None, :]
    # Multiply each pixel by (a a^T)
    A = np.einsum("irc,jrc->rcij", amplitudes, amplitudes)
    C = A * gamma * np.exp(1j * phase_diff)

    rl, cl = np.tril_indices(num_time, k=-1)

    C[..., rl, cl] = np.conj(np.transpose(C, axes=(0, 1, 3, 2))[..., rl, cl])

    # Reset the diagonals of each pixel to 1
    rs, cs = np.diag_indices(num_time)
    C[:, :, rs, cs] = 1
    return C


def _sim_signal(
    t: np.ndarray,
    signal_rate: float = 1.0,
    std_random: float = 0,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a signal with a linear trend, seasonal component, and random noise.

    Parameters
    ----------
    t : np.ndarray
        Time array in days.
    signal_rate : float, optional
        Linear rate of the signal in radians per year, by default 1.0.
    std_random : float, optional
        Standard deviation of the random noise component, by default 0.
    k : int, optional
        Seasonal parameter, 1 for annual and 2 for semi-annual, by default 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the simulated signal phase and the true signal.

    """
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


def make_noisy_samples(
    C: ArrayLike,
    defo_stack: ArrayLike,
) -> np.ndarray:
    """Create noisy deformation samples given a covariance matrix and deformation stack.

    Parameters
    ----------
    C : ArrayLike,
        Covariance matrix of shape (num_time, num_time) , or you can pass
        on matrix per pixel as shape (rows, cols, num_time, num_time).
    defo_stack : ArrayLike,
        Deformation stack of shape (num_time, rows, cols).

    Returns
    -------
    samps3d: np.ndarray
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
        stack_i = np.exp(1j * stack[:, None, :, :])
        # shape: (1, num_time, rows, cols)
        stack_j = np.exp(1j * stack[None, :, :, :])
        diff_stack = stack_i * stack_j.conj()

        # Reshape to (rows, cols, num_time, num_time)
        return diff_stack.transpose(2, 3, 0, 1)

    num_time, rows, cols = defo_stack.shape
    shape2d = (rows, cols)
    num_pixels = np.prod(shape2d)

    C_tiled = np.tile(C, (*shape2d, 1, 1)) if C.squeeze().ndim == 2 else C

    if C_tiled.shape != (rows, cols, num_time, num_time):
        raise ValueError(f"{C_tiled.shape=}, but {defo_stack.shape=}")
    # Reset the diagonals of each pixel to 1
    rs, cs = np.diag_indices(num_time)
    C_tiled[:, :, rs, cs] = 1

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
