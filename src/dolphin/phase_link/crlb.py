import numpy as np
from numpy.linalg import inv
from numpy.typing import ArrayLike


def compute_crlb(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0
) -> np.ndarray:
    r"""Compute the Cramer-Rao Lower Bound (CRLB) for phase linking estimation.

    Uses notation from [@Tebaldini2010MethodsPerformancesMultiPass], such that
    the Fisher information matrix, $X$, is computed as

    \begin{equation}
        X = \frac{2}{L} (\Gamma \circ \Gamma^{-1} - I)
    \end{equation}

    where $\Gamma$ is the complex coherence matrix, $L$ is the number of looks,
    and $I$ is the identity matrix.

    The CRLB is then computed as

    \begin{equation}
        \mathrm{CRLB} = \mathrm{inv}(\mathrm{\Theta}^T X \mathrm{\Theta})
    \end{equation}

    where $\mathrm{\Theta}$ is a matrix of partial derivatives, which, for direct
    phase estimation, is an identity matrix with on extra row of zeros.

    If the APS variance is non-zero, the CRLB is modified as

    \begin{equation}
        \mathrm{CRLB} = \mathrm{inv}(\mathrm{\Theta}^T (X + \mathrm{R}_\mathrm{APS}^{-1}) \mathrm{\Theta})
    \end{equation}

    where $\mathrm{R}_\mathrm{APS}^{-1}$ is the inverse of the APS covariance matrix,
    $\mathrm{R}_\mathrm{APS} = \alpha I

    See Equations (21) and (22) in [@Tebaldini2010MethodsPerformancesMultiPass].

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex coherence matrix (N x N)
    num_looks : int
        Number of looks used in estimation
    aps_variance : float
        Variance of the atmospheric phase screen.
        If 0, no the portion of the fisher information matrix corresponding
        to the APS variance is skipped, and only phase decorrelation is considered.

    Returns
    -------
    np.ndarray
        Array (shape (N,)) of standard deviations (in radians) for the estimator
        variance lower bound at each date.

    """  # noqa: E501
    N = coherence_matrix.shape[0]

    # For direct phase estimation, Theta should be (N x (N-1))
    # This maps N-1 phase differences to N phases
    Theta = np.zeros((N, N - 1))
    # First row is 0 (using day 0 as reference)
    Theta[1:, :] = np.eye(N - 1)  # Last N-1 rows are identity

    # Compute X matrix as in equation (17)
    abs_coherence = np.abs(coherence_matrix)
    X = 2 * num_looks * (abs_coherence * inv(abs_coherence) - np.eye(N))

    if aps_variance == 0:
        # Compute CRLB portions in equation (21)
        fim = Theta.T @ X @ Theta  # Now should be (N-1 x N-1)
        inv_fim = inv(fim)
    else:
        # Add APS contribution
        R_aps_inv = np.eye(N) / aps_variance
        # Otherwise, use full hybrid version, equation (22)
        fim = Theta.T @ (X + R_aps_inv) @ Theta
        inv_fim = inv(fim - Theta.T @ X @ inv(X + R_aps_inv) @ X @ Theta)

    return inv_fim


def compute_lower_bound_std(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0
) -> np.ndarray:
    """Compute the Cramer Rao lower bound on the phase linking estimator variance.

    Returns the result as a standard standard deviation (in radians) per epoch.

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex (true) coherence matrix (N x N)
    num_looks : int
        Number of looks used in estimation
    aps_variance : float
        Variance of the APS, in radians.
        If 0, The bound only considers the variance due to phase decorrelation, not
        atmospheric noise.
        Default is 0.

    Returns
    -------
    lower_bound_std : np.ndarray
        Lower bound on the standard deviation of the phase linking estimator.

    """
    crlb = compute_crlb(
        coherence_matrix=coherence_matrix,
        num_looks=num_looks,
        aps_variance=aps_variance,
    )

    estimator_stddev = np.sqrt(np.diag(crlb))
    return np.concatenate(([0], estimator_stddev))


def _examples(N=10, gamma0=0.6, rho=0.8):
    """Make example covariance matrices used in Tebaldini, 2010."""
    idxs = np.abs(np.arange(N).reshape(-1, 1) - np.arange(N).reshape(1, -1))
    # {Γ}nm = ρ^|n−m|; ρ = 0.8  # noqa: RUF003
    C_ar1 = rho**idxs
    # {Γ}nm = γ0 + (1 - γ0) δ{n-m};  # noqa: RUF003
    C_const_gamma = (1 - gamma0) * np.eye(N) + gamma0 * np.ones((N, N))
    return C_ar1, C_const_gamma
