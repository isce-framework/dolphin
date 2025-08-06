from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.scipy.linalg import solve
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
    N = np.asarray(coherence_matrix).shape[0]

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
        A = Theta.T @ X @ inv(X + R_aps_inv) @ X @ Theta
        fim = Theta.T @ X @ Theta - A
        inv_fim = inv(fim)

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


def _theta_X_theta_T(X: Array, ref: int) -> Array:  # noqa: N802
    """Project an (N,N) FIM to (N-1,N-1) by deleting `ref` row/col.

    Equivalent to the matmul:
        Theta.T @ X @ Theta

    where Theta is the (N,N-1) matrix with the row `ref` zero and the rest identity.
    """
    idx = jnp.concatenate([jnp.arange(ref), jnp.arange(ref + 1, X.shape[-1])])
    return X[..., idx[:, None], idx]


@partial(jit, static_argnums=(1, 2, 3, 4))
def compute_crlb_jax(
    coherence_matrices: Array,
    num_looks: int,
    reference_idx: int,
    aps_variance: float = 1e-2,
    jitter: float = 1e-4,
) -> Array:
    """Batched CRLB std-dev (per epoch) for a stack of Fisher Information Matrices.

    See Tebaldini 2010, eqs. 21-22).

    Parameters
    ----------
    coherence_matrices : Array
        Complex Γ with shape (..., N, N).  Leading dimensions are batched.
    num_looks : int
        Number of independent looks, `L`.
        Note that too-large `L` will lead to numerical instability.
    reference_idx : int
        Index of the reference epoch.
        Must be in the range [0, N-1].
    aps_variance : float
        Variance of the APS.
        Set to 0 to ignore APS contribution.
        Default is 1e-2.
    jitter : float
        Diagonal fudge added to each SPD solve for extra robustness.
        Default is 1e-4.

    Returns
    -------
    Array
        Standard-deviation lower bounds with shape (..., N) in radians.
        Element 0 (reference epoch) is fixed to 0.

    """
    *batch, N, _ = coherence_matrices.shape
    eye_N = jnp.eye(N, dtype=coherence_matrices.dtype)
    eye_N1 = jnp.eye(N - 1, dtype=coherence_matrices.dtype)

    # Fisher information X
    #    X = 2/L * (|Γ| ∘ |Γ|⁻¹ - I)
    abs_G = jnp.abs(coherence_matrices)
    eyeN_batch = jnp.broadcast_to(eye_N, abs_G.shape)
    abs_G = abs_G + jitter * eyeN_batch  # regularize to promote positive definiteness
    abs_G_inv = solve(abs_G, eyeN_batch, assume_a="pos")

    X = 2.0 * num_looks * (abs_G * abs_G_inv - eyeN_batch)  # Hadamard

    # Decorrelation-only term  Θᵀ X Θ
    F_base = _theta_X_theta_T(X, reference_idx)

    # APS hybrid correction (eq. 22) if `aps_variance` != 0
    #     A = Θᵀ X (X + R⁻¹)⁻¹ X Θ  ,  R⁻¹ = I/alpha
    #     We compute (X + R⁻¹)⁻¹ (XΘ) via solve().
    if aps_variance > 1e-6:
        R_inv = eyeN_batch / aps_variance  # (...,N,N)
        X_plus_R = X + R_inv + jitter * eyeN_batch  # SPD + εI
        X_plus_R_inv = solve(X_plus_R, eyeN_batch, assume_a="pos")
        A = _theta_X_theta_T(X @ X_plus_R_inv @ X, reference_idx)
        FIM = F_base - A
    else:
        FIM = F_base

    # Invert FIM via solve();   Σ = FIM⁻¹
    FIM = FIM + jitter * jnp.broadcast_to(eye_N1, FIM.shape)
    Sigma = solve(FIM, jnp.broadcast_to(eye_N1, FIM.shape), assume_a="pos")

    sig = jnp.sqrt(jnp.diagonal(Sigma, axis1=-2, axis2=-1))  # (...,N-1)
    # insert 0 for reference epoch
    sig = jnp.concatenate([jnp.zeros((*batch, 1), dtype=sig.dtype), sig], axis=-1)
    return sig


def _examples(N=10, gamma0=0.6, rho=0.8):
    """Make example covariance matrices used in Tebaldini, 2010."""
    idxs = np.abs(np.arange(N).reshape(-1, 1) - np.arange(N).reshape(1, -1))
    # {Γ}nm = ρ^|n−m|; ρ = 0.8  # noqa: RUF003
    C_ar1 = rho**idxs
    # {Γ}nm = γ0 + (1 - γ0) δ{n-m};  # noqa: RUF003
    C_const_gamma = (1 - gamma0) * np.eye(N) + gamma0 * np.ones((N, N))
    return C_ar1, C_const_gamma


def demo_from_slc_stack(  # noqa: D103
    slc_vrt_filename: str = "slc_stack.vrt",
    hw: tuple[int, int] = (5, 5),
    center_pixel: tuple[int, int] = (50, 50),
    aps_variance: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    from dolphin import io
    from dolphin.phase_link import covariance

    hwr, hwc = hw
    reader = io.VRTStack.from_vrt_file(slc_vrt_filename)
    r0, c0 = center_pixel
    samples = reader[:, r0 - hwr : r0 + hwr, c0 - hwc : c0 + hwc].reshape(
        len(reader), -1
    )
    C = covariance.coh_mat_single(samples)
    # num_looks = (2 * hwr + 1) * (2 * hwc + 1)
    num_looks = np.sqrt(hwr * hwc)
    return C, compute_lower_bound_std(C, num_looks, aps_variance=aps_variance)
