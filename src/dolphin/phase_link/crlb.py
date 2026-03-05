from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.scipy.linalg import solve
from numpy.linalg import inv
from numpy.typing import ArrayLike


def compute_crlb(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0.01
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
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0.01
) -> np.ndarray:
    """Compute the Cramer Rao lower bound on the phase linking estimator variance.

    Returns the result as a standard standard deviation (in radians) per epoch.

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex coherence matrix (N x N)
    num_looks : int
        Number of looks used in estimation
    aps_variance : float
        Variance of the APS, in radians squared.
        If 0, The bound only considers the variance due to phase decorrelation, not
        atmospheric noise.
        Default is 0.01.

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


def _theta_indices(n: int, ref: int) -> Array:
    return jnp.concatenate([jnp.arange(ref), jnp.arange(ref + 1, n)])


def _build_fisher_from_abs_gamma(
    abs_G: Array, abs_G_inv: Array, num_looks: float
) -> Array:
    """Create the Fisher Information Matrix from |coherence matrix|.

    Useful when |coherence matrix| is already computed and inverted.

    Parameters
    ----------
    abs_G : Array
        Absolute value of the coherence matrix
    abs_G_inv : Array
        Inverse of the absolute value of the coherence matrix
    num_looks : float
        Number of looks used in the coherence matrix

    Returns
    -------
    Array
        Fisher Information Matrix

    """
    eyeN = jnp.eye(abs_G.shape[-1], dtype=abs_G.dtype)
    eyeN = jnp.broadcast_to(eyeN, abs_G.shape)
    return 2.0 * num_looks * (abs_G * abs_G_inv - eyeN)


def _crlb_from_x(
    X: Array, reference_idx: int, aps_variance: float, fim_jitter: float
) -> Array:
    *batch, N, _ = X.shape
    idx = _theta_indices(N, reference_idx)

    eyeN = jnp.eye(N, dtype=X.dtype)
    eyeN1 = jnp.eye(N - 1, dtype=X.dtype)
    eyeN = jnp.broadcast_to(eyeN, X.shape)
    eyeN1 = jnp.broadcast_to(eyeN1, (*batch, N - 1, N - 1))

    # Θᵀ X Θ  by indexing
    F_base = X[..., idx[:, None], idx]

    if aps_variance > 0.0:
        R_inv = eyeN / aps_variance
        X_plus_R = X + R_inv + 0.0 * eyeN  # no implicit extra jitter here
        # (X + R⁻¹)⁻¹ (X Θ) via solve, where Θ selects columns 'idx'
        X_cols = X[..., :, idx]  # (..., N, N-1)
        AXTheta = solve(X_plus_R, X_cols, assume_a="pos")  # (..., N, N-1)
        A = (X @ AXTheta)[..., idx, :]  # (..., N-1, N-1)
        FIM = F_base - A
    else:
        FIM = F_base

    # Σ = inverse of FIM
    if fim_jitter != 0.0:
        FIM = FIM + fim_jitter * eyeN1
    Sigma = solve(FIM, eyeN1, assume_a="pos")
    sig = jnp.sqrt(jnp.diagonal(Sigma, axis1=-2, axis2=-1))
    return jnp.insert(sig, reference_idx, 0.0, axis=-1)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def compute_crlb_jax(
    coherence_matrices: Array,
    num_looks: int,
    reference_idx: int,
    aps_variance: float = 0.0,
    gamma_jitter: float = 0.0,
    fim_jitter: float = 1e-6,
    mask_zero_blocks: bool = True,
    zero_tol: float = 1e-7,
) -> Array:
    """Compute CRLB for a batch of coherence matrices.

    Parameters
    ----------
    coherence_matrices : Array
        Coherence matrices, shape (..., N, N)
    num_looks : int
        Number of independent looks, `L`.
    reference_idx : int
        Reference epoch index (time index of 0 output)
    aps_variance : float
        Atmospheric phase screen variance.
        If 0, APS term is skipped.
    gamma_jitter : float
        Jitter added to regularize the inversion of |Γ|.
    fim_jitter : float
        Jitter added to regularize the inversion of the Fisher Information Matrix.
    mask_zero_blocks : bool
        Set output to nan where |Γ| is (near) zero.
        Default is True.
    zero_tol : float
        Tolerance for zero-blocks

    """
    *_batch, N, _ = coherence_matrices.shape
    eyeN = jnp.eye(N, dtype=coherence_matrices.dtype)
    eyeNb = jnp.broadcast_to(eyeN, coherence_matrices.shape)

    abs_G = jnp.abs(coherence_matrices)

    # Detect obviously singular blocks (your toy Γ=0 case)
    block_max = jnp.max(abs_G, axis=(-2, -1), keepdims=True)
    is_zero_block = block_max < zero_tol  # (..., 1, 1)

    # Keep the solve from crashing: replace zero-blocks by I *for the solve only*
    abs_G_safe = abs_G + gamma_jitter * eyeNb
    abs_G_safe = jnp.where(is_zero_block, eyeNb, abs_G_safe)

    abs_G_inv = solve(abs_G_safe, eyeNb, assume_a="pos")

    # Build X once and do the inverse-free CRLB from X
    X = _build_fisher_from_abs_gamma(abs_G, abs_G_inv, num_looks)
    sig = _crlb_from_x(X, reference_idx, aps_variance, fim_jitter)

    if mask_zero_blocks:
        # overwrite sigma on zero blocks to NaN to mimic NumPy error/NaN
        mask = jnp.squeeze(is_zero_block, axis=(-2, -1))
        nanv = jnp.full(sig.shape, jnp.nan, dtype=sig.dtype)
        sig = jnp.where(mask, nanv, sig)
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
