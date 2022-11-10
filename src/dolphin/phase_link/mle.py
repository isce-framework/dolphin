import logging

import numpy as np
import numpy.linalg as la
from numba import njit

from dolphin.utils import take_looks

logger = logging.getLogger(__name__)


# TODO: make a version with same API as GPU (multithreaded)


@njit(cache=True)
def mle(cov_mat, beta=0.0):
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
        Gamma_inv = la.inv(_reg_beta(Gamma, beta)).astype(dtype)
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


@njit(cache=True)
def coh_mat(neighbor_stack, cov_mat=None):
    """Given a (n_slc, n_samps) samples, estimate the coherence matrix."""
    nslc = neighbor_stack.shape[0]
    if cov_mat is None:
        cov_mat = np.zeros((nslc, nslc), dtype=np.complex64)
    for ti in range(nslc):
        for tj in range(ti + 1, nslc):
            cov = _covariance(neighbor_stack[ti, :], neighbor_stack[tj, :])
            cov_mat[ti, tj] = cov
            cov_mat[tj, ti] = np.conjugate(cov)
        cov_mat[ti, ti] = 1.0

    return cov_mat


@njit(cache=True)
def _covariance(c1, c2):
    a1 = np.nansum(np.abs(c1) ** 2)
    a2 = np.nansum(np.abs(c2) ** 2)

    cov = np.nansum(c1 * np.conjugate(c2)) / (np.sqrt(a1) * np.sqrt(a2))
    return cov


def regularize_C(C, how="beta", beta=0.1, alpha=1e-3):
    """Regularize the sample covariance matrix.

    Parameters
    ----------
    C : np.array
        The sample covariance matrix
    how : str, optional
        The regularization method, by default "beta"
    beta : float, optional
        The regularization parameter for `how='beta'` , by default 0.1
    alpha : float, optional
        The regularization parameter for `how='eye'`, by default 1e-3

    Returns
    -------
    np.array
        The regularized covariance matrix
    """
    # Regularization
    if how == "eye":
        return _reg_eye(C, alpha)
    elif how == "beta":
        return _reg_beta(C, beta)
    else:
        raise ValueError(f"Unknown regularization method: {how}")


@njit(cache=True)
def _reg_eye(C, alpha=1e-3):
    return (C + alpha * np.eye(C.shape[0])).astype(C.dtype)


@njit(cache=True)
def _reg_beta(C, beta=1e-1):
    return (1 - beta) * C + beta * np.eye(C.shape[0], dtype=C.dtype)


def full_cov_multilooked(slcs, looks):
    """Estimate the full covariance matrix for each pixel.

    Parameters
    ----------
    slcs : np.array
        The stack of complex SLC data, shape (nslc, rows, cols)
    looks : np.array
        The number of looks as (row looks, col_looks)

    Returns
    -------
    C: np.array
        The full covariance matrix for each pixel, shape (rows, cols, nslc, nslc)
        i.e. C[i, j] is the covariance matrix for pixel (i, j)
    """
    try:
        import cupy as cp

        xp = cp.get_array_module(slcs)
    except ImportError:
        logger.debug("No GPU available or cupy not installed. Using numpy")
        xp = np

    # Perform an outer product of the slcs and their conjugate, then multilook
    numer = take_looks(
        xp.einsum("ajk, bjk-> abjk", slcs, slcs.conj(), optimize=True), *looks
    )

    # Do the same for the powers
    s_pow_looked = take_looks(xp.abs(slcs) ** 2, *looks)
    denom = xp.einsum("ajk, bjk-> abjk", s_pow_looked, s_pow_looked, optimize=True)

    C = numer / xp.sqrt(denom)
    return xp.transpose(C, (2, 3, 0, 1))


def mle_stack(C_arrays: np.ndarray, beta: float = 0.0, reference_idx: float = 0):
    """Estimate the linked phase for a stack of covariance matrices.

    Will use cupy if available, (and if the input is a cupy array).
    Otherwise, falls back to numpy.

    Parameters
    ----------
    C_arrays : np.array, shape = (rows, cols, nslc, nslc)
        The sample covariance matrix at each pixel (e.g. from `full_cov_multilooked`)
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        If the SLC stack from which `C_arrays` was computed contained
        compressed SLCs at the stack, then this should be the index
        of the first non-compressed SLC.

    Returns
    -------
    np.array, shape = (nslc, rows, cols)
        The estimated linked phase, same shape as the input slcs (possibly multilooked)
    """
    try:
        import cupy as cp

        xp = cp.get_array_module(C_arrays)
    except ImportError:
        logger.debug("cupy not installed, falling back to numpy")
        xp = np
    # estimate the wrapped phase based on the EMI paper
    # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix
    Gamma = xp.abs(C_arrays)

    if beta > 0:
        # Perform regularization
        Id = xp.eye(Gamma.shape[-1], dtype=Gamma.dtype)
        # repeat the identity matrix for each pixel
        Id = xp.tile(Id, (Gamma.shape[0], Gamma.shape[1], 1, 1))
        Gamma = (1 - beta) * Gamma + beta * Id

    Gamma_inv = xp.linalg.inv(Gamma)
    _, V = xp.linalg.eigh(Gamma_inv * C_arrays)

    # The shape of V is (rows, cols, nslc, nslc)
    # at pixel (r, c), the columns of V[r, c] are the eigenvectors.
    # They're ordered by increasing eigenvalue, so the first column is the
    # eigenvector corresponding to the smallest eigenvalue (our phase solution).
    evd_estimate = V[:, :, :, 0]
    # The phase estimate on the reference day will be size (rows, cols)
    ref = evd_estimate[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = evd_estimate * xp.conjugate(ref[:, :, None])

    # Return the phase (still as a GPU array)
    phase_stack = xp.angle(evd_estimate)
    # Move the SLC dimension to the front (to match the SLC stack shape)
    return np.moveaxis(phase_stack, -1, 0)


def compress(
    slc_stack: np.ndarray,
    mle_estimate: np.ndarray,
):
    """Compress the stack of SLC data using the estimated phase.

    Parameters
    ----------
    slc_stack : np.array
        The stack of complex SLC data, shape (nslc, rows, cols)
    mle_estimate : np.array
        The estimated phase from `run_mle_gpu`, shape (nslc, rows, cols)

    Returns
    -------
    np.array
        The compressed SLC data, shape (rows, cols)
    """
    try:
        import cupy as cp

        xp = cp.get_array_module(slc_stack)
    except ImportError:
        logger.debug("cupy not installed, falling back to numpy")
        xp = np
    # For each pixel, project the SLCs onto the estimated phase
    # by performing a pixel-wise complex dot product
    return xp.nanmean(slc_stack * xp.conjugate(mle_estimate), axis=0)
