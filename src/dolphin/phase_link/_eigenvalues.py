"""Eigenvalue solvers specialized for the phase linking use case.

|                 | EVD               |         | EMI               |         |
| --------------- | ----------------- | ------- | ----------------- | ------- |
|                 | Runtime (seconds) | Speedup | Runtime (seconds) | Speedup |
| `scipy.eigh`    | 185               | -       | 398               | -       |
| Power iteration | 19                | 9.7     | 57                | 6.8     |

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.lax import while_loop
from jax.typing import ArrayLike

# For both largest and smallest eig, we map over the first two dimensions
# so now instead of one scalar eigenvalue, we have (rows, cols) eigenvalues


@partial(jit, static_argnames="mu")
def eigh_smallest_stack(
    C_arrays: ArrayLike,
    mu: float,
) -> tuple[Array, Array]:
    """Get the smallest (eigenvalue, eigenvector) for each pixel in a 3D stack.

    Uses shift inverse iteration to find the eigenvalue closest to `mu`.
    Pick `mu` to be slightly below the smallest eigenvalue for fastest convergence.

    Returns real eigenvalues, assuming the arrays are Hermitian.

    Parameters
    ----------
    C_arrays : ArrayLike
        The stack of coherence matrices.
        Shape = (rows, cols, nslc, nslc)
    mu : float
        The value to use for the shift inverse iteration.
        The eigenvalue closest to this value is returned.
    v0 : ArrayLike, optional
        The initial guess for the eigenvector.
        If None, a vector of 1s is used.
        Shape = (rows, cols, nslc)

    Returns
    -------
    eigenvalues : Array
        The smallest eigenvalue for each pixel's matrix
        Shape = (rows, cols)
    eigenvectors : Array
        The normalized eigenvector corresponding to the smallest eigenvalue
        Shape = (rows, cols, nslc)

    """
    in_axes = (0, None)
    eig_vals, eig_vecs = vmap(
        vmap(inverse_iteration, in_axes=in_axes), in_axes=in_axes
    )(C_arrays, mu)
    return eig_vals.real, eig_vecs


@jit
def eigh_largest_stack(C_arrays: ArrayLike) -> tuple[Array, Array]:
    """Get the largest (eigenvalue, eigenvector) for each pixel in a 3D stack.

    Returns real eigenvalues, assuming the arrays are Hermitian.

    Parameters
    ----------
    C_arrays : ArrayLike
        The stack of coherence matrices.
        Shape = (rows, cols, nslc, nslc)

    Returns
    -------
    eigenvalues : Array
        The largest eigenvalue for each pixel's matrix
        Shape = (rows, cols)
    eigenvectors : Array
        The normalized eigenvector corresponding to the largest eigenvalue
        Shape = (rows, cols, nslc)

    """
    eig_vals, eig_vecs = vmap(vmap(power_iteration))(C_arrays)
    return eig_vals.real, eig_vecs


@partial(jit, static_argnames=("tol",))
def power_iteration(
    A: jnp.array, tol: float = 1e-5, max_iters: int = 50
) -> tuple[jnp.array, jnp.array]:
    """Compute the dominant eigenpair of a matrix using power iteration.

    Parameters
    ----------
    A : jnp.array
        The input matrix.
    tol : float, optional
        The tolerance for convergence (default is 1e-3).
    max_iters : int, optional
        The maximum number of iterations (default is 50).

    Returns
    -------
    eigenvalue : jnp.array
        The dominant eigenvalue of the matrix.
    vk : jnp.array
        The corresponding eigenvector of the dominant eigenvalue.

    """
    n = A.shape[-1]
    vk = jnp.ones(n, dtype=A.dtype)
    vk = vk / jnp.linalg.norm(vk)

    def body_fun(val):
        vk, _, idx = val
        A_times_vk = A @ vk
        next_vk = A_times_vk / jnp.linalg.norm(A_times_vk)
        diff = jnp.linalg.norm(next_vk - vk)
        return next_vk, diff, idx + 1

    def cond_fun(val):
        _, diff, iters = val
        return jnp.logical_and(diff > tol, iters < max_iters)

    init_val = (vk, 1, 1)
    vk, _, _end_iters = while_loop(cond_fun, body_fun, init_val)

    # vk is normalized to 1, so no need to divide by (vk.T @ vk)
    eigenvalue = vk.conj() @ A @ vk
    return eigenvalue, vk


@partial(jit, static_argnames=("mu", "tol", "max_iters"))
def inverse_iteration(
    A: ArrayLike,
    mu: float,
    tol: float = 1e-5,
    max_iters: int = 50,
) -> tuple[Array, Array]:
    """Compute the eigenvalue of the positive definite matrix `A` closest to `mu`.

    Inverse iteration (or inverse power iteration) is an iterative method used to
    find the eigenvalue of a matrix A that is closest to a given scalar mu.

    Parameters
    ----------
    A : ArrayLike
        Square matrix of which we seek an eigenvalue and eigenvector.
    mu : float
        The shift value, around which we seek the closest eigenvalue of A.
    tol : float, optional
        Tolerance for convergence of the method. The default is 1e-5.
    max_iters : int, optional
        Maximum number of iterations to perform. The default is 50.

    Returns
    -------
    eigenvalue : jnp.array
        The eigenvalue of the matrix closest to `mu`.
    vk : jnp.array
        The corresponding eigenvector.

    Notes
    -----
    This method may not converge with a poor guess of `mu`.
    The close the guess of `mu` is to your eigenvalue, the quicker the convergence.
    However, picking the exact eigenvalue may lead to numerical instability.

    References
    ----------
    [1] https://services.math.duke.edu/~jtwong/math361-2019/lectures/Lec10eigenvalues.pdf

    """
    n = A.shape[0]
    vk = jnp.ones(n, dtype=A.dtype)

    vk = vk / jnp.linalg.norm(vk)
    Id = jnp.eye(A.shape[0], dtype=A.dtype)
    # Prefactor A - mu I to quickly solve each iteration
    lu_and_pivots = jax.scipy.linalg.lu_factor(A - mu * Id)

    def body_fun(val):
        vk_cur, _, iters = val
        # Perform power iteration on (A - mu I)^{-1} ,
        # which is the same as solving (A - mu I)x = v_k
        vk_new = jax.scipy.linalg.lu_solve(lu_and_pivots, vk_cur)
        vk_new = vk_new / jnp.linalg.norm(vk_new)
        diff = jnp.linalg.norm(vk_new - vk_cur)
        return vk_new, diff, iters + 1

    def cond_fun(val):
        _, diff, iters = val
        return jnp.logical_and(diff > tol, iters < max_iters)

    init_val = (vk, 1.0, 1)
    vk_sol, _, _end_iters = while_loop(cond_fun, body_fun, init_val)

    eigenvalue = vk_sol.conj() @ A @ vk_sol
    return eigenvalue, vk_sol
