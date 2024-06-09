from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import while_loop


@partial(jit, static_argnames=("tol",))
def power_iteration(
    A: jnp.array, tol: float = 1e-3, max_iters: int = 50
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
    vk, _, end_iters = while_loop(cond_fun, body_fun, init_val)

    # debug
    # jax.debug.print("power_iteration: end_iters: {}", end_iters)

    eigenvalue = vk @ A @ vk
    return eigenvalue, vk


@partial(jit, static_argnames=("tol", "max_iters"))
def shifted_inverse_iteration(
    A: jnp.array, mu: float, tol: float = 1e-3, max_iters: int = 50
) -> tuple[jnp.array, jnp.array]:
    """Compute the eigenvalue of A closest to mu."""
    n = A.shape[0]
    vk = jnp.ones(n, dtype=A.dtype)
    vk = vk / jnp.linalg.norm(vk)
    Id = jnp.eye(A.shape[0], dtype=A.dtype)
    cho_fact = jax.scipy.linalg.cho_factor(A - mu * Id)

    def body_fun(val):
        vk_cur, _, iters = val
        vk_new = jax.scipy.linalg.cho_solve(cho_fact, vk_cur)
        vk_new = vk_new / jnp.linalg.norm(vk_new)
        diff = jnp.linalg.norm(vk_new - vk_cur)
        return vk_new, diff, iters + 1

    def cond_fun(val):
        _, diff, iters = val
        return jnp.logical_and(diff > tol, iters < max_iters)

    init_val = (vk, 1.0, 1)
    vk_sol, _, end_iters = while_loop(cond_fun, body_fun, init_val)

    eigenvalue = (vk_sol.conj() @ A @ vk_sol).real
    return eigenvalue, vk_sol
