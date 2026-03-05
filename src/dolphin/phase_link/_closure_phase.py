import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def compute_nearest_closure_phases(
    cov_matrix: Array,
) -> Array:
    """Compute the nearest-neighbor closure phases for a single covariance matrix."""
    # Extract the diagonals we need
    # First super-diagonal: Used for (i, i+1), then (i+1, i+2)
    diag_1 = jnp.diag(cov_matrix, k=1)  # length N-1
    # Second super-diagonal (i, i+2). Used for the bandwidth-2 interferograms
    diag_2 = jnp.diag(cov_matrix, k=2)  # length N-2

    # Compute closure phases as complex numbers, then take the angle
    closure_complex = diag_1[:-1] * diag_1[1:] * jnp.conj(diag_2)
    return jnp.angle(closure_complex)


# Vectorized version for multiple covariance matrices (e.g., different pixels)
@jax.jit
def compute_nearest_closure_phases_batch(
    cov_matrices: Array,
) -> Array:
    """Compute nearest-neighbor closure phases for a batch of covariance matrices.

    Parameters
    ----------
    cov_matrices : Array
        Complex (..., R, C, N, N) array of M covariance matrices

    Returns
    -------
    Array
        Closure phases: (..., R, C, N-2) array of closure phases

    """
    return jax.vmap(jax.vmap(compute_nearest_closure_phases))(cov_matrices)
