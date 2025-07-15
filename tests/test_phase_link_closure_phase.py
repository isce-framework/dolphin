import jax
import jax.numpy as jnp
import pytest

from dolphin.phase_link._closure_phase import (
    compute_nearest_closure_phases,
    compute_nearest_closure_phases_batch,
)


@pytest.mark.parametrize("n", [4, 10, 23])
def test_output_shape(n):
    """Output must be (n-2,) for a single covariance matrix."""
    C = jnp.eye(n, dtype=jnp.complex64)
    out = compute_nearest_closure_phases(C)
    assert out.shape == (n - 2,)


def test_rank1_outer_product_closure_is_zero():
    """For C = v v^H (rank-1), every closure phase is identically zero."""
    n = 12
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (n,)) + 1j * jax.random.normal(key, (n,))
    C = jnp.outer(v, jnp.conj(v))  # v v^H, Hermitian rank-1

    phi = compute_nearest_closure_phases(C)
    assert jnp.allclose(phi, 0.0, atol=1e-6)


def test_random_hermitian_manual_check():
    """Compare against a direct NumPy/JAX implementation for one matrix."""
    n = 7
    key = jax.random.PRNGKey(123)
    A = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    C = A + A.T.conj()  # make it Hermitian

    phi = compute_nearest_closure_phases(C)

    # Manual reference computation
    d1 = jnp.diag(C, k=1)  # C_{i,i+1}
    d2 = jnp.diag(C, k=2)  # C_{i,i+2}
    expected = jnp.angle(d1[:-1] * d1[1:] * jnp.conj(d2))

    assert jnp.allclose(phi, expected, atol=1e-6)


def test_batch_vectorization():
    """Sanity-check batched call and that [0,0] matches single-matrix path."""
    r, c, n = 3, 4, 9
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (r, c, n, n)) + 1j * jax.random.normal(key, (r, c, n, n))
    C = A + jnp.swapaxes(A, -2, -1).conj()  # Hermitian (r,c,n,n)

    batch_phi = compute_nearest_closure_phases_batch(C)
    assert batch_phi.shape == (r, c, n - 2)

    # spot-check one element against the scalar implementation
    assert jnp.allclose(
        batch_phi[0, 0],
        compute_nearest_closure_phases(C[0, 0]),
        atol=1e-6,
    )
