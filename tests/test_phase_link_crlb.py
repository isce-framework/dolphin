# tests/test_crlb.py
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dolphin.phase_link.covariance import coh_mat_single
from dolphin.phase_link.crlb import (
    _examples,
    compute_crlb,
    compute_crlb_jax,
    compute_lower_bound_std,
)
from dolphin.phase_link.simulate import simulate_coh, simulate_neighborhood_stack


def _get_example(name: str, N: int = 10) -> np.ndarray:
    C_ar1, C_const = _examples(N=N)
    mapping = {"ar1": C_ar1, "const": C_const}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown example '{name}'.") from exc


@pytest.mark.parametrize("example", ["ar1", "const"])
def test_crlb_and_std_consistency(example: str) -> None:
    N = 8
    C = _get_example(example, N)
    num_looks = 7
    aps_var = 0.0

    # Full FIM-inverse (Σ) and std-dev vector
    sigma = compute_crlb(C, num_looks, aps_var)
    std_vec = compute_lower_bound_std(C, num_looks, aps_var)

    # Shape checks
    assert sigma.shape == (N - 1, N - 1)
    assert std_vec.shape == (N,)

    # Consistency:  std_vec[1:] == sqrt(diag(sigma)); std_vec[0] is the reference epoch
    np.testing.assert_allclose(
        std_vec[1:], np.sqrt(np.diag(sigma)), rtol=1e-8, atol=1e-10
    )
    assert std_vec[0] == 0.0


@pytest.mark.parametrize("example", ["ar1", "const"])
@pytest.mark.parametrize("aps_var", [0.0, 1e-2])
def test_numpy_vs_jax(example: str, aps_var: float) -> None:
    """`compute_crlb_jax` should match `compute_lower_bound_std` (numpy)"""
    N = 9
    C = _get_example(example, N)
    num_looks = 10
    ref_idx = 0

    # NumPy “truth”
    std_np = compute_lower_bound_std(C, num_looks, aps_var)

    # JAX implementation
    std_jax = np.asarray(
        compute_crlb_jax(
            jnp.asarray(C),
            num_looks=num_looks,
            reference_idx=ref_idx,
            aps_variance=aps_var,
            gamma_jitter=0.0,  # disable extra regularization for fair comparison
            fim_jitter=0.0,
        )
    )

    np.testing.assert_allclose(std_np, std_jax, rtol=5e-4, atol=1e-6)


def test_large_n_no_nans() -> None:
    """Test CRLB computation with large N (~100) and verify no NaNs are produced."""
    N = 100
    C = _get_example("ar1", N)
    num_looks = 10
    aps_var = 1e-3

    # Test NumPy implementation
    std_np = compute_lower_bound_std(C, num_looks, aps_var)
    assert not np.any(np.isnan(std_np)), "NumPy implementation produced NaNs"
    assert std_np.shape == (N,), f"Expected shape ({N},), got {std_np.shape}"
    assert std_np[0] == 0.0, "Reference epoch should have zero std"
    assert np.all(std_np[1:] > 0), "All non-reference epochs should have positive std"

    # Test JAX implementation
    std_jax = np.asarray(
        compute_crlb_jax(
            jnp.asarray(C),
            num_looks=num_looks,
            reference_idx=0,
            aps_variance=aps_var,
            gamma_jitter=1e-6,  # Small jitter for numerical stability
            fim_jitter=1e-6,
        )
    )
    assert not np.any(np.isnan(std_jax)), "JAX implementation produced NaNs"
    assert std_jax.shape == (N,), f"Expected shape ({N},), got {std_jax.shape}"
    assert std_jax[0] == 0.0, "Reference epoch should have zero std"
    assert np.all(std_jax[1:] > 0), "All non-reference epochs should have positive std"

    # Verify they are reasonably close
    np.testing.assert_allclose(std_np, std_jax, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("num_acq", [10, 25])
@pytest.mark.parametrize("neighbor_samples", [100, 500])
def test_crlb_with_empirical_coherence(num_acq: int, neighbor_samples: int) -> None:
    """Test CRLB computation using empirical coherence matrices from simulations.

    This tests different properties than theoretical matrices, especially
    with regard to positive definiteness and numerical stability.
    """
    np.random.seed(42)  # For reproducible results

    # Generate a theoretical coherence matrix using simulate_coh
    C_true, _ = simulate_coh(
        num_acq=num_acq,
        gamma_inf=0.2,
        gamma0=0.9,
        Tau0=100,
        acq_interval=12,
        add_signal=False,
    )

    # Generate empirical samples using simulate_neighborhood_stack
    samples = simulate_neighborhood_stack(C_true, neighbor_samples=neighbor_samples)

    # Compute empirical coherence matrix
    C_empirical = coh_mat_single(samples)

    # Test parameters
    num_looks = int(np.sqrt(neighbor_samples))  # Effective looks
    aps_var = 1e-3
    ref_idx = 0

    # Test that CRLB computation doesn't fail with empirical matrix
    std_np = compute_lower_bound_std(C_empirical, num_looks, aps_var)
    std_jax = np.asarray(
        compute_crlb_jax(
            jnp.asarray(C_empirical),
            num_looks=num_looks,
            reference_idx=ref_idx,
            aps_variance=aps_var,
            gamma_jitter=1e-5,
            fim_jitter=1e-5,
        )
    )

    # Basic sanity checks
    assert not np.any(
        np.isnan(std_np)
    ), "NumPy implementation produced NaNs with empirical matrix"
    assert not np.any(
        np.isnan(std_jax)
    ), "JAX implementation produced NaNs with empirical matrix"
    assert std_np.shape == (
        num_acq,
    ), f"Expected shape ({num_acq},), got {std_np.shape}"
    assert std_jax.shape == (
        num_acq,
    ), f"Expected shape ({num_acq},), got {std_jax.shape}"
    assert std_np[0] == 0.0, "Reference epoch should have zero std"
    assert std_jax[0] == 0.0, "Reference epoch should have zero std"
    assert np.all(
        std_np[1:] > 0
    ), "All non-reference epochs should have positive std (NumPy)"
    assert np.all(
        std_jax[1:] > 0
    ), "All non-reference epochs should have positive std (JAX)"

    # The empirical and theoretical should be reasonably close, but differ
    # due to sampling variation and potential positive definiteness issues
    np.testing.assert_allclose(std_np, std_jax, rtol=1e-2, atol=1e-4)

    # Test that empirical results are reasonable compared to theoretical
    std_theoretical = compute_lower_bound_std(C_true, num_looks, aps_var)

    # Empirical should be in the same ballpark as theoretical, but may differ
    # due to sampling effects and different matrix properties
    assert np.all(std_np[1:]) > 0
    assert np.all(std_theoretical[1:]) > 0
    # Allow some tolerance since empirical can vary substantially from theoretical
    assert 0.1 < np.median(std_np[1:]) / np.median(std_theoretical[1:]) < 10.0
