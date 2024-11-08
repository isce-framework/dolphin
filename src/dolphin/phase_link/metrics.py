"""Module for computing quality metrics of estimated solutions."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike


@jit
def estimate_temp_coh(cpx_phase: ArrayLike, C_arrays: ArrayLike) -> Array:
    """Estimate the temporal coherence for a block of solutions.

    Parameters
    ----------
    cpx_phase : ArrayLike
        Complex valued phase linking results from [dolphin.phase_link.run_cpl][]
        shape = (nslc, rows, cols).
        If cpx_phase.shape = (nslc,) (a single pixel), will be reshaped to (nslc, 1, 1)
    C_arrays : ArrayLike, shape = (rows, cols, nslc, nslc)
        The sample covariance matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance][]).
        If one covariance matrix is passed (C_arrays.shape = (nslc, nslc)),
        will be reshaped to (1, 1, nslc, nslc)

    Returns
    -------
    jax.Array
        The temporal coherence of the time series compared to cov_matrix.
        Output shape is (rows, cols)

    """
    if cpx_phase.ndim == 1:
        cpx_phase = cpx_phase.reshape(1, 1, -1)
    if C_arrays.ndim == 2:
        C_arrays = C_arrays.reshape(1, 1, *C_arrays.shape)

    _temp_coh_3d = vmap(estimate_temp_coh_single)
    estimate_temp_coh = vmap(_temp_coh_3d)
    return estimate_temp_coh(cpx_phase, C_arrays)


@jit
def estimate_temp_coh_single(cpx_phase: ArrayLike, C: ArrayLike) -> Array:
    """Estimate the temporal coherence for one covariance matrix/phase solution.

    Parameters
    ----------
    cpx_phase : ArrayLike
        1D-Complex valued phase linking results from [dolphin.phase_link.run_cpl][]
    C : ArrayLike, shape = (nslc, nslc)
        The sample covariance matrix at one pixel.

    Returns
    -------
    jax.Array
        The temporal coherence of the time series compared to cov_matrix.
        Output shape is ()

    """
    # For original Squeesar temp coh, everything is equally weighted
    W = jnp.ones(C.shape, dtype="float32")
    return _general_temp_coh_single(cpx_phase=cpx_phase, C=C, W=W)


@jit
def estimate_weighted_temp_coh_single(cpx_phase: ArrayLike, C: ArrayLike) -> Array:
    """Estimate the weighted temporal coherence for one pixel."""
    # Weight the differences by pass in weights coherence magnitudes
    W = jnp.abs(C)
    return _general_temp_coh_single(cpx_phase=cpx_phase, C=C, W=W)


@jit
def _general_temp_coh_single(cpx_phase: ArrayLike, C: ArrayLike, W: ArrayLike) -> Array:
    """Estimate the (weighted) temporal coherence for a single solution."""
    # Make outer product of complex phase (reform all possible ifgs from the solution)
    reformed_ifg_phases = jnp.angle(cpx_phase[:, None] @ cpx_phase[None, :].conj())

    C_angles = jnp.exp(1j * jnp.angle(C))
    differences = C_angles * jnp.exp(-1j * reformed_ifg_phases)
    # Weight the differences by the weights passed in
    differences *= W

    nslc, _ = C.shape
    # Only add up the upper triangle of the hermitian matrix
    rows, cols = jnp.triu_indices(nslc, k=1)
    upper_diffs = differences[rows, cols]
    upper_weights = W[rows, cols]

    # Get the total weight used for the divisor
    diff_is_nan = jnp.isnan(upper_diffs)
    total_diffs = jnp.sum(jnp.where(diff_is_nan, 0, upper_diffs))
    total_weights = jnp.sum(jnp.where(diff_is_nan, 0, upper_weights))
    out = jnp.abs(total_diffs / total_weights)
    # Protect against all-zero `total_weights`
    return jnp.nan_to_num(out, nan=0, posinf=0, neginf=0)
