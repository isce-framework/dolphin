"""Module for computing quality metrics of estimated solutions."""
import jax.numpy as jnp
from jax import Array, jit
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
        cpx_phase = cpx_phase.reshape(-1, 1, 1)
    if C_arrays.ndim == 2:
        C_arrays = C_arrays.reshape(1, 1, *C_arrays.shape)

    # Move to match the SLC dimension at the end for the covariances
    est_arrays = jnp.moveaxis(cpx_phase, 0, -1)
    # Get only the phase of the covariance (not correlation/magnitude)
    C_angles = jnp.exp(1j * jnp.angle(C_arrays))

    est_phase_diffs = jnp.einsum("jka, jkb->jkab", est_arrays, est_arrays.conj())
    # shape will be (rows, cols, nslc, nslc)
    differences = C_angles * est_phase_diffs.conj()

    # # Get just the upper triangle of the differences (not the diagonal)
    nslc = C_angles.shape[-1]
    # Get the upper triangle (not including the diagonal) of a matrix.
    rows, cols = jnp.triu_indices(nslc, k=1)
    upper_diffs = differences[:, :, rows, cols]
    # get number of non-nan values
    count = jnp.count_nonzero(~jnp.isnan(upper_diffs), axis=-1)
    return jnp.abs(jnp.nansum(upper_diffs, axis=-1)) / count


from jax import vmap


@jit
def estimate_temp_coh_single(cpx_phase: ArrayLike, C: ArrayLike) -> Array:
    """Estimate the temporal coherence for a single solution."""
    reformed_ifg_phases = jnp.angle(cpx_phase[:, None] @ cpx_phase[None, :].conj())
    C_angles = jnp.exp(1j * jnp.angle(C))
    differences = C_angles * jnp.exp(-1j * reformed_ifg_phases)

    nslc, _ = C.shape
    rows, cols = jnp.triu_indices(nslc, k=1)
    upper_diffs = differences[rows, cols]
    # get number of non-nan values
    count = jnp.count_nonzero(~jnp.isnan(upper_diffs))
    return jnp.abs(jnp.nansum(upper_diffs)) / count


temp_coh2 = vmap(estimate_temp_coh_single)
temp_coh_stack = vmap(temp_coh2)


def estimate_weighted_temp_coh_single(cpx_phase: ArrayLike, C: ArrayLike) -> Array:
    """Estimate the weighted temporal coherence for a single solution."""
    reformed_ifg_phases = jnp.angle(cpx_phase[:, None] @ cpx_phase[None, :].conj())
    C_angles = jnp.exp(1j * jnp.angle(C))
    differences = C_angles * jnp.exp(-1j * reformed_ifg_phases)
    # Weight the differences by the coherence magnitudes
    W = jnp.abs(C)
    differences *= W

    nslc, _ = C.shape
    rows, cols = jnp.triu_indices(nslc, k=1)
    upper_diffs = differences[rows, cols]
    upper_weights = W[rows, cols]

    # Get the total weight used for the divisor
    isnan_mask = ~jnp.isnan(upper_diffs)
    total_weights_used = jnp.nansum(upper_weights[isnan_mask])
    total_diff = jnp.nansum(upper_diffs[isnan_mask])
    return jnp.abs(total_diff / total_weights_used)
