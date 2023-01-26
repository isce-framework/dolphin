"""Module for computing quality metrics of estimated solutions."""
import functools

from dolphin.utils import get_array_module


def estimate_temp_coh(est, C_arrays):
    """Estimate the temporal coherence for a block of solutions.

    Parameters
    ----------
    est : np.ndarray or cupy.ndarray
        The estimated phase from, e.g., [dolphin.phase_link.run_mle][]
        shape = (nslc, rows, cols).
        If est.shape = (nslc,) (a single pixel), will be reshaped to (nslc, 1, 1)
    C_arrays : np.ndarray or cupy.ndarray, shape = (rows, cols, nslc, nslc)
        The sample covariance matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance_cpu][]).
        If one covariance matrix is passed (C_arrays.shape = (nslc, nslc)),
        will be reshaped to (1, 1, nslc, nslc)

    Returns
    -------
    np.ndarray or cupy.ndarray
        The temporal coherence of the time series compared to cov_matrix.
        Output shape is (rows, cols)
    """
    if est.ndim == 1:
        est = est.reshape(-1, 1, 1)
    if C_arrays.ndim == 2:
        C_arrays = C_arrays.reshape(1, 1, *C_arrays.shape)

    xp = get_array_module(C_arrays)
    # Move to match the SLC dimension at the end for the covariances
    est_arrays = xp.moveaxis(est, 0, -1)
    # Get only the phase of the covariance (not correlation/magnitude)
    C_angles = xp.exp(1j * xp.angle(C_arrays))

    est_phase_diffs = xp.einsum("jka, jkb->jkab", est_arrays, est_arrays.conj())
    # shape will be (rows, cols, nslc, nslc)
    differences = C_angles * est_phase_diffs.conj()

    # # Get just the upper triangle of the differences (not the diagonal)
    nslc = C_angles.shape[-1]
    rows, cols = _get_upper_tri_idxs(nslc, xp)
    upper_diffs = differences[:, :, rows, cols]
    # get number of non-nan values
    count = xp.count_nonzero(~xp.isnan(upper_diffs), axis=-1)
    return xp.abs(xp.nansum(upper_diffs, axis=-1)) / count


@functools.lru_cache(maxsize=32)
def _get_upper_tri_idxs(nslc, xp):
    """Get the upper triangle (not including the diagonal) of a matrix.

    Caching is used to avoid re-computing the indices for the same nslc,
    which is useful when running on the GPU.
    """
    rows, cols = xp.triu_indices(nslc, k=1)
    return rows, cols
