from __future__ import annotations

import logging
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from jax.scipy.linalg import cho_factor, cho_solve, eigh
from jax.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import take_looks

from . import covariance, metrics
from ._ps_filling import fill_ps_pixels

logger = logging.getLogger(__name__)


DEFAULT_STRIDES = Strides(1, 1)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""


class EstimatorType(IntEnum):
    """Type of estimator used for phase linking."""

    EVD = 0
    EMI = 1


class PhaseLinkOutput(NamedTuple):
    """Output of the MLE solver."""

    cpx_phase: np.ndarray
    """Estimated linked phase."""

    temp_coh: np.ndarray
    """Temporal coherence of the optimization.
    A goodness of fit parameter from 0 to 1 at each pixel.
    """

    eigenvalues: np.ndarray
    """The smallest (largest) eigenvalue resulting from EMI (EVD)."""

    estimator: np.ndarray  # dtype: np.int8
    """The estimator type used for phase linking at each pixel."""

    avg_coh: np.ndarray | None = None
    """Average coherence across dates for each SLC."""


def run_phase_linking(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    use_evd: bool = False,
    beta: float = 0.0,
    reference_idx: int = 0,
    nodata_mask: np.ndarray = None,
    ps_mask: Optional[np.ndarray] = None,
    neighbor_arrays: Optional[np.ndarray] = None,
    avg_mag: Optional[np.ndarray] = None,
    use_slc_amp: bool = True,
    calc_average_coh: bool = False,
) -> PhaseLinkOutput:
    """Estimate the linked phase for a stack of SLCs.

    If passing a `ps_mask`, will combine the PS phases with the
    estimated DS phases.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    nodata_mask : np.ndarray, optional
        A mask of bad/nodata pixels to ignore when estimating the covariance.
        Pixels with `True` (or 1) are ignored, by default None
        If None, all pixels are used, by default None.
    ps_mask : np.ndarray, optional
        A mask of pixels marking persistent scatterers (PS) to
        skip when multilooking.
        Pixels with `True` (or 1) are PS and will be ignored
        (combined with `nodata_mask`).
        The phase from these pixels will be inserted back
        into the final estimate directly from `slc_stack`.
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
        If None, the average magnitude will be computed from `slc_stack`.
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.
    calc_average_coh : bool, optional, default = False
        Whether to calculate the average coherence for each SLC date.

    Returns
    -------
    PhaseLinkOutput:
        A Named tuple with the following fields
    linked_phase : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_images, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    eigenvalues : np.ndarray[np.float32]
        The smallest (largest) eigenvalue resulting from EMI (EVD).
    `avg_coh` : np.ndarray[np.float32]
        (only If `calc_average_coh` is True) the average coherence for each SLC date

    """
    _, rows, cols = slc_stack.shape
    # Common pre-processing for both CPU and GPU versions:

    # Mask nodata pixels if given
    if nodata_mask is None:
        nodata_mask = np.zeros((rows, cols), dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool)

    # Track the PS pixels, if given, and remove them from the stack
    # This will prevent the large amplitude PS pixels from dominating
    # the covariance estimation.
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    _raise_if_all_nan(slc_stack)

    # Make sure we also are ignoring pixels which are nans for all SLCs
    if nodata_mask.shape != (rows, cols) or ps_mask.shape != (rows, cols):
        msg = (
            f"nodata_mask.shape={nodata_mask.shape}, ps_mask.shape={ps_mask.shape},"
            f" but != SLC (rows, cols) {rows, cols}"
        )
        raise ValueError(msg)
    # for any area that has nans in the SLC stack, mark it as nodata
    nodata_mask |= np.any(np.isnan(slc_stack), axis=0)
    # Make sure the PS mask didn't have extra burst borders that are nodata here
    ps_mask[nodata_mask] = False

    # TODO: Any other masks we need?
    ignore_mask = np.logical_or.reduce((nodata_mask, ps_mask))

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, ignore_mask] = np.nan

    cpl_out = run_cpl(
        slc_stack=slc_stack_masked,
        half_window=half_window,
        strides=strides,
        use_evd=use_evd,
        beta=beta,
        reference_idx=reference_idx,
        neighbor_arrays=neighbor_arrays,
        calc_average_coh=calc_average_coh,
    )

    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        # we need to match `io.compute_out_shape` here
        slcs_decimated = decimate(slc_stack, strides)
        cpx_phase = np.exp(1j * np.angle(cpl_out.cpx_phase)) * np.abs(slcs_decimated)
    else:
        cpx_phase = np.exp(1j * np.angle(cpl_out.cpx_phase))

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(nodata_mask, *strides, func_type="all")

    # Set no data pixels to np.nan
    temp_coh = np.where(mask_looked, np.nan, cpl_out.temp_coh)

    # Fill in the PS pixels from the original SLC stack, if it was given
    if np.any(ps_mask):
        fill_ps_pixels(
            cpx_phase,
            temp_coh,
            slc_stack,
            ps_mask,
            strides,
            avg_mag,
            reference_idx,
        )

    return PhaseLinkOutput(
        cpx_phase,
        temp_coh,
        # Convert the rest to numpy for writing
        np.array(cpl_out.eigenvalues),
        np.array(cpl_out.estimator),
        cpl_out.avg_coh,
    )


def run_cpl(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides,
    use_evd: bool = False,
    beta: float = 0,
    reference_idx: int = 0,
    neighbor_arrays: Optional[np.ndarray] = None,
    calc_average_coh: bool = False,
) -> PhaseLinkOutput:
    """Run the Combined Phase Linking (CPL) algorithm.

    Estimates a coherence matrix for each SLC pixel, then
    runs the EMI/EVD solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    calc_average_coh : bool, default=False
        If requested, the average of each row of the covariance matrix is computed
        for the purposes of finding the best reference (highest coherence) date

    Returns
    -------
    cpx_phase : Array
        Optimized SLC phase, shape same as `slc_stack` unless Strides are requested.
    temp_coh : Array
        Temporal coherence of the optimization.
        A goodness of fit parameter from 0 to 1 at each pixel.
        shape = (out_rows, out_cols)
    eigenvalues : Array
        The eigenvalues of the coherence matrices.
        If `use_evd` is True, these are the largest eigenvalues;
        Otherwise, for EMI they are the smallest.
        shape = (out_rows, out_cols)
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI
        shape = (out_rows, out_cols)
    avg_coh : np.ndarray | None
        The average coherence of each row of the coherence matrix,
        if requested.
        shape = (nslc, out_rows, out_cols)

    """
    C_arrays = covariance.estimate_stack_covariance(
        slc_stack,
        half_window,
        strides,
        neighbor_arrays=neighbor_arrays,
    )

    cpx_phase, eigenvalues, estimator = process_coherence_matrices(
        C_arrays,
        use_evd=use_evd,
        beta=beta,
        reference_idx=reference_idx,
    )
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)

    if calc_average_coh:
        # If requested, average the Cov matrix at each row for reference selection
        avg_coh_per_date = jnp.abs(C_arrays).mean(axis=3)
        avg_coh = np.argmax(avg_coh_per_date, axis=2)
    else:
        avg_coh = None

    # Reshape the (rows, cols, nslcs) output to be same as input stack
    cpx_phase_reshaped = jnp.moveaxis(cpx_phase, -1, 0)
    return PhaseLinkOutput(
        cpx_phase_reshaped, temp_coh, eigenvalues, estimator, avg_coh
    )


@partial(jit, static_argnames=("use_evd", "beta", "reference_idx"))
def process_coherence_matrices(
    C_arrays,
    use_evd: bool = False,
    beta: float = 0.0,
    reference_idx: int = 0,
) -> tuple[Array, Array, Array]:
    """Estimate the linked phase for a stack of coherence matrices.

    This function is used after coherence estimation to estimate the
    optimized SLC phase.

    Parameters
    ----------
    C_arrays : ndarray, shape = (rows, cols, nslc, nslc)
        The sample coherence matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance][])
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm of [@Ansari2018EfficientPhaseEstimation].
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
        Default is 0 (no regularization).
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        All outputs are multiplied by the conjugate of the data at this index.

    Returns
    -------
    eig_vecs : ndarray[float32], shape = (rows, cols, nslc)
        The phase resulting from the optimization at each output pixel.
        Shape is same as input slcs unless Strides > (1, 1)
    eig_vals : ndarray[float], shape = (rows, cols)
        The smallest (largest) eigenvalue as solved by EMI (EVD).
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI

    """
    rows, cols, n, _ = C_arrays.shape
    if use_evd:
        # EVD
        eig_vals, eig_vecs = eigh_largest_stack(C_arrays)
        estimator = jnp.zeros(eig_vals.shape, dtype=bool)
    else:
        # EMI
        # estimate the wrapped phase based on the EMI paper
        # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix

        # Identity used for regularization and for solving
        Id = jnp.eye(n, dtype=C_arrays.dtype)
        # repeat the identity matrix for each pixel
        Id = jnp.tile(Id, (rows, cols, 1, 1))

        Gamma = jnp.abs(C_arrays)
        if beta > 0:
            # Perform regularization
            Gamma = (1 - beta) * Gamma + beta * Id
        # Attempt to invert Gamma
        cho, is_lower = cho_factor(Gamma)

        # Check: If it fails the cholesky factor, it's close to singular and
        # we should just fall back to EVD
        # Use the already- factored |Gamma|^-1, solving Ax = I gives the inverse
        Gamma_inv = cho_solve((cho, is_lower), Id)
        emi_eig_vals, emi_eig_vecs = eigh_smallest_stack(Gamma_inv * C_arrays)
        # From the EMI paper, normalize the eigenvectors to have norm sqrt(n)
        emi_eig_vecs = (
            jnp.sqrt(n)
            * emi_eig_vecs
            / jnp.linalg.norm(emi_eig_vecs, axis=-1, keepdims=True)
        )
        # is the output is the inverse of the eigenvectors? or inverse conj?

        # For places where inverting |Gamma| failed: fall back to computing EVD
        evd_eig_vals, evd_eig_vecs = eigh_largest_stack(C_arrays)

        # Use https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html
        # Note that `if` would fail the jit tracing
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#cond
        inv_has_nans = jnp.any(jnp.isnan(Gamma_inv), axis=(-1, -2))

        # Must broadcast the 2D boolean array so it's the same size as the outputs
        inv_has_nans_3d = jnp.tile(inv_has_nans[:, :, None], (1, 1, n))

        eig_vecs = lax.select(
            inv_has_nans_3d,
            # Run this on True: EVD, since we failed to invert:
            evd_eig_vecs,
            # Otherwise, on False, we're fine to use EMI
            emi_eig_vecs,
        )

        eig_vals = lax.select(inv_has_nans, evd_eig_vals, emi_eig_vals)
        # Make array of ints to indicate which estimator was used for each pixel
        # 0 means EVD, 1 mean EMI
        evd_used = jnp.zeros(emi_eig_vals.shape, dtype=jnp.int8)
        emi_used = jnp.ones(emi_eig_vals.shape, dtype=jnp.int8)
        estimator = lax.select(inv_has_nans, evd_used, emi_used)

    # Now the shape of eig_vecs is (rows, cols, nslc)
    # at pixel (r, c), eig_vecs[r, c] is the largest (smallest) eigenvector if
    # we picked EVD (EMI)
    # The phase estimate on the reference day will be size (rows, cols)
    ref = eig_vecs[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = eig_vecs * jnp.exp(-1j * jnp.angle(ref[:, :, None]))

    return evd_estimate, eig_vals, estimator


# The eigenvalues are in ascending order
# Column j of `eig_vecs` is the normalized eigenvector corresponding
# to eigenvalue `lam[j]``
@jit
def _get_smallest_eigenpair(C) -> tuple[Array, Array]:
    lam, eig_vecs = eigh(C)
    return lam[0], eig_vecs[:, 0]


@jit
def _get_largest_eigenpair(C) -> tuple[Array, Array]:
    lam, eig_vecs = eigh(C)
    return lam[-1], eig_vecs[:, -1]


# We map over the first two dimensions, so now instead of one scalar eigenvalue,
# we have (rows, cols) eigenvalues
@jit
def eigh_smallest_stack(C_arrays: ArrayLike) -> tuple[Array, Array]:
    """Get the smallest (eigenvalue, eigenvector) for each pixel in a 3D stack.

    Parameters
    ----------
    C_arrays : ArrayLike
        The stack of coherence matrices.
        Shape = (rows, cols, nslc, nslc)

    Returns
    -------
    eigenvalues : Array
        The smallest eigenvalue for each pixel's matrix
        Shape = (rows, cols)
    eigenvectors : Array
        The normalized eigenvector corresponding to the smallest eigenvalue
        Shape = (rows, cols, nslc)

    """
    return vmap(vmap(_get_smallest_eigenpair))(C_arrays)


@jit
def eigh_largest_stack(C_arrays: ArrayLike) -> tuple[Array, Array]:
    """Get the largest (eigenvalue, eigenvector) for each pixel in a 3D stack.

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
    return vmap(vmap(_get_largest_eigenpair))(C_arrays)


def decimate(arr: ArrayLike, strides: Strides) -> Array:
    """Decimate an array by strides in the x and y directions.

    Output will match [`io.compute_out_shape`][dolphin.io.compute_out_shape]

    Parameters
    ----------
    arr : ArrayLike
        2D or 3D array to decimate.
    strides : dict[str, int]
        The strides in the x and y directions.

    Returns
    -------
    ArrayLike
        The decimated array.

    """
    ys, xs = strides
    rows, cols = arr.shape[-2:]
    start_r = ys // 2
    start_c = xs // 2
    end_r = (rows // ys) * ys + 1
    end_c = (cols // xs) * xs + 1
    return arr[..., start_r:end_r:ys, start_c:end_c:xs]


def _raise_if_all_nan(slc_stack: np.ndarray):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        msg = f"slc_stack[{bad_slc_idxs}] out of {len(slc_stack)} are all NaNs."
        raise PhaseLinkRuntimeError(msg)
