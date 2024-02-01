from __future__ import annotations

from typing import Optional

import numpy as np
from jax import Array
from jax.typing import ArrayLike

from dolphin._log import get_log
from dolphin._types import HalfWindow, Strides

from . import covariance, metrics
from ._core import MleOutput, mle_stack

logger = get_log(__name__)


def run_cpl(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides,
    use_evd: bool = False,
    beta: float = 0.01,
    reference_idx: int = 0,
    use_slc_amp: bool = True,
    neighbor_arrays: Optional[np.ndarray] = None,
    calc_average_coh: bool = False,
    **kwargs,
) -> MleOutput:
    """Run the CPU version of the stack covariance estimator and MLE solver.

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
        The regularization parameter, by default 0.01.
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
    **kwargs : dict, optional
        Additional keyword arguments not used by CPU version.

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_slc, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    C_arrays = covariance.estimate_stack_covariance(
        slc_stack,
        half_window,
        strides,
        neighbor_arrays=neighbor_arrays,
    )

    output_phase = mle_stack(
        C_arrays,
        use_evd=use_evd,
        beta=beta,
        reference_idx=reference_idx,
    )
    cpx_phase = np.exp(1j * output_phase)
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)
    mle_est = np.exp(1j * output_phase)

    if calc_average_coh:
        # If requested, average the Cov matrix at each row for reference selection
        avg_coh_per_date = np.abs(C_arrays).mean(axis=3)
        avg_coh = np.argmax(avg_coh_per_date, axis=2)
    else:
        avg_coh = None

    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        # we need to match `io.compute_out_shape` here
        slcs_decimated = decimate(slc_stack, strides)
        mle_est *= np.abs(slcs_decimated)

    return MleOutput(mle_est, temp_coh, avg_coh)


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
