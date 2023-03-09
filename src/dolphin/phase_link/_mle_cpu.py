import logging
from typing import Dict, Optional

import numpy as np

from dolphin._types import Filename
from dolphin.workflows import ShpMethod

from . import covariance, metrics, shp
from .mle import mle_stack

logger = logging.getLogger(__name__)


def run_cpu(
    slc_stack: np.ndarray,
    half_window: Dict[str, int],
    strides: Dict[str, int] = {"x": 1, "y": 1},
    beta: float = 0.01,
    reference_idx: int = 0,
    use_slc_amp: bool = True,
    shp_method: str = ShpMethod.KL,
    avg_mag: Optional[np.ndarray] = None,
    var_mag: Optional[np.ndarray] = None,
    output_cov_file: Optional[Filename] = None,
    n_workers: int = 1,
    **kwargs,
):
    """Run the CPU version of the stack covariance estimator and MLE solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : Dict[str, int]
        The half window size as {"x": half_win_x, "y": half_win_y}
        The full window size is 2 * half_window + 1 for x, y.
    strides : Dict[str, int], optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    beta : float, optional
        The regularization parameter, by default 0.01.
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.
    shp_method : Optional[str]
        The SHP estimator to use. Either None, "KL" or "KS".
        By default "KL", uses the KL distance estimator.
        If None, turns of the SHP search and uses a rectangular window.
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to find the SHP
        neighbors to fill within each look window if shp_method is "KL".
        If None, the average magnitude is estimated from the SLC stack.
        By default None.
    var_mag : np.ndarray, optional
        The variance of the magnitude of the SLC stack, used to find the
        SHP neighbors to fill within each look window if shp_
    output_cov_file : str, optional
        HDF5 filename to save the estimated covariance at each pixel.
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_slc, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    if shp_method.lower() == ShpMethod.KL:
        if avg_mag is None:
            avg_mag = np.mean(np.abs(slc_stack), axis=0)
        if var_mag is None:
            var_mag = np.var(np.abs(slc_stack), axis=0)
        halfwin_rowcol = (half_window["y"], half_window["x"])
        # TODO: threshold as a parameter?
        logger.info("Estimating SHP neighbors using KL distance")
        neighbor_arrays = shp.estimate_neighbors_kl_cpu(
            avg_mag, var_mag, halfwin_rowcol=halfwin_rowcol, threshold=0.5
        )
    elif shp_method == ShpMethod.RECT:
        neighbor_arrays = None
    else:
        logger.warning(f"SHP method {shp_method} is not implemented for CPU yet")
        neighbor_arrays = None

    C_arrays = covariance.estimate_stack_covariance_cpu(
        slc_stack,
        half_window,
        strides,
        neighbor_arrays=neighbor_arrays,
        n_workers=n_workers,
    )
    if output_cov_file:
        covariance._save_covariance(output_cov_file, C_arrays)

    output_phase = mle_stack(C_arrays, beta, reference_idx, n_workers=n_workers)
    cpx_phase = np.exp(1j * output_phase)
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)
    mle_est = np.exp(1j * output_phase)
    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        xs, ys = strides["x"], strides["y"]
        _, rows, cols = slc_stack.shape
        # we need to match `io.compute_out_shape` here
        start_r = ys // 2
        start_c = xs // 2
        end_r = (rows // ys) * ys + 1
        end_c = (cols // xs) * xs + 1
        slcs_decimated = slc_stack[:, start_r:end_r:ys, start_c:end_c:xs]
        mle_est *= np.abs(slcs_decimated)

    return mle_est, temp_coh
