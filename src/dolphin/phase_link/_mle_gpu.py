from __future__ import annotations

from math import ceil
from typing import Optional

import numpy as np
from numba import cuda

from dolphin.io import compute_out_shape
from dolphin.utils import decimate

from . import covariance, metrics
from .mle import MleOutput, mle_stack


def run_gpu(
    slc_stack: np.ndarray,
    half_window: dict[str, int],
    strides: dict[str, int] = {"x": 1, "y": 1},
    beta: float = 0.01,
    reference_idx: int = 0,
    use_slc_amp: bool = True,
    threads_per_block: tuple[int, int] = (8, 8),
    neighbor_arrays: Optional[np.ndarray] = None,
    free_mem: bool = False,
    calc_average_coh: bool = True,
    **kwargs,
    # ) -> tuple[np.ndarray, np.ndarray]:
) -> MleOutput:
    """Run the GPU version of the stack covariance estimator and MLE solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : dict[str, int]
        The half window size as {"x": half_win_x, "y": half_win_y}
        The full window size is 2 * half_window + 1 for x, y.
    strides : dict[str, int], optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    beta : float, optional
        The regularization parameter, by default 0.01.
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default True.
    threads_per_block : tuple[int, int], optional
        The number of threads per block to use for the GPU kernel.
        By default (8, 8)
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    free_mem : bool, default=False
        Whether to free the memory of the covariance matrix after the MLE
        estimation. By default False.
    calc_average_coh : bool, default=False
        If requested, the average of each row of the covariance matrix is computed
        for the purposes of finding the best reference (highest coherence) date

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_slc, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    import cupy as cp

    num_slc, rows, cols = slc_stack.shape

    # Can't use dict in numba kernels, so pass the values as a tuple
    halfwin_rowcol = (half_window["y"], half_window["x"])
    strides_rowcol = (strides["y"], strides["x"])

    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    # Copy the read-only data to the device
    d_slc_stack = cuda.to_device(slc_stack)

    # Divide up the output shape using a 2D grid
    blocks_x = ceil(out_cols / threads_per_block[0])
    blocks_y = ceil(out_rows / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    # Make a buffer for each pixel's coherence matrix
    # d_ means "device_", i.e. on the GPU
    d_C_arrays = cp.zeros((out_rows, out_cols, num_slc, num_slc), dtype=np.complex64)

    if neighbor_arrays is not None and neighbor_arrays.size > 1:
        # contiguous needed for cupy, or slicing can make it error
        d_neighbor_arrays = cuda.to_device(np.ascontiguousarray(neighbor_arrays))
        do_shp = True
    else:
        # Dummy array to pass in for the same type
        d_neighbor_arrays = cp.zeros((1, 1, 1, 1), dtype=np.bool_)
        do_shp = False

    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slc_stack,
        halfwin_rowcol,
        strides_rowcol,
        d_neighbor_arrays,
        d_C_arrays,
        do_shp,
    )

    d_output_phase = mle_stack(d_C_arrays, beta=beta, reference_idx=reference_idx)
    d_cpx_phase = cp.exp(1j * d_output_phase)

    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(d_cpx_phase, d_C_arrays).get()

    if calc_average_coh:
        # If requested, average the Cov matrix at each row for reference selection
        d_avg_coh_per_date = cp.abs(d_C_arrays).mean(axis=3)
        avg_coh = cp.argmax(d_avg_coh_per_date, axis=2).get()
        # avg_coh = cp.abs(d_C_arrays).mean(axis=3).get()
    else:
        avg_coh = None

    mle_est = d_cpx_phase.get()
    # # https://docs.cupy.dev/en/stable/user_guide/memory.html
    # may just be cached a lot of the huge memory available on aurora
    # But if we need to free GPU memory:
    if free_mem:
        del d_slc_stack
        del d_C_arrays
        del d_output_phase
        del d_cpx_phase
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        # we need to match `io.compute_out_shape` here
        slcs_decimated = decimate(slc_stack, strides)
        mle_est *= np.abs(slcs_decimated)

    return MleOutput(mle_est, temp_coh, avg_coh)
