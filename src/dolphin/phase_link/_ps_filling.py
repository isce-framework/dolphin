from __future__ import annotations

import logging
import warnings

import numpy as np

from dolphin._types import Strides
from dolphin.utils import take_looks

logger = logging.getLogger(__name__)


def fill_ps_pixels(
    cpx_phase: np.ndarray,
    temp_coh: np.ndarray,
    slc_stack: np.ndarray,
    ps_mask: np.ndarray,
    strides: Strides,
    avg_mag: np.ndarray,
    reference_idx: int = 0,
    use_max_ps: bool = False,
):
    """Fill in the PS locations in the MLE estimate with the original SLC data.

    Overwrites `cpx_phase` and `temp_coh` in place.

    Parameters
    ----------
    cpx_phase : ndarray, shape = (nslc, rows, cols)
        The complex valued-MLE estimate of the phase.
    temp_coh : ndarray, shape = (rows, cols)
        The temporal coherence of the estimate.
    slc_stack : np.ndarray
        The original SLC stack, with shape (n_images, n_rows, n_cols)
    ps_mask : ndarray, shape = (rows, cols)
        Boolean mask of pixels marking persistent scatterers (PS).
    strides : Strides
        The decimation (y, x) factors
    avg_mag : np.ndarray, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
    reference_idx : int, default = 0
        SLC to use as reference for PS pixels. All pixel values are multiplied
        by the conjugate of this index
    use_max_ps : bool, optional, default = False
        If True, use the brightest PS pixel in each look window to fill in the
        MLE estimate. If False, use the average of all PS pixels in each look window.

    """
    if avg_mag is None:
        # Get the average magnitude of the SLC stack
        # nanmean will ignore single NaNs, but not all NaNs, per pixel
        with warnings.catch_warnings():
            # ignore the warning about nansum/nanmean of empty slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_mag = np.nanmean(np.abs(slc_stack), axis=0)
    mag = avg_mag.copy()

    # null out all the non-PS pixels when finding the brightest PS pixels
    mag[~ps_mask] = np.nan
    # For ps_mask, we set to True if any pixels within the window were PS
    ps_mask_looked = take_looks(ps_mask, *strides, func_type="any", edge_strategy="pad")
    # make sure it's the same size as the MLE result/temp_coh after padding
    ps_mask_looked = ps_mask_looked[: cpx_phase.shape[1], : cpx_phase.shape[2]]

    if use_max_ps:
        ps_phases = np.empty(cpx_phase.shape, dtype=np.float32)
        logger.info("Using max PS pixel to fill in MLE estimate")
        # Get the indices of the brightest pixels within each look window
        slc_r_idxs, slc_c_idxs = _get_max_idxs(mag, *strides)
        # we're only filling where there are PS pixels
        ref = np.exp(-1j * np.angle(slc_stack[reference_idx][slc_r_idxs, slc_c_idxs]))
        for i in range(len(slc_stack)):
            slc_phase = np.angle(slc_stack[i][slc_r_idxs, slc_c_idxs])
            cur_amp = np.abs(cpx_phase[i][slc_r_idxs, slc_c_idxs])
            new_value = cur_amp * np.exp(1j * slc_phase) * ref
            cpx_phase[i][ps_mask_looked] = new_value

        ps_phases = np.angle(cpx_phase[:, ps_mask_looked])
    else:
        # Get the average of all PS pixels within each look window
        # The referencing to SLC 0 is done in _get_avg_ps
        avg_ps = _get_avg_ps(slc_stack, ps_mask, strides)[
            :, : cpx_phase.shape[1], : cpx_phase.shape[2]
        ]
        ps_phases = np.angle(avg_ps[:, ps_mask_looked])

    # Set the angle only, don't change magnitude
    cpx_phase[:, ps_mask_looked] = np.abs(cpx_phase[:, ps_mask_looked]) * np.exp(
        1j * ps_phases
    )

    # Force PS pixels to have high temporal coherence
    temp_coh[ps_mask_looked] = 1


def _get_avg_ps(
    slc_stack: np.ndarray, ps_mask: np.ndarray, strides: Strides
) -> np.ndarray:
    # First, set all non-PS pixels to NaN
    slc_stack_nanned = slc_stack.copy()
    slc_stack_nanned[:, ~ps_mask] = np.nan
    # Reference all ps pixels in the SLC stack to the first SLC
    slc_stack_nanned[:, ps_mask] *= np.exp(
        -1j * np.angle(slc_stack_nanned[0, ps_mask])
    )[None]
    # Then, take the average of all PS pixels within each look window
    return take_looks(
        slc_stack_nanned,
        *strides,
        func_type="nanmean",
        edge_strategy="pad",
    )


def _get_max_idxs(arr, row_looks, col_looks):
    """Get the indices of the maximum value in each look window."""
    if row_looks == 1 and col_looks == 1:
        # No need to pad if we're not looking
        return np.where(arr == arr)
    # Adjusted from this answer to not take every moving window
    # https://stackoverflow.com/a/72742009/4174466
    windows = np.lib.stride_tricks.sliding_window_view(arr, (row_looks, col_looks))[
        ::row_looks, ::col_looks
    ]
    maxvals = np.nanmax(windows, axis=(2, 3))
    indx = np.array((windows == np.expand_dims(maxvals, axis=(2, 3))).nonzero())

    # In [82]: (windows == np.expand_dims(maxvals, axis = (2, 3))).nonzero()
    # This gives 4 arrays:
    # First two are the window indices
    # (array([0, 0, 0, 1, 1, 1]),
    # array([0, 1, 2, 0, 1, 2]),
    # last two are the relative indices (within each window)
    # array([0, 0, 1, 1, 1, 1]),
    # array([1, 1, 1, 1, 1, 0]))
    window_positions, relative_positions = indx.reshape((2, 2, -1))
    # Multiply the first two by the window size to get the absolute indices
    # of the top lefts of the windows
    window_offsets = np.array([row_looks, col_looks]).reshape((2, 1))
    # Then add the last two to get the relative indices
    rows, cols = relative_positions + window_positions * window_offsets
    return rows, cols
