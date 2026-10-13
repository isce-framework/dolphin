from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from dolphin.utils import upsample_nearest


def compress(
    slc_stack: ArrayLike,
    pl_cpx_phase: ArrayLike,
    is_real_mask: ArrayLike | None = None,
    slc_mean: ArrayLike | None = None,
    reference_idx: int | None = None,
):
    """Compress the stack of SLC data using the estimated phase.

    Parameters
    ----------
    slc_stack : ArrayLike
        The stack of complex SLC data, shape (nslc, rows, cols)
    pl_cpx_phase : ArrayLike
        The estimated complex phase from phase linking.
        shape  = (nslc, rows // strides.y, cols // strides.x)
    is_real_mask : ArrayLike, optional
        Boolean mask of shape (nslc,), where `True` marks a "real" (non-compressed)
        SLC within `slc_stack`. Entries marked `False` (already-Compressed SLCs)
        are excluded from the dot product during compression, regardless of their
        position in the stack.
        If None, all entries are treated as real.
    slc_mean : ArrayLike, optional
        The mean SLC magnitude, shape (rows, cols), to use as output pixel magnitudes.
        If None, the mean is computed from the input SLC stack.
    reference_idx : int, optional
        If provided, the `pl_cpx_phase` will be re-referenced to `reference_idx` before
        performing the dot product.
        Default is `None`, which keeps `pl_cpx_phase` as provided.

    Returns
    -------
    np.array
        The compressed SLC data, shape (rows, cols)

    """
    if reference_idx is not None:
        pl_referenced = pl_cpx_phase * pl_cpx_phase[reference_idx][None, :, :].conj()
    else:
        pl_referenced = pl_cpx_phase

    if is_real_mask is None:
        is_real_mask = np.ones(np.asarray(slc_stack).shape[0], dtype=bool)
    else:
        is_real_mask = np.asarray(is_real_mask, dtype=bool)

    # Exclude the already-compressed layers *after* doing the reference.
    pl_referenced = pl_referenced[is_real_mask, :, :]
    slcs = slc_stack[is_real_mask]
    # If the output is downsampled, we need to make `pl_cpx_phase` the same shape
    # as the output
    pl_estimate_upsampled = upsample_nearest(pl_referenced, slcs.shape[1:])
    # For each pixel, project the SLCs onto the (normalized) estimated phase
    # by performing a pixel-wise complex dot product
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        phase = np.angle(np.nansum(slcs * np.conjugate(pl_estimate_upsampled), axis=0))
    if slc_mean is None:
        slc_mean = np.mean(np.abs(slcs), axis=0)
    # If the phase is invalid, set the mean to NaN
    slc_mean[phase == 0] = np.nan
    return slc_mean * np.exp(1j * phase)
