import warnings

import numpy as np

from dolphin.utils import upsample_nearest


def compress(
    slc_stack: np.ndarray,
    pl_cpx_phase: np.ndarray,
):
    """Compress the stack of SLC data using the estimated phase.

    Parameters
    ----------
    slc_stack : np.array
        The stack of complex SLC data, shape (nslc, rows, cols)
    pl_cpx_phase : np.array
        The estimated complex phase from phase linking.
        shape  = (nslc, rows // strides.y, cols // strides.x)

    Returns
    -------
    np.array
        The compressed SLC data, shape (rows, cols)

    """
    # If the output is downsampled, we need to make `pl_cpx_phase` the same shape
    # as the output
    pl_estimate_upsampled = upsample_nearest(pl_cpx_phase, slc_stack.shape[1:])
    # For each pixel, project the SLCs onto the (normalized) estimated phase
    # by performing a pixel-wise complex dot product
    pl_norm = np.linalg.norm(pl_estimate_upsampled, axis=0)
    # Avoid divide by zero (there may be 0s at the upsampled boundary)
    pl_norm[pl_norm == 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        return (
            np.nansum(slc_stack * np.conjugate(pl_estimate_upsampled), axis=0) / pl_norm
        )
