import warnings

import numpy as np

from dolphin.utils import upsample_nearest


def compress(
    slc_stack: np.ndarray,
    mle_estimate: np.ndarray,
):
    """Compress the stack of SLC data using the estimated phase.

    Parameters
    ----------
    slc_stack : np.array
        The stack of complex SLC data, shape (nslc, rows, cols)
    mle_estimate : np.array
        The estimated phase from [`run_mle`][dolphin.phase_link.mle.run_mle],
        shape (nslc, rows // strides['y'], cols // strides['x'])

    Returns
    -------
    np.array
        The compressed SLC data, shape (rows, cols)
    """
    # If the output is downsampled, we need to make `mle_estimate` the same shape
    # as the output
    mle_estimate_upsampled = upsample_nearest(mle_estimate, slc_stack.shape[1:])
    # For each pixel, project the SLCs onto the (normalized) estimated phase
    # by performing a pixel-wise complex dot product
    mle_norm = np.linalg.norm(mle_estimate_upsampled, axis=0)
    # Avoid divide by zero (there may be 0s at the upsampled boundary)
    mle_norm[mle_norm == 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        return (
            np.nansum(slc_stack * np.conjugate(mle_estimate_upsampled), axis=0)
            / mle_norm
        )
