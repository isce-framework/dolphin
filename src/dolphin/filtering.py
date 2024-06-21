import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    correlation: ArrayLike,
    mask_cutoff: float = 0.5,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Parameters
    ----------
    unwrapped_phase : np.ndarray, 2D complex array
        Unwrapped interferogram phase to filter.
    correlation : Arraylike, 2D
        Array of interferometric correlation from 0 to 1.
    mask_cutoff: float
        Threshold to use on `correlation` so that pixels where
        `correlation[i, j] > mask_cutoff` are used and the rest are ignored.
        The default is 0.5.
    wavelength_cutoff: float
        Spatial wavelength threshold to filter the unwrapped phase.
        Signals with wavelength longer than 'wavelength_cutoff' are filtered out.
        The default is 50*1e3 (m).
    pixel_spacing : float
        Pixel spatial spacing. Assume same spacing for x, y axes.
        The default is 30 (m).

    Returns
    -------
    filtered_ifg : 2D complex array
        filtered interferogram that does not contain signals with spatial wavelength
        longer than a threshold.

    """
    nrow, ncol = correlation.shape

    mask = (correlation > mask_cutoff).astype("bool")

    # Create Boolean mask for Zero-filled boundary area to be False
    # and the rest to be True
    mask_boundary = ~(correlation == 0).astype("bool")

    plane = fit_ramp_plane(unwrapped_phase, mask)

    # Replace masked out pixels with the ramp plane
    unw_ifg_interp = np.copy(unwrapped_phase)
    unw_ifg_interp[~mask * mask_boundary] = plane[~mask * mask_boundary]

    # Copy the edge pixels for the boundary area before filling them by reflection
    EV_fill = np.copy(unw_ifg_interp)

    Xdata = np.argwhere(mask)  # Get indices of non-NaN & masked pixels
    NW = Xdata[np.argmin(Xdata[:, 0])]  # Get indices of upper left corner pixel
    SE = Xdata[np.argmax(Xdata[:, 0])]  # Get indices of lower right corner pixel
    SW = Xdata[np.argmin(Xdata[:, 1])]  # Get indices of lower left corner pixel
    NE = Xdata[np.argmax(Xdata[:, 1])]  # Get indices of upper left corner pixel

    for k in range(NW[1], NE[1] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[0 : NE[0] + 1, k] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        EV_fill[0:n_zeros, k] = EV_fill[n_zeros, k]
    for k in range(SW[1], SE[1] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[SW[0] + 1 :, k] == 0
        )  # count zeros in South direction
        if n_zeros == 0:
            continue
        EV_fill[-n_zeros:, k] = EV_fill[-n_zeros - 1, k]
    for k in range(NW[0], SW[0] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[k, 0 : NW[1] + 1] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        EV_fill[k, 0:n_zeros] = EV_fill[k, n_zeros]
    for k in range(NE[0], SE[0] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[k, SE[1] + 1 :] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        EV_fill[k, -n_zeros:] = EV_fill[k, -n_zeros - 1]

    # Fill the boundary area reflecting the pixel values
    Reflect_fill = np.copy(EV_fill)

    for k in range(NW[1], NE[1] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[0 : NE[0] + 1, k] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        Reflect_fill[0:n_zeros, k] = np.flipud(EV_fill[n_zeros : n_zeros + n_zeros, k])
    for k in range(SW[1], SE[1] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[SW[0] + 1 :, k] == 0
        )  # count zeros in South direction
        if n_zeros == 0:
            continue
        Reflect_fill[-n_zeros:, k] = np.flipud(
            EV_fill[-n_zeros - n_zeros : -n_zeros, k]
        )
    for k in range(NW[0], SW[0] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[k, 0 : NW[1] + 1] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        Reflect_fill[k, 0:n_zeros] = np.flipud(EV_fill[k, n_zeros : n_zeros + n_zeros])
    for k in range(NE[0], SE[0] + 1):
        n_zeros = np.count_nonzero(
            unw_ifg_interp[k, SE[1] + 1 :] == 0
        )  # count zeros in North direction
        if n_zeros == 0:
            continue
        Reflect_fill[k, -n_zeros:] = np.flipud(
            EV_fill[k, -n_zeros - n_zeros : -n_zeros]
        )

    Reflect_fill[0 : NW[0], 0 : NW[1]] = np.flipud(
        Reflect_fill[NW[0] : NW[0] + NW[0], 0 : NW[1]]
    )  # upper left corner area
    Reflect_fill[0 : NE[0], NE[1] + 1 :] = np.fliplr(
        Reflect_fill[0 : NE[0], NE[1] + 1 - (ncol - NE[1] - 1) : NE[1] + 1]
    )  # upper right corner area
    Reflect_fill[SW[0] + 1 :, 0 : SW[1]] = np.fliplr(
        Reflect_fill[SW[0] + 1 :, SW[1] : SW[1] + SW[1]]
    )  # lower left corner area
    Reflect_fill[SE[0] + 1 :, SE[1] + 1 :] = np.flipud(
        Reflect_fill[SE[0] + 1 - (nrow - SE[0] - 1) : SE[0] + 1, SE[1] + 1 :]
    )  # lower right corner area

    # 2D filtering with Gaussian kernel
    # wavelength_cutoff: float = 50*1e3,
    # dx: float = 30,
    cutoff_value = 0.5  # 0 < cutoff_value < 1
    sigma_f = (
        1 / wavelength_cutoff / np.sqrt(np.log(1 / cutoff_value))
    )  # fc = sqrt(ln(1/cutoff_value))*sigma_f
    sigma_x = 1 / np.pi / 2 / sigma_f
    sigma = sigma_x / pixel_spacing

    lowpass_filtered = ndimage.gaussian_filter(Reflect_fill, sigma)
    filtered_ifg = unwrapped_phase - lowpass_filtered * mask_boundary

    return filtered_ifg


def fit_ramp_plane(unw_ifg: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Fit a ramp plane to the given data.

    Parameters
    ----------
    unw_ifg : ArrayLike
        2D array where the unwrapped interferogram data is stored.
    mask : ArrayLike
        2D boolean array indicating the valid (non-NaN) pixels.

    Returns
    -------
    np.ndarray
        2D array of the fitted ramp plane.

    """
    # Extract data for non-NaN & masked pixels
    Y = unw_ifg[mask]
    Xdata = np.argwhere(mask)  # Get indices of non-NaN & masked pixels

    # Include the intercept term (bias) in the model
    X = np.c_[np.ones((len(Xdata))), Xdata]

    # Compute the parameter vector theta using the least squares solution
    theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    # Prepare grid for the entire image
    nrow, ncol = unw_ifg.shape
    X1_, X2_ = np.mgrid[:nrow, :ncol]
    X_ = np.hstack(
        (np.reshape(X1_, (nrow * ncol, 1)), np.reshape(X2_, (nrow * ncol, 1)))
    )
    X_ = np.hstack((np.ones((nrow * ncol, 1)), X_))

    # Compute the fitted plane
    plane = np.reshape(X_ @ theta, (nrow, ncol))

    return plane
