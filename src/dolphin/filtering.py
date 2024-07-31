import numpy as np
from numpy.typing import ArrayLike
from scipy import fft, ndimage


def filter_long_wavelength_old(
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
    mask = correlation > mask_cutoff
    mask_boundary = ~(correlation == 0)

    plane = fit_ramp_plane(unwrapped_phase, mask)

    unw_ifg_interp = np.where(mask & mask_boundary, unwrapped_phase, plane)

    # Pad the array with edge values
    pad_width = ((nrow // 2, nrow // 2), (ncol // 2, ncol // 2))
    # See here for illustration of `mode="reflect"`
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_edge_modes.html#interpolation-edge-modes
    padded = np.pad(unw_ifg_interp, pad_width, mode="reflect")

    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)
    # Apply Gaussian filter
    input_ = fft.fft2(padded, workers=6)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.fft.ifft2(result).real.astype(unwrapped_phase.dtype)

    # Crop back to original size
    lowpass_filtered = result[nrow // 2 : -nrow // 2, ncol // 2 : -ncol // 2]

    filtered_ifg = unwrapped_phase - lowpass_filtered * mask_boundary

    return filtered_ifg


def _compute_filter_sigma(
    wavelength_cutoff: float, pixel_spacing: float, cutoff_value: float = 0.5
) -> float:
    sigma_f = 1 / wavelength_cutoff / np.sqrt(np.log(1 / cutoff_value))
    sigma_x = 1 / np.pi / 2 / sigma_f
    sigma = sigma_x / pixel_spacing
    return sigma


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
