import numpy as np
from numpy.typing import ArrayLike
from scipy import fft, ndimage


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    bad_pixel_mask: ArrayLike,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
    workers: int = 1,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Parameters
    ----------
    unwrapped_phase : ArrayLike
        Unwrapped interferogram phase to filter.
    bad_pixel_mask : ArrayLike
        Boolean array with same shape as `unwrapped_phase` where `True` indicates a
        pixel to ignore during ramp fitting
    wavelength_cutoff : float
        Spatial wavelength threshold to filter the unwrapped phase.
        Signals with wavelength longer than 'wavelength_cutoff' are filtered out.
        The default is 50*1e3 (m).
    pixel_spacing : float
        Pixel spatial spacing. Assume same spacing for x, y axes.
        The default is 30 (m).
    workers : int
        Number of `fft` workers to use for `scipy.fft.fft2`.
        Default is 1.

    Returns
    -------
    filtered_ifg : 2D complex array
        filtered interferogram that does not contain signals with spatial wavelength
        longer than a threshold.

    """
    good_pixel_mask = ~bad_pixel_mask

    unw0 = np.nan_to_num(unwrapped_phase)
    # Take either nan or 0 pixels in `unwrapped_phase` to be nodata
    nodata_mask = unw0 == 0
    in_bounds_pixels = ~nodata_mask

    total_valid_mask = in_bounds_pixels & good_pixel_mask

    plane = fit_ramp_plane(unw0, total_valid_mask)
    # Remove the plane, setting to 0 where we had no data for the plane fit:
    unw_ifg_interp = np.where((~nodata_mask & good_pixel_mask), unw0, plane)

    # Find the filter `sigma` which gives the correct cutoff in meters
    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)

    # Pad the array with edge values
    # The padding extends further than the default "radius = 2*sigma + 1",
    # which given specified in `gaussian_filter`
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
    pad_rows = pad_cols = int(3 * sigma)
    # See here for illustration of `mode="reflect"`
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_edge_modes.html#interpolation-edge-modes
    padded = np.pad(
        unw_ifg_interp, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="reflect"
    )

    # Apply Gaussian filter
    result = fft.fft2(padded, workers=workers)
    result = ndimage.fourier_gaussian(result, sigma=sigma)
    # Make sure to only take the real part (ifft returns complex)
    result = fft.ifft2(result, workers=workers).real.astype(unwrapped_phase.dtype)

    # Crop back to original size
    lowpass_filtered = result[pad_rows:-pad_rows, pad_cols:-pad_cols]

    filtered_ifg = unw_ifg_interp - lowpass_filtered * in_bounds_pixels
    return np.where(in_bounds_pixels, filtered_ifg, 0)


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
