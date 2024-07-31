import numpy as np
from numpy.typing import ArrayLike
from scipy import fft, ndimage


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    correlation: ArrayLike,
    correlation_cutoff: float = 0.5,
    connected_component_labels: ArrayLike | None = None,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
    workers: int = 1,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Parameters
    ----------
    unwrapped_phase : np.ndarray, 2D complex array
        Unwrapped interferogram phase to filter.
    correlation : Arraylike, 2D
        Array of interferometric correlation from 0 to 1.
    correlation_cutoff : float
        Threshold to use on `correlation` so that pixels where
        `correlation[i, j] > correlation_cutoff` are used and the rest are ignored.
        The default is 0.5.
    connected_component_labels : ArrayLike, optional
        Integer labels of connected components from the unwrapped interferogram.
        If provided, ignores pixels with label 0.
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
    nrow, ncol = correlation.shape
    good_pixel_mask = correlation > correlation_cutoff
    non_boundary_mask = ~(correlation == 0)

    if connected_component_labels is not None:
        good_pixel_mask = good_pixel_mask & (connected_component_labels != 0)

    unw0 = np.nan_to_num(unwrapped_phase)
    unw_valid_mask = unw0 != 0
    total_valid_mask = unw_valid_mask & good_pixel_mask

    # Shift to be zero mean, then reset the borders to 0
    offset = np.mean(unw0[total_valid_mask])
    unw0 -= offset
    unw0[~total_valid_mask] = 0

    plane = fit_ramp_plane(unw0, total_valid_mask)

    unw_ifg_interp = np.where(total_valid_mask, unw0, plane)

    # Pad the array with edge values
    pad_rows = nrow // 4
    pad_cols = ncol // 4
    # See here for illustration of `mode="reflect"`
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_edge_modes.html#interpolation-edge-modes
    padded = np.pad(
        unw_ifg_interp, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="reflect"
    )

    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)
    # Apply Gaussian filter
    input_ = fft.fft2(padded, workers=workers)
    result = ndimage.fourier_gaussian(input_, sigma=sigma)
    # Make sure to only take the real part (ifft returns complex)
    result = fft.ifft2(result, workers=workers).real.astype(unwrapped_phase.dtype)

    # Crop back to original size
    lowpass_filtered = result[pad_rows:-pad_rows, pad_cols:-pad_cols]

    filtered_ifg = unw0 - lowpass_filtered * non_boundary_mask

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
