from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft, ndimage


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    bad_pixel_mask: ArrayLike,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
    workers: int = 1,
    fill_value: float | None = None,
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
    fill_value : float, optional
        Value to place in output pixels which were masked.
        If `None`, masked pixels are filled with the ramp value fitted
        before filtering to suppress outliers.

    Returns
    -------
    filtered_ifg : 2D complex array
        filtered interferogram that does not contain signals with spatial wavelength
        longer than a threshold.

    Raises
    ------
    ValueError
        If wavelength_cutoff too large for image size/pixel spacing.

    """
    good_pixel_mask = np.logical_not(bad_pixel_mask)

    rows, cols = unwrapped_phase.shape
    unw0 = np.nan_to_num(unwrapped_phase)
    # Take either nan or 0 pixels in `unwrapped_phase` to be nodata
    nodata_mask = unw0 == 0
    in_bounds_pixels = np.logical_not(nodata_mask)

    total_valid_mask = in_bounds_pixels & good_pixel_mask

    plane = fit_ramp_plane(unw0, total_valid_mask)
    # Remove the plane, setting to 0 where we had no data for the plane fit:
    unw_ifg_interp = np.where(total_valid_mask, unw0, plane)

    # Find the filter `sigma` which gives the correct cutoff in meters
    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)

    if sigma > unw0.shape[0] or sigma > unw0.shape[0]:
        msg = f"{wavelength_cutoff = } too large for image."
        msg += f"Shape = {(rows, cols)}, and {pixel_spacing = }"
        raise ValueError(msg)
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
    if fill_value is not None:
        return np.where(total_valid_mask, filtered_ifg, fill_value)
    else:
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


def filter_rasters(
    unw_filenames: list[Path],
    cor_filenames: list[Path] | None = None,
    conncomp_filenames: list[Path] | None = None,
    temporal_coherence_filename: Path | None = None,
    wavelength_cutoff: float = 50_000,
    correlation_cutoff: float = 0.5,
    output_dir: Path | None = None,
    max_workers: int = 4,
) -> list[Path]:
    """Filter a list of unwrapped interferogram files using a long-wavelength filter.

    Remove long-wavelength components from each unwrapped interferogram.
    It can optionally use temporal coherence, correlation, and connected component
    information for masking.

    Parameters
    ----------
    unw_filenames : list[Path]
        List of paths to unwrapped interferogram files to be filtered.
    cor_filenames : list[Path] | None
        List of paths to correlation files
        Passing None skips filtering on correlation.
    conncomp_filenames : list[Path] | None
        List of paths to connected component files, filters any 0 labeled pixels.
        Passing None skips filtering on connected component labels.
    temporal_coherence_filename : Path | None
        Path to the temporal coherence file for masking.
        Passing None skips filtering on temporal coherence.
    wavelength_cutoff : float, optional
        Spatial wavelength cutoff (in meters) for the filter. Default is 50,000 meters.
    correlation_cutoff : float, optional
        Threshold of correlation (if passing `cor_filenames`) to use to ignore pixels
        during filtering.
    output_dir : Path | None, optional
        Directory to save the filtered results.
        If None, saves in the same location as inputs with .filt.tif extension.
    max_workers : int, optional
        Number of parallel images to process. Default is 4.

    Returns
    -------
    list[Path]
        Output filtered rasters.

    Notes
    -----
    - If temporal_coherence_filename is provided, pixels with coherence < 0.5 are masked

    """
    from dolphin import io

    bad_pixel_mask = np.zeros(
        io.get_raster_xysize(unw_filenames[0])[::-1], dtype="bool"
    )
    if temporal_coherence_filename:
        bad_pixel_mask = bad_pixel_mask | (
            io.load_gdal(temporal_coherence_filename) < 0.5
        )

    if output_dir is None:
        assert unw_filenames
        output_dir = unw_filenames[0].parent
    output_dir.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers, mp_context=ctx) as pool:
        return list(
            pool.map(
                _filter_and_save,
                unw_filenames,
                cor_filenames or repeat(None),
                conncomp_filenames or repeat(None),
                repeat(output_dir),
                repeat(wavelength_cutoff),
                repeat(bad_pixel_mask),
                repeat(correlation_cutoff),
            )
        )


def _filter_and_save(
    unw_filename: Path,
    cor_path: Path | None,
    conncomp_path: Path | None,
    output_dir: Path,
    wavelength_cutoff: float,
    bad_pixel_mask: NDArray[np.bool_],
    correlation_cutoff: float = 0.5,
) -> Path:
    """Filter one interferogram (wrapper for multiprocessing)."""
    from dolphin import io
    from dolphin._overviews import Resampling, create_image_overviews

    # Average for the pixel spacing for filtering
    _, x_res, _, _, _, y_res = io.get_raster_gt(unw_filename)
    pixel_spacing = (abs(x_res) + abs(y_res)) / 2

    if cor_path is not None:
        bad_pixel_mask |= io.load_gdal(cor_path) < correlation_cutoff
    if conncomp_path is not None:
        bad_pixel_mask |= io.load_gdal(conncomp_path, masked=True).astype(bool) == 0

    unw = io.load_gdal(unw_filename)
    filt_arr = filter_long_wavelength(
        unwrapped_phase=unw,
        wavelength_cutoff=wavelength_cutoff,
        bad_pixel_mask=bad_pixel_mask,
        pixel_spacing=pixel_spacing,
        workers=1,
    )
    io.round_mantissa(filt_arr, keep_bits=9)
    output_name = output_dir / Path(unw_filename).with_suffix(".filt.tif").name
    io.write_arr(arr=filt_arr, like_filename=unw_filename, output_name=output_name)

    create_image_overviews(output_name, resampling=Resampling.AVERAGE)

    return output_name


def gaussian_filter_nan(
    image: ArrayLike, sigma: float | Sequence[float], mode="constant", **kwargs
) -> np.ndarray:
    """Apply a gaussian filter to an image with NaNs (avoiding all nans).

    The scipy.ndimage `gaussian_filter` will make the output all NaNs if
    any of the pixels in the input that touches the kernel is NaN

    Source:
    https://stackoverflow.com/a/36307291

    Parameters
    ----------
    image : ndarray
        Image with nans to filter
    sigma : float
        Size of filter kernel. passed into `gaussian_filter`
    mode : str, default = "constant"
        Boundary mode for `[scipy.ndimage.gaussian_filter][]`
    **kwargs : Any
        Passed into `[scipy.ndimage.gaussian_filter][]`

    Returns
    -------
    ndarray
        Filtered version of `image`.

    """
    from scipy.ndimage import gaussian_filter

    if np.sum(np.isnan(image)) == 0:
        return gaussian_filter(image, sigma=sigma, mode=mode, **kwargs)

    V = image.copy()
    nan_idxs = np.isnan(image)
    V[nan_idxs] = 0
    V_filt = gaussian_filter(V, sigma, **kwargs)

    W = np.ones(image.shape)
    W[nan_idxs] = 0
    W_filt = gaussian_filter(W, sigma, **kwargs)

    return V_filt / W_filt
