import numpy as np
from numba import njit, stencil
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

TWOPI = 2 * np.pi


@njit(nogil=True)
def compute_phase_diffs(phase):
    """Compute the total number phase jumps > pi between adjacent pixels.

    If part of `phase` is known to be bad phase (e.g. over water),
    the values should be set to zero or a masked array should be passed:

        unwrapping_error(np.ma.masked_where(bad_area_mask, phase))


    Parameters
    ----------
    phase : ArrayLike
        Unwrapped interferogram phase.

    Returns
    -------
    int
        Total number of jumps exceeding pi.

    """
    return _compute_phase_diffs(phase)


@stencil
def _compute_phase_diffs(phase):
    d1 = np.abs(phase[0, 0] - phase[0, 1]) / np.pi
    d2 = np.abs(phase[0, 0] - phase[1, 0]) / np.pi
    # Subtract 0.5 so that anything below 1 gets rounded to 0
    return round(d1 - 0.5) + round(d2 - 0.5)


def rewrap_to_twopi(arr: ArrayLike) -> np.ndarray:
    """Rewrap `arr` to be between [-pi and pi].

    Parameters
    ----------
    arr : ArrayLike
        Unwrapped floating point phase

    Returns
    -------
    np.ndarray
        Array of phases between -pi and pi

    """
    return np.mod(np.pi + arr, TWOPI) - np.pi


def gaussian_filter_nan(
    image: ArrayLike, sigma: float, mode="constant", **kwargs
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


def _get_ambiguities(unw: ArrayLike, round_decimals: int = 4) -> np.ndarray:
    mod_2pi_image = np.mod(np.pi + unw, TWOPI) - np.pi
    re_wrapped = np.round(mod_2pi_image, round_decimals)
    return np.round((unw - re_wrapped) / (TWOPI), round_decimals - 1)


def _fill_masked_ambiguities(
    amb_image: ArrayLike, mask: ArrayLike, filter_sigma: int = 60
) -> np.ndarray:
    masked_ambs = amb_image.copy()
    masked_ambs[mask] = np.nan
    ambs_filled = np.round(gaussian_filter_nan(amb_image, filter_sigma))

    out_filled = amb_image.copy()
    out_filled[mask] = ambs_filled[mask]
    return out_filled


def _smooth_masked_areas(unw, mask, filter_sigma: int = 60):
    amb = _get_ambiguities(unw)
    amb_filled = _fill_masked_ambiguities(amb, mask, filter_sigma=filter_sigma)

    out = unw.copy()
    rewrapped_phase_vec = np.mod(np.pi + unw[mask], TWOPI) - np.pi
    new_amb_vec = amb_filled[mask]
    out[mask] = rewrapped_phase_vec + (new_amb_vec * TWOPI)
    return out
