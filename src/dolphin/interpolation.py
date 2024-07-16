import logging

import numba
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .similarity import get_circle_idxs

logger = logging.getLogger(__name__)


def interpolate(
    ifg: ArrayLike,
    weights: ArrayLike,
    weight_cutoff: float = 0.5,
    num_neighbors: int = 20,
    max_radius: int = 51,
    min_radius: int = 0,
    alpha: float = 0.75,
) -> np.ndarray:
    """Interpolate a complex interferogram based on pixel weights.

    Build upon persistent scatterer interpolation used in
    [@Chen2015PersistentScattererInterpolation] and
    [@Wang2022AccuratePersistentScatterer] by allowing floating-point weights
    instead of 0/1 PS weights.

    Parameters
    ----------
    ifg : np.ndarray, 2D complex array
        wrapped interferogram to interpolate
    weights : 2D float array
        Array of weights from 0 to 1 indicating how strongly to weigh
        the ifg values when interpolating.
        A special case of this is a PS mask where
            weights[i,j] = True if radar pixel (i,j) is a PS
            weights[i,j] = False if radar pixel (i,j) is not a PS
        Can also pass a coherence image to use as weights.
    weight_cutoff: float
        Threshold to use on `weights` so that pixels where
        `weight[i, j] < weight_cutoff` have phase values replaced by
        an interpolated value.
        The default is 0.5: pixels with weight less than 0.5 are replaced with a
        smoothed version of the surrounding pixels.
    num_neighbors: int (optional)
        number of nearest PS pixels used for interpolation
        num_neighbors = 20 by default
    max_radius : int (optional)
        maximum radius (in pixels) for PS searching
        max_radius = 51 by default
    min_radius : int (optional)
        minimum radius (in pixels) for PS searching
        max_radius = 0 by default
    alpha : float (optional)
        hyperparameter controlling the weight of PS in interpolation: smaller
        alpha means more weight is assigned to PS closer to the center pixel.
        alpha = 0.75 by default

    Returns
    -------
    interpolated_ifg : 2D complex array
        interpolated interferogram with the same amplitude, but different
        wrapped phase at non-ps pixels.

    """
    nrow, ncol = weights.shape

    weights_float = np.clip(weights.astype(np.float32), 0, 1)
    # Ensure weights are between 0 and 1
    if np.any(weights_float > 1):
        logger.warning("weights array has values greater than 1. Clipping to 1.")
    if np.any(weights_float < 0):
        logger.warning("weights array has negative values. Clipping to 0.")
    weights_float = np.clip(weights_float, 0, 1)

    interpolated_ifg = np.zeros((nrow, ncol), dtype=np.complex64)

    indices = np.array(
        get_circle_idxs(max_radius, min_radius=min_radius, sort_output=False)
    )

    _interp_loop(
        ifg,
        weights_float,
        weight_cutoff,
        num_neighbors,
        alpha,
        indices,
        interpolated_ifg,
    )
    return interpolated_ifg


@numba.njit(parallel=True)
def _interp_loop(
    ifg, weights, weight_cutoff, num_neighbors, alpha, indices, interpolated_ifg
):
    nrow, ncol = weights.shape
    nindices = len(indices)
    for r0 in numba.prange(nrow):
        for c0 in range(ncol):
            if weights[r0, c0] >= weight_cutoff:
                interpolated_ifg[r0, c0] = ifg[r0, c0]
                continue

            csum = 0.0 + 0j
            counter = 0
            r2 = np.zeros(num_neighbors, dtype=np.float64)
            cphase = np.zeros(num_neighbors, dtype=np.complex128)

            for i in range(nindices):
                idx = indices[i]
                r = r0 + idx[0]
                c = c0 + idx[1]

                if (
                    (r >= 0)
                    and (r < nrow)
                    and (c >= 0)
                    and (c < ncol)
                    and weights[r, c] >= weight_cutoff
                ):
                    # calculate the square distance to the center pixel
                    r2[counter] = idx[0] ** 2 + idx[1] ** 2

                    cphase[counter] = np.exp(1j * np.angle(ifg[r, c]))
                    counter += 1
                    if counter >= num_neighbors:
                        break

            # `counter` got up to one more than the number of elements
            # The last one will be the largest radius
            r2_norm = (r2[counter - 1] ** alpha) / 2
            for i in range(counter):
                csum += np.exp(-r2[i] / r2_norm) * cphase[i]

            interpolated_ifg[r0, c0] = np.abs(ifg[r0, c0]) * np.exp(1j * np.angle(csum))


def interpolate_along_axis(oldCoord, newCoord, data, axis=2):
    """Interpolate an array of 3-D data along one axis. This function.

    assumes that the x-coordinate increases monotonically.
    """
    if oldCoord.ndim > 1:
        stackedData = np.concatenate([oldCoord, data, newCoord], axis=axis)
        out = np.apply_along_axis(
            interp_vector, axis=axis, arr=stackedData, Nx=oldCoord.shape[axis]
        )
    else:
        out = np.apply_along_axis(
            interp_v,
            axis=axis,
            arr=data,
            old_x=oldCoord,
            new_x=newCoord,
            left=np.nan,
            right=np.nan,
        )

    return out


def interp_vector(vec, Nx):
    """Interpolate data from a single vector containing the original.

    x, the original y, and the new x, in that order. Nx tells the
    number of original x-points.
    """
    x = vec[:Nx]
    y = vec[Nx : 2 * Nx]
    xnew = vec[2 * Nx :]
    f = interp1d(x, y, bounds_error=False, copy=False, assume_sorted=True)
    return f(xnew)


def interp_v(y, old_x, new_x, left=None, right=None, period=None):
    """Rearrange np.interp's arguments."""
    return np.interp(new_x, old_x, y, left=left, right=right, period=period)


def fillna3d(array: np.ndarray, axis: int = -1, fill_value: float = 0.0) -> np.ndarray:
    """Fill in NaNs in 3D arrays using the nearest non-NaN value for "low" NaNs.

    and a specified fill value (default 0.0) for "high" NaNs.

    Parameters
    ----------
    array : np.ndarray
        3D array, where the last axis is the "z" dimension.
    axis : int, optional
        The axis along which to fill values. Default is -1 (the last axis).
    fill_value : float, optional
        The value used for filling NaNs. Default is 0.0.

    Returns
    -------
    np.ndarray
        3D array with low NaNs filled using nearest neighbors and high NaNs
        filled with the specified fill value.

    """
    # fill lower NaNs with nearest neighbor
    narr = np.moveaxis(array, axis, -1)
    nars = narr.reshape((np.prod(narr.shape[:-1]), narr.shape[-1]))
    dfd = pd.DataFrame(data=nars).interpolate(axis=1, limit_direction="backward")
    out = dfd.values.reshape(array.shape)

    # fill upper NaNs with 0s
    outmat = np.moveaxis(out, -1, axis)
    outmat[np.isnan(outmat)] = fill_value
    return outmat
