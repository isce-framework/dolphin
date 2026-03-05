from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def goldstein(
    phase: NDArray[np.complex64] | NDArray[np.float64], alpha: float, psize: int = 32
) -> np.ndarray:
    """Apply the Goldstein adaptive filter to the given data.

    Parameters
    ----------
    phase : np.ndarray
        2D array of floating point phase, or complex data, to be filtered.
    alpha : float
        Filtering parameter for Goldstein algorithm
        Must be between 0 (no filtering) and 1 (maximum filtering)
    psize : int, optional
        edge length of square patch
        Default = 32

    Returns
    -------
        2D numpy array of filtered data.

    References
    ----------
    [@Goldstein1998RadarInterferogramFiltering]

    """

    def apply_pspec(data: NDArray[np.complex64]) -> np.ndarray:
        # NaN is allowed value
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha = }")

        weight = np.power(np.abs(data) ** 2, alpha / 2)
        data = weight * data
        return data

    def make_weight(nxp: int, nyp: int) -> np.ndarray:
        # Create arrays of horizontal and vertical weights
        wx = 1.0 - np.abs(np.arange(nxp // 2) - (nxp / 2.0 - 1.0)) / (nxp / 2.0 - 1.0)
        wy = 1.0 - np.abs(np.arange(nyp // 2) - (nyp / 2.0 - 1.0)) / (nyp / 2.0 - 1.0)
        # Compute the outer product of wx and wy to create
        # the top-left quadrant of the weight matrix
        quadrant = np.outer(wy, wx)
        # Create a full weight matrix by mirroring the quadrant along both axes
        weight = np.block(
            [
                [quadrant, np.flip(quadrant, axis=1)],
                [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)],
            ]
        )
        return weight

    def patch_goldstein_filter(
        data: NDArray[np.complex64], weight: NDArray[np.float64], psize: int
    ) -> np.ndarray:
        """Apply the filter to a single patch of data.

        Parameters
        ----------
        data : np.ndarray
            2D complex array containing the data to be filtered.
        weight : np.ndarray
            weight matrix for summing neighboring data
        psize : int
            edge length of square FFT area

        Returns
        -------
            2D numpy array of filtered data.

        """
        # Calculate alpha
        data = np.fft.fft2(data, s=(psize, psize))
        data = apply_pspec(data)
        data = np.fft.ifft2(data, s=(psize, psize))
        return weight * data

    def apply_goldstein_filter(data: NDArray[np.complex64]) -> np.ndarray:
        # Mark invalid pixels (NaN or zero magnitude)
        empty_mask = np.isnan(data) | (data == 0)
        # ignore processing for empty chunks
        if np.all(empty_mask):
            return data

        nrows, ncols = data.shape
        step = psize // 2

        # Pad on all sides with reflection to handle edges without artifacts.
        # Padding of step ensures original (0,0) gets weight from overlapping windows.
        pad_top = step
        pad_left = step
        pad_bottom = step + (step - (nrows % step)) % step
        pad_right = step + (step - (ncols % step)) % step
        data_padded = np.pad(
            data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
        )

        # Create output arrays matching padded size
        out = np.zeros(data_padded.shape, dtype=np.complex64)
        weight_sum = np.zeros(data_padded.shape, dtype=np.float64)
        weight_matrix = make_weight(psize, psize)

        # Iterate over windows using full psize windows
        padded_rows, padded_cols = data_padded.shape
        for i in range(0, padded_rows - psize + 1, step):
            for j in range(0, padded_cols - psize + 1, step):
                data_window = data_padded[i : i + psize, j : j + psize]
                filtered_window = patch_goldstein_filter(
                    data_window, weight_matrix, psize
                )
                out[i : i + psize, j : j + psize] += filtered_window
                weight_sum[i : i + psize, j : j + psize] += weight_matrix

        # Normalize by accumulated weights
        valid = weight_sum > 0
        out[valid] /= weight_sum[valid]

        # Crop back to original size and apply empty mask
        out = out[pad_top : pad_top + nrows, pad_left : pad_left + ncols]
        out[empty_mask] = 0
        return out

    if np.iscomplexobj(phase):
        return apply_goldstein_filter(phase)
    else:
        return apply_goldstein_filter(np.exp(1j * phase))
