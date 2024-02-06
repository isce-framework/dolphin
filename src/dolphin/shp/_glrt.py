from __future__ import annotations

from math import log
from typing import Optional

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin._types import Strides
from dolphin.utils import compute_out_shape

from ._common import _make_loop_function, _read_cutoff_csv


@numba.njit(nogil=True)
def _compute_glrt_test_stat(scale_1, scale_2):
    """Compute the GLRT test statistic."""
    scale_pooled = (scale_1 + scale_2) / 2
    return 2 * log(scale_pooled) - log(scale_1) - log(scale_2)


_loop_over_pixels = _make_loop_function(_compute_glrt_test_stat)


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: Optional[dict] = None,
    alpha: float = 0.05,
    prune_disconnected: bool = False,
):
    """Estimate the number of neighbors based on the GLRT.

    Based on the method described in [@Parizzi2011AdaptiveInSARStack].
    Assumes Rayleigh distributed amplitudes ([@Siddiqui1962ProblemsConnectedRayleigh])

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel.
    var: ArrayLike, 2D
        Variance of each pixel's amplitude.
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    nslc : int
        Number of images in the stack used to compute `mean` and `var`.
        Used to compute the degrees of freedom for the t- and F-tests to
        determine the critical values.
    strides: dict, optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    alpha : float, default=0.05
        Significance level at which to reject the null hypothesis.
        Rejecting means declaring a neighbor is not a SHP.
    prune_disconnected : bool, default=False
        If True, keeps only SHPs that are 8-connected to the current pixel.
        Otherwise, any pixel within the window may be considered an SHP, even
        if it is not directly connected.


    Notes
    -----
    When `strides` is not (1, 1), the output first two dimensions
    are smaller than `mean` and `var` by a factor of `strides`. This
    will match the downstream shape of the strided phase linking results.

    Returns
    -------
    is_shp : np.ndarray, 4D
        Boolean array marking which neighbors are SHPs for each pixel in the block.
        Shape is (out_rows, out_cols, window_rows, window_cols), where
            `out_rows` and `out_cols` are computed by
            `[dolphin.io.compute_out_shape][]`
            `window_rows = 2 * halfwin_rowcol[0] + 1`
            `window_cols = 2 * halfwin_rowcol[1] + 1`

    """
    if strides is None:
        strides = {"x": 1, "y": 1}
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    threshold = get_cutoff(alpha=alpha, N=nslc)

    strides_rowcol = (strides["y"], strides["x"])
    out_rows, out_cols = compute_out_shape((rows, cols), Strides(*strides_rowcol))
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=np.bool_
    )
    return _loop_over_pixels(
        mean,
        var,
        halfwin_rowcol,
        strides_rowcol,
        threshold,
        prune_disconnected,
        is_shp,
    )


def get_cutoff(alpha: float, N: int) -> float:
    r"""Compute the upper cutoff for the GLRT test statistic.

    Statistic is

    \[
    2\log(\sigma_{pooled}) - \log(\sigma_{p}) -\log(\sigma_{q})
    \]

    Parameters
    ----------
    alpha: float
        Significance level (0 < alpha < 1).
    N: int
        Number of samples.

    Returns
    -------
    float
        Cutoff value for the GLRT test statistic.

    """
    n_alpha_to_cutoff = _read_cutoff_csv("glrt")
    try:
        return n_alpha_to_cutoff[(N, alpha)]
    except KeyError as e:
        msg = f"Not implemented for {N = }, {alpha = }"
        raise NotImplementedError(msg) from e
