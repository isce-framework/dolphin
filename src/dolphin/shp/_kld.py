from __future__ import annotations

from typing import Optional

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin._types import Strides
from dolphin.utils import compute_out_shape

from ._common import _make_loop_function, _read_cutoff_csv


@numba.njit(nogil=True)
def _kld(sigma2_p, sigma2_q):
    """Compute KL divergence of two Rayleigh PDFs given their scale parameters."""
    return np.log(sigma2_q / sigma2_p) + (sigma2_p / sigma2_q) - 1


@numba.njit(nogil=True)
def _compute_test_stat_js(sigma2_p, sigma2_q):
    """Compute the Jensen-Shannon divergence."""
    return (_kld(sigma2_p, sigma2_q) + _kld(sigma2_q, sigma2_p)) / 2


_loop_over_pixels = _make_loop_function(_compute_test_stat_js)


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: Optional[dict] = None,
    alpha: float = 0.05,
    prune_disconnected: bool = False,
):
    """Estimate the number of neighbors using the Jensen-Shannon.

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

    threshold = get_cutoff(nslc, alpha)

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


def get_cutoff(N: int, alpha: float) -> float:
    """Get the pre-computed test statistic cutoff.

    Parameters
    ----------
    N : int
        number of SLCs used for mean/variance.
    alpha : float
        Significance level

    Returns
    -------
    float
        Threshold, above which to reject the null hypothesis.

    Raises
    ------
    ValueError
        If a (N, alpha) combination is passed which hasn't been precomputed.

    """
    n_alpha_to_cutoff = _read_cutoff_csv("kld")
    try:
        return n_alpha_to_cutoff[(N, alpha)]
    except KeyError as e:
        msg = f"Not implemented for {N = }, {alpha = }"
        raise NotImplementedError(msg) from e
