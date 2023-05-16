from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin.utils import _get_slices

# https://numba.readthedocs.io/en/stable/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function


def _make_loop_function(
    _compute_test_stat: Callable,
):
    @numba.njit(nogil=True, parallel=True)
    def _loop_over_pixels(
        mean: ArrayLike,
        var: ArrayLike,
        halfwin_rowcol: tuple[int, int],
        strides_rowcol: tuple[int, int],
        threshold: float,
        is_shp: np.ndarray,
    ) -> np.ndarray:
        """Loop common to SHP tests using only mean and variance."""
        half_row, half_col = halfwin_rowcol
        row_strides, col_strides = strides_rowcol
        # location to start counting from in the larger input
        r0, c0 = row_strides // 2, col_strides // 2
        in_rows, in_cols = mean.shape
        out_rows, out_cols = is_shp.shape[:2]

        # Convert mean/var to the Rayleigh scale parameter
        scale_squared = (var + mean**2) / 2

        for out_r in numba.prange(out_rows):
            for out_c in range(out_cols):
                in_r = r0 + out_r * row_strides
                in_c = c0 + out_c * col_strides

                scale_1 = scale_squared[in_r, in_c]
                # Clamp the window to the image bounds
                (r_start, r_end), (c_start, c_end) = _get_slices(
                    half_row, half_col, in_r, in_c, in_rows, in_cols
                )

                for in_r2 in range(r_start, r_end):
                    for in_c2 in range(c_start, c_end):
                        # window offsets for dims 3,4 of `is_shp`
                        r_off = in_r2 - r_start
                        c_off = in_c2 - c_start

                        # itself is always a neighbor
                        if in_r2 == in_r and in_c2 == in_c:
                            is_shp[out_r, out_c, r_off, c_off] = True
                            continue
                        scale_2 = scale_squared[in_r2, in_c2]

                        T = _compute_test_stat(scale_1, scale_2)

                        is_shp[out_r, out_c, r_off, c_off] = T < threshold

        return is_shp

    return _loop_over_pixels


@lru_cache
def _read_cutoff_csv(test_name):
    # Replace with the actual filename
    filename = Path(__file__).parent / f"{test_name}_cutoffs.csv"

    result = {}
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            n = int(row["N"])
            alpha = float(row["alpha"])
            cutoff = float(row["cutoff"])
            result[(n, alpha)] = cutoff

    return result
