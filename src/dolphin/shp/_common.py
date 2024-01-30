from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin.utils import _get_slices

_get_slices = numba.njit(_get_slices)


@numba.njit(nogil=True)
def remove_unconnected(data: ArrayLike, inplace: bool = True) -> np.ndarray:
    """Remove groups of True values in a boolean matrix not connected to the center.

    Parameters
    ----------
    data : ArrayLike
        2D input boolean matrix.
    inplace : bool, default True
        If True, modifies the input array directly. Otherwise, creates a copy.

    Returns
    -------
    np.ndarray
        Boolean matrix with only the connected group of True values centered
        around the middle pixel.

    Notes
    -----
    This function considers the 8 surrounding neighbors as connected
        (i.e. includes diagonals.)
    """
    if not inplace:
        data = data.copy()
    rows, cols = data.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)
    # Using a stack instead of recursive calls for speed
    # Pre-allocate the stack
    max_stack_size = 1 + rows * cols
    stack = np.zeros((3 * max_stack_size, 2), dtype=np.int16)
    stack_ptr = 0  # Stack pointer to keep track of the top of the stack

    # Start at the center and search outward.
    start_row, start_col = rows // 2, cols // 2
    visited[start_row, start_col] = True
    stack[stack_ptr] = start_row, start_col
    stack_ptr += 1

    connected_idxs = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Run the Depth First Search to get connected pixels
    while stack_ptr > 0:
        stack_ptr -= 1
        x, y = stack[stack_ptr]

        for dx, dy in connected_idxs:
            x2, y2 = x + dx, y + dy
            # Skip OOB pixels
            if x2 < 0 or x2 >= rows or y2 < 0 or y2 >= cols:
                continue
            # Check we haven't visited:
            if visited[x2, y2]:
                continue
            # Now push to stack if this pixel is True
            if data[x2, y2]:
                visited[x2, y2] = True
                stack[stack_ptr] = x2, y2
                stack_ptr += 1

    # Finally only keep ones from the original that were visited
    data &= visited
    return data


# Factory function to make a JIT-ed function calling `compute_test_stat`
# https://numba.readthedocs.io/en/stable/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function


def _make_loop_function(
    compute_test_stat: Callable,
):
    """Create a JIT-ed function computing a test statistic for each pixel.

    `compute_test_stat` should have the signature:

    compute_test_stat(rayleigh_scale1: float, rayleigh_scale2: float) -> float
    """

    @numba.njit(nogil=True, parallel=True)
    def _loop_over_pixels(
        mean: ArrayLike,
        var: ArrayLike,
        halfwin_rowcol: tuple[int, int],
        strides_rowcol: tuple[int, int],
        threshold: float,
        prune_disconnected: bool,
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

                        T = compute_test_stat(scale_1, scale_2)

                        is_shp[out_r, out_c, r_off, c_off] = threshold > T
                if prune_disconnected:
                    # For this pixel, prune the groups not connected to the center
                    remove_unconnected(is_shp[out_r, out_c], inplace=True)

        return is_shp

    return _loop_over_pixels


@lru_cache
def _read_cutoff_csv(test_name):
    # Replace with the actual filename
    filename = Path(__file__).parent / f"{test_name}_cutoffs.csv"

    result = {}
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            n = int(row["N"])
            alpha = float(row["alpha"])
            cutoff = float(row["cutoff"])
            result[(n, alpha)] = cutoff

    return result
