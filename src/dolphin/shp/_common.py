from __future__ import annotations

import numba
import numpy as np
from numpy.typing import ArrayLike


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
