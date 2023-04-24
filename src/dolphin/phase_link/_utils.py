import numba


@numba.njit
def _get_slices(half_r: int, half_c: int, r: int, c: int, rows: int, cols: int):
    """Get the slices for the given pixel and half window size."""
    # Clamp min indexes to 0
    r_start = max(r - half_r, 0)
    c_start = max(c - half_c, 0)
    # Clamp max indexes to the array size
    r_end = min(r + half_r + 1, rows)
    c_end = min(c + half_c + 1, cols)
    return (r_start, r_end), (c_start, c_end)
