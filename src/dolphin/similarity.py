"""Module for computing phase similarity between complex interferogram pixels.

Uses metric from @[Wang2022AccuratePersistentScatterer] for similarity.
"""

import numba
import numpy as np
from numpy.typing import ArrayLike

# from dolphin.io import iter_blocks


@numba.njit
def phase_similarity(x1: ArrayLike, x2: ArrayLike):
    """Compute the similarity between two complex 1D vectors."""
    n = len(x1)
    out = 0.0
    for i in range(n):
        out += np.real(x1[i] * np.conj(x2[i]))
    return out / n


def median_similarity(
    ifg_stack: ArrayLike, search_radius: int, weights: ArrayLike | None = None
):
    """Compute the median similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (5) of @[Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
        max_radius = 51 by default
    weights: ArrayLike (optional)
        Array of weights from 0 to 1 indicating how strongly to weigh
        the ifg values when interpolating.
        A special case of this is a PS mask where
            weights[i,j] = True if radar pixel (i,j) is a PS
            weights[i,j] = False if radar pixel (i,j) is not a PS

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the median similarity at each pixel.

    """
    n_ifg, rows, cols = ifg_stack.shape
    if not np.iscomplexobj(ifg_stack):
        raise ValueError("ifg_stack must be complex")

    unit_ifgs = np.exp(1j * np.angle(ifg_stack))
    out_similarity = np.zeros((rows, cols), dtype="float32")
    if weights is None:
        weights = np.ones((rows, cols), dtype="float32")
    idxs = get_circle_idxs(search_radius)
    return _median_sim_loop(unit_ifgs, idxs, weights, out_similarity)


@numba.njit(nogil=True, parallel=True)
def _median_sim_loop(
    ifg_stack: np.ndarray,
    idxs: np.ndarray,
    weights: np.ndarray,
    out_similarity: np.ndarray,
) -> np.ndarray:
    """Loop common to SHP tests using only mean and variance."""
    _, rows, cols = ifg_stack.shape

    num_compare_pixels = len(idxs)
    # Buffer to hold all comparison during the parallel loop
    cur_sim = np.zeros((rows, cols, num_compare_pixels))

    for r0 in numba.prange(rows):
        for c0 in range(cols):
            # Get the current pixel
            x0 = ifg_stack[:, r0, c0]
            cur_sim_vec = cur_sim[r0, c0]

            w_sum = 0.0
            # compare to all pixels in the circle around it
            for i_idx in range(num_compare_pixels):
                ir, ic = idxs[i_idx]
                # Clip to the image bounds
                r = max(min(r0 + ir, rows - 1), 0)
                c = max(min(c0 + ic, cols - 1), 0)

                x = ifg_stack[:, r, c]
                w = weights[r, c]

                cur_sim_vec[i_idx] = w * phase_similarity(x0, x)
                w_sum += w

            # Scale back based on the total weight
            scale = num_compare_pixels / w_sum
            out_similarity[r0, c0] = np.median(cur_sim_vec * scale)
    return out_similarity


def get_circle_idxs(max_radius: int, min_radius: int = 0) -> np.ndarray:
    """Get the relative indices of neighboring pixels in a circle.

    Adapted from c++ version of `psps` package:
    https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp#L16
    """
    # using the mid-point circle drawing algorithm to search for neighboring PS pixels
    # # code adapted from "https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/"
    visited = np.zeros((max_radius, max_radius), dtype=bool)
    visited[0][0] = True

    indices = []
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1

            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    # Sorting makes it run faster, better data access patterns
    return np.array(sorted(indices))
