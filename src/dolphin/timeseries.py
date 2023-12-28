from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from dolphin.utils import flatten


@jax.jit
def solve(
    A: ArrayLike,
    dphi: np.ndarray,
) -> jax.Array:
    """Solve the SBAS problem for a list of ifg pairs and phase differences.

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[int, int]]
        List of ifg pairs representated as tuples of (index 1, index 2)
    dphi : np.array 1D
        The phase differences between the ifg pairs

    Returns
    -------
    phi : np.array 1D
        The estimated phase for each SAR acquisition
    """
    phi = jnp.linalg.lstsq(A, dphi, rcond=None)[0]
    # Add 0 for the reference date to the front
    return jnp.concatenate([jnp.array([0]), phi])


# Vectorize the solve function to work on 2D and 3D arrays
solve_2d = jax.vmap(solve, in_axes=(None, 1), out_axes=1)
solve_3d = jax.vmap(solve_2d, in_axes=(None, 2), out_axes=2)


def get_incidence_matrix(ifg_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Build the indiciator matrix from a list of ifg pairs (index 1, index 2).

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[int, int]]
        List of ifg pairs representated as tuples of (index 1, index 2)

    Returns
    -------
    A : np.array 2D
        The incident-like matrix from the SBAS paper: A*phi = dphi
        Each row corresponds to an ifg, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
    """
    sar_dates = sorted(set(flatten(ifg_pairs)))

    M = len(ifg_pairs)
    N = len(sar_dates) - 1
    A = np.zeros((M, N))

    # Create a dictionary mapping sar dates to matrix columns
    # We take the first SAR acquisition to be time 0, leave out of matrix
    date_to_col = {date: i for i, date in enumerate(sar_dates[1:])}
    # Populate the matrix
    for i, (early, later) in enumerate(ifg_pairs):
        if early in date_to_col:
            A[i, date_to_col[early]] = -1
        if later in date_to_col:
            A[i, date_to_col[later]] = +1

    return A
