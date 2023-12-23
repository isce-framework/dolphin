from typing import Sequence

import numpy as np

from dolphin.utils import flatten


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


def build_B_matrix(ifg_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Build the SBAS B (velocity coeff) matrix.

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[int, int]]
        List of ifg pairs representated as tuples of (index 1, index 2)

    Returns
    -------
    B : np.array 2D
        2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
        Each row corresponds to an ifg, each column to a SAR date
        value will be t_k+1 - t_k for columns after the -1 in A,
        up to and including the +1 entry
    """
    sar_dates = sorted(set(flatten(ifg_pairs)))
    A = get_incidence_matrix(ifg_pairs)
    B = np.zeros_like(A)
    timediffs = np.diff(sar_dates)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = 0
        for idx, item in enumerate(row):
            if item == -1:
                start_idx = idx + 1
            elif item == 1:
                end_idx = idx + 1

        # Now only fill in the time diffs in the range from the early ifg index
        # to the later ifg index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    return B
