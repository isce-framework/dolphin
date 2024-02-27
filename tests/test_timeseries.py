import itertools
from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.linalg import lstsq as lstsq_numpy

from dolphin import timeseries

NUM_DATES = 10
DT = 12
START_DATE = datetime(2020, 1, 1)
SHAPE = 50, 50


def make_sar_dates():
    sar_date_list = []
    for idx in range(NUM_DATES):
        sar_date_list.append(START_DATE + timedelta(days=idx * DT))
    return sar_date_list


def make_sar_phases(sar_dates):
    out = np.empty((NUM_DATES, *SHAPE), dtype="float32")
    # Make a linear ramp which increases each time interval
    slope = 0.2  # rad / day
    ramp0 = np.ones((SHAPE[0], 1)) * np.arange(SHAPE[1]).reshape(1, -1)
    ramp0 /= SHAPE[1]
    for idx in range(len(sar_dates)):
        cur_slope = slope * DT * idx
        out[idx] = cur_slope * ramp0
    return out


def make_ifg_date_pairs(sar_dates):
    """Form all possible unwrapped interferogram pairs from the date list"""
    return [tuple(pair) for pair in itertools.combinations(sar_dates, 2)]


def make_ifgs(sar_phases):
    """Form all possible unwrapped interferogram pairs from the sar images"""
    return np.stack(
        [(ref - sec) for (ref, sec) in itertools.combinations(sar_phases, 2)]
    )


@pytest.fixture(scope="module")
def data():
    sar_dates = make_sar_dates()
    sar_phases = make_sar_phases(sar_dates)
    ifg_date_pairs = make_ifg_date_pairs(sar_dates)
    ifgs = make_ifgs(sar_phases)
    return sar_dates, sar_phases, ifg_date_pairs, ifgs


def test_incidence_matrix():
    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert A.shape == (5, 5)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ]
    )
    np.testing.assert_array_equal(A, expected)


def solve_with_removal(A, b):
    good_idxs = (~np.isnan(b)).flatten()
    b_clean = b[good_idxs]
    A_clean = A[good_idxs, :]
    return lstsq_numpy(A_clean, b_clean)[0]


def solve_by_zeroing(A, b):
    weight = (~np.isnan(b)).astype(float)
    A0 = A.copy()
    A0 = A * weight[:, None]
    b2 = np.nan_to_num(b)
    return lstsq_numpy(A0, b2)[0]


def test_solve(data):
    sar_dates, sar_phases, ifg_date_pairs, ifgs = data

    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    # Get some
    dphi = ifgs[:, 0, 0]
    phi = timeseries.solve(A, dphi)
    assert phi.shape == (sar_dates - 1)


if __name__ == "__main__":
    # import the fixtures
    sar_dates = make_sar_dates()
    sar_phases = make_sar_phases(sar_dates)
    ifg_date_pairs = make_ifg_date_pairs(sar_dates)
    ifgs = make_ifgs(sar_phases)
