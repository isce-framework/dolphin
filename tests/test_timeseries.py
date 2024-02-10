import numpy as np
import pytest
import troposim.igrams
from numpy.linalg import lstsq as lstsq_numpy

from dolphin import timeseries

NUM_DATES = 50
DT = 12


@pytest.fixture(scope="module")
def ifg_maker():
    igm = troposim.igrams.IgramMaker(repeat_interval_days=DT)
    igm.make_sar_stack()
    return igm


@pytest.fixture(scope="module")
def igrams_full(igm):
    igram_stack, igram_date_list = igm.make_igram_stack(max_temporal_baseline=5000)
    return igram_stack, igram_date_list


@pytest.fixture(scope="module")
def igrams_nearest3(igm):
    igram_stack, igram_date_list = igm.make_igram_stack(max_temporal_baseline=3 * DT)
    return igram_stack, igram_date_list


def test_incidence_matrix():
    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert A.shape == (5, 5)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
            [0, 0, 0, -1],
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


def test_solve(ifg_maker: troposim.igrams.IgramMaker, igrams_full):
    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    phi = troposim.igrams.solve(A, dphi)
    assert phi.shape == (ifg_maker.num_sar_dates,)
